import argparse, json, os, math, subprocess, tempfile
import numpy as np
from moviepy import (
    ImageClip, AudioFileClip, CompositeAudioClip,
    CompositeVideoClip, ColorClip, VideoClip, concatenate_videoclips
)
from PIL import Image, ImageDraw, ImageFont
from edge_tts_helper import tts_edge

# ---------- font helper (Windows/Linux) ----------
def get_font(font_size: int):
    candidates = [
        os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "segoeui.ttf"),
        os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "arial.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, font_size)
            except Exception:
                pass
    return ImageFont.load_default()

# ---------- helpers ----------
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def hex_to_rgb(hx):
    hx = hx.lstrip("#")
    return (int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16))

def read_odd_even_groups(csv_path, pairs_per_group=5):
    """
    Одноколонковий CSV:
      1-й рядок — заголовок (може бути 'text' із BOM),
      далі слова блоками по 10 рядків:
        непарні (позиції 1,3,5,7,9) -> зліва
        парні  (позиції 2,4,6,8,10) -> справа
    Повертає список груп: [(left5, right5), ...]
    """
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    if raw.startswith("\ufeff"):
        raw = raw.lstrip("\ufeff")

    lines = [ln.strip().strip(",") for ln in raw.splitlines()]
    lines = [ln for ln in lines if ln != ""]
    if not lines:
        raise ValueError("CSV порожній або не читається.")

    first = lines[0].lower()
    if first in {"text","word","словo","слово"} or "text" in first:
        lines = lines[1:]

    words = lines
    chunk = pairs_per_group * 2  # 10
    groups = []
    for start in range(0, len(words), chunk):
        block = words[start:start+chunk]
        if len(block) < chunk:
            break
        # odd/even розкладка:
        left  = block[0::2]  # 0,2,4,6,8  -> 1,3,5,7,9
        right = block[1::2]  # 1,3,5,7,9  -> 2,4,6,8,10
        groups.append((left, right))
    if not groups:
        raise ValueError("Не знайшов жодної повної групи по 10 рядків у CSV.")
    return groups

def rgba_image_to_clip(img_rgba, position, duration):
    arr = np.array(img_rgba)
    if arr.ndim == 3 and arr.shape[2] == 4:
        rgb = arr[:, :, :3]
        alpha = (arr[:, :, 3].astype(float) / 255.0)
        base = ImageClip(rgb).with_duration(duration)
        mask = VideoClip(lambda t: alpha).with_duration(duration)
        return base.with_mask(mask).with_position(position)
    else:
        return ImageClip(arr).with_position(position).with_duration(duration)

def make_column_image_fixed_rows(lines, n_rows, font_size, width_px, color_hex, line_spacing=1.25, pad_y=10):
    """Рівно n_rows рядків (порожні теж) — ідеальне вирівнювання рядок-до-рядка."""
    font = get_font(font_size)
    ascent, descent = font.getmetrics()
    line_h = ascent + descent
    row_h = int(line_h * line_spacing)
    total_h = n_rows * row_h + 2 * pad_y

    img = Image.new("RGBA", (width_px, total_h), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    r,g,b = hex_to_rgb(color_hex); fill = (r,g,b,255)

    for i in range(n_rows):
        txt = lines[i] if i < len(lines) else ""
        bbox = draw.textbbox((0, 0), txt, font=font)
        w_px, h_px = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (width_px - w_px)//2
        y = pad_y + i*row_h + (row_h - h_px)//2
        draw.text((x,y), txt, font=font, fill=fill)
    return img

def ring_image(w, h, frac, center, radius, thickness, color_rgb, bg_opacity=0.3):
    img = Image.new("RGBA", (w,h), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    cx, cy = center
    bbox = [cx-radius, cy-radius, cx+radius, cy+radius]
    draw.ellipse(bbox, outline=(255,255,255,int(255*bg_opacity)), width=thickness)
    if frac > 0:
        end = 360*frac
        r,g,b = color_rgb
        draw.arc(bbox, start=-90, end=-90+end, width=thickness, fill=(r,g,b,255))
    return img

def digit_image(w, h, text, center_y, font_size, color_hex):
    img = Image.new("RGBA", (w,h), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    font = get_font(font_size)
    r,g,b = hex_to_rgb(color_hex); fill = (r,g,b,255)
    bbox = draw.textbbox((0, 0), text, font=font)
    w_px, h_px = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (w - w_px)//2
    y = int(center_y - h_px*0.5)
    draw.text((x,y), text, font=font, fill=fill)
    return img

def make_timer_clip(W,H,dur,center,radius,thickness,accent_rgb,bg_opacity, number_center_y, font_size, text_color):
    # ТІЛЬКИ КІЛЬЦЕ, БЕЗ ЦИФР

    def ring_frame(t):
        frac = min(1.0, max(0.0, t/dur))
        img = ring_image(W,H,frac,center,radius,thickness,accent_rgb,bg_opacity)
        return np.array(img)[:, :, :3]  # RGB

    def ring_mask(t):
        frac = min(1.0, max(0.0, t/dur))
        img = ring_image(W,H,frac,center,radius,thickness,accent_rgb,bg_opacity)
        return np.array(img)[:, :, 3].astype(float)/255.0  # 0..1

    ring_clip = VideoClip(ring_frame).with_duration(dur)
    ring_mask_clip = VideoClip(ring_mask).with_duration(dur)
    ring_clip = ring_clip.with_mask(ring_mask_clip)
    return ring_clip

def tts_edge_uk(text, out_path):
    # Генерує MP3 українським голосом
    tts_edge(text, "uk", out_path)

def tts_edge_multi(text, lang_code, out_path):
    # Універсальна озвучка: lang_code = 'uk' | 'es' | 'ru' | 'de' | 'fr' | 'pt' | 'tr'
    tts_edge(text, lang_code, out_path)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="inputs/seq1.csv")   # 1 колонка: odd->left, even->right (10 рядків на групу)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--out", default="outputs/lesson1.mp4")
    ap.add_argument("--stage_seconds", type=float, default=3.5, help="крок відкриття перекладу (сек.)")
    ap.add_argument("--hold_seconds",  type=float, default=1.5, help="утримання всіх 10 слів після 5-ї пари (мінімум, сек.)")
    ap.add_argument("--end_hold_seconds", type=float, default=1.0, help="утримання після останньої групи (мінімум, сек.)")
    ap.add_argument("--intro_delay",   type=float, default=1.5, help="затримка на старті (фон+аватар), сек")
    ap.add_argument("--tts_lang_right", default="uk", help="Код мови для озвучки правої колонки (uk, es, ru, de, fr, pt, tr)")
    args = ap.parse_args()

    C = load_config(args.config)
    W,H = C["width"], C["height"]; FPS = C["fps"]

    print(">>> MODE: odd/even groups of 10 (1,3,5,7,9 left; 2,4,6,8,10 right), intro 1.5s, 5s steps, proper last TTS hold")

    # фон і аватар
    bg = ColorClip(size=(W,H), color=hex_to_rgb(C["bg_color"])).with_duration(0.1)
    avatar_img = Image.open(C["avatar"]).convert("RGBA")
    aw = int(W*C["avatar_scale"]); ah = int(avatar_img.height * (aw/avatar_img.width))
    avatar_img = avatar_img.resize((aw,ah), resample=Image.LANCZOS)
    avatar_pos = ("center", int(H*C["avatar_y"] - ah/2))

    # геометрія
    cols_top_y = int(H*C["cols_top_y"])
    col_w = int(W*C["col_width"]); margin = int(W*C["col_margin"])
    left_x = margin; right_x = W - margin - col_w

    font_size = int(C.get("font_size",56)); text_color = C.get("text_color","#ffffff")

    # таймер
    accent_rgb = hex_to_rgb(C["accent_color"])
    timer_center = (W//2, int(H*C["timer_top_y"]))
    timer_radius = C["timer_radius"]; timer_thickness = C["timer_thickness"]
    timer_bg_opacity = C["timer_bg_opacity"]
    number_center_y = int(H*C["timer_top_y"] - C["timer_radius"]*0.15)
    num_font_size = C["timer_font_size"]

    # дані
    groups = read_odd_even_groups(args.csv, pairs_per_group=5)
    print(f">>> Loaded groups: {len(groups)}")

    stage = args.stage_seconds
    base_hold = max(0.001, args.hold_seconds)
    end_hold  = max(0.001, args.end_hold_seconds)

    clips = []; audio_segments = []
    t_cursor = 0.0  # відео-таймлайн
    tmpdir = tempfile.mkdtemp(prefix="tts_batch_")

    # інтро 1.5 с (тільки фон + аватар)
    intro = max(0.0, args.intro_delay)
    if intro > 0:
        avatar_intro = rgba_image_to_clip(avatar_img, avatar_pos, intro)
        comp_intro = CompositeVideoClip([bg.with_duration(intro), avatar_intro]).with_duration(intro)
        clips.append(comp_intro)
        t_cursor += intro

    total_groups = len(groups)
    for gi, (left_words, right_words) in enumerate(groups):
        n = len(left_words)  # 5
        # Ліві слова (1,3,5,7,9) малюємо на всю групу
        left_block_img = make_column_image_fixed_rows(left_words, n, font_size, col_w, text_color)

        # КРОК 0: 5с таймера + ліві 5, справа — порожньо
        right_empty_img = make_column_image_fixed_rows([], n, font_size, col_w, text_color)
        left_block_clip0  = rgba_image_to_clip(left_block_img,  (left_x, cols_top_y), stage)
        right_block_clip0 = rgba_image_to_clip(right_empty_img, (right_x, cols_top_y), stage)
        avatar_clip0      = rgba_image_to_clip(avatar_img,      avatar_pos,          stage)
        ring0 = make_timer_clip(W,H,stage, timer_center,timer_radius,timer_thickness,
                                accent_rgb,timer_bg_opacity, number_center_y,num_font_size,text_color)
        comp0 = CompositeVideoClip([
            bg.with_duration(stage),
            avatar_clip0,
            left_block_clip0,
            right_block_clip0,
            ring0
        ]).with_duration(stage)
        clips.append(comp0)
        t_cursor += stage

        # КРОКИ 1..5: кожні 5с додаємо переклад справа; озвучка стартує в момент появи
        last_vo_dur = 0.0
        for i in range(n):
            is_last = (i == n-1)
            dur_video = stage if not is_last else 0.001  # 5-е — миттєво; HOLD покажемо окремо

            accumulated = right_words[:i+1]
            right_block_img = make_column_image_fixed_rows(accumulated, n, font_size, col_w, text_color)
            right_block_clip = rgba_image_to_clip(right_block_img, (right_x, cols_top_y), dur_video)
            left_block_clip  = rgba_image_to_clip(left_block_img,  (left_x, cols_top_y), dur_video)
            avatar_clipi     = rgba_image_to_clip(avatar_img,      avatar_pos,          dur_video)

            layers = [bg.with_duration(dur_video), avatar_clipi, left_block_clip, right_block_clip]
            if not is_last:
                ringi = make_timer_clip(W,H,dur_video, timer_center,timer_radius,timer_thickness,
                                        accent_rgb,timer_bg_opacity, number_center_y,num_font_size,text_color)
                layers += [ringi]
            compi = CompositeVideoClip(layers).with_duration(dur_video)
            clips.append(compi)

            # TTS: для 1..4 стартує на t_cursor (момент появи); для 5-го — в HOLD, одразу після цього 0.001с кадру
            wav_path = os.path.join(tmpdir, f"uk_g{gi}_i{i}.mp3")
            try:
                tts_edge_multi(right_words[i], args.tts_lang_right, wav_path)
                if is_last:
                    seg = AudioFileClip(wav_path).with_start(t_cursor + dur_video)  # HOLD починається після цього кадру
                    last_vo_dur = seg.duration or 0.0
                else:
                    seg = AudioFileClip(wav_path).with_start(t_cursor)
                audio_segments.append(seg)
            except Exception as e:
                print("TTS error:", e)

            t_cursor += dur_video

        # HOLD: показати всі 10 слів
        base = base_hold if gi < total_groups-1 else end_hold
        group_hold = max(base, last_vo_dur)  # тримаємо мінімум base, але не менше за тривалість озвучки 5-го
        right_full_img  = make_column_image_fixed_rows(right_words, n, font_size, col_w, text_color)
        right_full_clip = rgba_image_to_clip(right_full_img, (right_x, cols_top_y), group_hold)
        left_full_clip  = rgba_image_to_clip(left_block_img, (left_x, cols_top_y), group_hold)
        avatar_hold     = rgba_image_to_clip(avatar_img,     avatar_pos,           group_hold)
        comp_hold = CompositeVideoClip([bg.with_duration(group_hold), avatar_hold, left_full_clip, right_full_clip]).with_duration(group_hold)
        clips.append(comp_hold)
        t_cursor += group_hold

    # фінал: відео + звук
    video = concatenate_videoclips(clips, method="compose")
    if audio_segments:
        final_audio = CompositeAudioClip(audio_segments).with_duration(video.duration)
        video = video.with_audio(final_audio)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    video.write_videofile(args.out, fps=FPS, codec="libx264", audio_codec="aac", preset="medium", threads=4)

if __name__ == "__main__":
    print(">>> MODE ACTIVE: odd/even by 10, intro 1.5s, 5s stages, last TTS during full-10 hold")
    main()
