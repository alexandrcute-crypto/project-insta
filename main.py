import argparse, json, os, csv, math, subprocess, tempfile
import numpy as np
from moviepy.editor import (
    ImageClip, AudioFileClip, CompositeAudioClip,
    CompositeVideoClip, ColorClip, concatenate_videoclips
)
from PIL import Image, ImageDraw, ImageFont


# -------------------- УТИЛІТИ --------------------

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def hex_to_rgb(hx):
    hx = hx.lstrip("#")
    return (int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16))

def read_sequence_to_groups(csv_path, group_pairs=5):
    """
    Одноколонковий CSV:
      перший рядок — заголовок (може бути 'text' із BOM),
      далі слова послідовно: 1,3,5,7,9 — ліворуч (ES), 2,4,6,8,10 — праворуч (UA).
    Групуємо по 5 пар (10 рядків).
    """
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()

    # прибираємо BOM, якщо є
    if raw.startswith("\ufeff"):
        raw = raw.lstrip("\ufeff")

    # розбиваємо на рядки, обрізаємо пробіли/коми й порожні
    lines = [ln.strip().strip(",") for ln in raw.splitlines()]
    lines = [ln for ln in lines if ln != ""]
    if not lines:
        raise ValueError("CSV порожній або не читається.")

    # прибираємо заголовок, якщо схожий
    first = lines[0].lower()
    if first in {"text", "word", "словo", "слово"} or "text" in first:
        lines = lines[1:]

    words = lines

    # має бути парна кількість
    if len(words) % 2 != 0:
        words = words[:-1]  # відкидаємо «хвіст», щоб не падати

    groups = []
    chunk_size = group_pairs * 2  # 10 рядків = 5 пар
    for start in range(0, len(words), chunk_size):
        chunk = words[start:start + chunk_size]
        if len(chunk) < chunk_size:
            break
        left = chunk[0::2]   # непарні за змістом
        right = chunk[1::2]  # парні за змістом
        groups.append((left, right))

    if not groups:
        raise ValueError("Не знайшов жодної повної групи по 5 парах у CSV (перевір порядок і кількість рядків).")

    return groups

def rgba_image_to_clip(img_rgba, position, duration):
    """PIL RGBA -> (RGB clip + альфа-маска), щоб уникнути помилок 3 vs 4 канали."""
    arr = np.array(img_rgba)
    if arr.ndim == 3 and arr.shape[2] == 4:
        rgb = arr[:, :, :3]
        alpha = (arr[:, :, 3].astype(float) / 255.0)
        base = ImageClip(rgb).set_position(position).set_duration(duration)
        mask = ImageClip(alpha, ismask=True).set_position(position).set_duration(duration)
        return base.set_mask(mask)
    else:
        return ImageClip(arr).set_position(position).set_duration(duration)

def make_column_image_fixed_rows(lines, n_rows, font_size, width_px, color_hex, line_spacing=1.25, pad_y=10):
    """Малює рівно n_rows рядків (порожні теж), щоб вирівнювання рядок-до-рядка було ідеальним."""
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, font_size)

    ascent, descent = font.getmetrics()
    line_h = ascent + descent
    row_h = int(line_h * line_spacing)
    total_h = n_rows * row_h + 2 * pad_y

    img = Image.new("RGBA", (width_px, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    r, g, b = hex_to_rgb(color_hex)
    fill = (r, g, b, 255)

    for i in range(n_rows):
        txt = lines[i] if i < len(lines) else ""
        w_px, h_px = draw.textsize(txt, font=font)
        x = (width_px - w_px) // 2
        y = pad_y + i * row_h + (row_h - h_px) // 2
        draw.text((x, y), txt, font=font, fill=fill)

    return img

def ring_image(w, h, frac, center, radius, thickness, color_rgb, bg_opacity=0.3):
    img = Image.new("RGBA", (w, h), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    cx, cy = center
    bbox = [cx-radius, cy-radius, cx+radius, cy+radius]
    bg_a = int(255*bg_opacity)
    draw.ellipse(bbox, outline=(255,255,255,bg_a), width=thickness)
    if frac > 0:
        end = 360*frac
        r,g,b = color_rgb
        draw.arc(bbox, start=-90, end=-90+end, width=thickness, fill=(r,g,b,255))
    return img

def digit_image(w, h, text, center_y, font_size, color_hex):
    img = Image.new("RGBA", (w, h), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, font_size)
    r,g,b = hex_to_rgb(color_hex)
    fill = (r,g,b,255)
    w_px, h_px = draw.textsize(text, font=font)
    x = (w - w_px)//2
    y = int(center_y - h_px*0.5)
    draw.text((x,y), text, font=font, fill=fill)
    return img

def make_timer_clip(W,H,dur,center,radius,thickness,accent_rgb,bg_opacity, number_center_y, font_size, text_color):
    r0 = ring_image(W,H,0.0,center,radius,thickness,accent_rgb,bg_opacity)
    n0 = digit_image(W,H,f"{int(math.ceil(dur))}",number_center_y,font_size,text_color)

    def ring_frame(t):
        frac = min(1.0, max(0.0, t/dur))
        return np.array(ring_image(W,H,frac,center,radius,thickness,accent_rgb,bg_opacity))[:, :, :3]
    def ring_mask(t):
        frac = min(1.0, max(0.0, t/dur))
        return np.array(ring_image(W,H,frac,center,radius,thickness,accent_rgb,bg_opacity))[:, :, 3].astype(float)/255.0
    def num_frame(t):
        n = max(0, int(math.ceil(dur - t)))
        return np.array(digit_image(W,H,f"{n}",number_center_y,font_size,text_color))[:, :, :3]
    def num_mask(t):
        n = max(0, int(math.ceil(dur - t)))
        img = digit_image(W,H,f"{n}",number_center_y,font_size,text_color)
        return np.array(img)[:, :, 3].astype(float)/255.0

    ring_clip = ImageClip(np.array(r0)[:, :, :3]).set_make_frame(ring_frame).set_duration(dur)
    ring_clip = ring_clip.set_mask(
        ImageClip((np.array(r0)[:, :, 3].astype(float)/255.0), ismask=True).set_make_frame(ring_mask).set_duration(dur)
    )

    num_clip = ImageClip(np.array(n0)[:, :, :3]).set_make_frame(num_frame).set_duration(dur)
    num_clip = num_clip.set_mask(
        ImageClip((np.array(n0)[:, :, 3].astype(float)/255.0), ismask=True).set_make_frame(num_mask).set_duration(dur)
    )

    return ring_clip, num_clip

def tts_espeak_uk(text, wav_path, rate=150):
    subprocess.run(["espeak-ng", "-v", "uk", "-s", str(rate), "-w", wav_path, text], check=True)


# -------------------- ОСНОВНА --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="inputs/seq1.csv")   # один стовпець: непарні=ES, парні=UA
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--out", default="outputs/lesson1.mp4")
    ap.add_argument("--stage_seconds", type=float, default=5.0,
                    help="кожен крок відкриття перекладу (сек.)")
    ap.add_argument("--last_hold_seconds", type=float, default=1.5,
                    help="показати ВСІ 10 слів після 5-ї пари (сек.)")
    ap.add_argument("--intro_delay", type=float, default=1.0,
                    help="затримка на старті (тільки фон+аватар), сек")
    args = ap.parse_args()

    C = load_config(args.config)
    W,H = C["width"], C["height"]
    FPS = C["fps"]

    print(">>> MODE: odd->left (ES), even->right (UA), groups of 5 pairs; 1s intro; 1.5s hold after 5th pair.")

    # фон
    bg = ColorClip(size=(W,H), color=hex_to_rgb(C["bg_color"])).set_duration(0.1)

    # аватар
    avatar_img = Image.open(C["avatar"]).convert("RGBA")
    aw = int(W*C["avatar_scale"])
    ah = int(avatar_img.height * (aw/avatar_img.width))
    avatar_img = avatar_img.resize((aw,ah), resample=Image.LANCZOS)
    avatar_pos = ("center", int(H*C["avatar_y"] - ah/2))

    # геометрія колонок
    cols_top_y = int(H*C["cols_top_y"])
    col_w = int(W*C["col_width"])
    margin = int(W*C["col_margin"])
    left_x = margin
    right_x = W - margin - col_w

    font_size = int(C.get("font_size",56))
    text_color = C.get("text_color","#ffffff")

    # таймер
    accent_rgb = hex_to_rgb(C["accent_color"])
    timer_center = (W//2, int(H*C["timer_top_y"]))
    timer_radius = C["timer_radius"]
    timer_thickness = C["timer_thickness"]
    timer_bg_opacity = C["timer_bg_opacity"]
    number_center_y = int(H*C["timer_top_y"] - C["timer_radius"]*0.15)
    num_font_size = C["timer_font_size"]

    # дані
    groups = read_sequence_to_groups(args.csv, group_pairs=5)
    print(f">>> Loaded groups: {len(groups)}")

    stage = args.stage_seconds
    hold_dur = max(0.001, args.last_hold_seconds)

    clips = []
    audio_segments = []
    t_cursor = 0.0  # відео-таймлайн
    t_audio  = 0.0  # аудіо-таймлайн

    # --- 1-секундна заставка перед першою п’ятіркою (тільки фон + аватар) ---
    intro = max(0.0, args.intro_delay)
    if intro > 0:
        avatar_intro = rgba_image_to_clip(avatar_img, avatar_pos, intro)
        comp_intro = CompositeVideoClip([bg.set_duration(intro), avatar_intro]).set_duration(intro)
        clips.append(comp_intro)
        t_cursor += intro
        t_audio  += intro

    tmpdir = tempfile.mkdtemp(prefix="tts_batch_")
    group_idx = 0

    for (left_words, right_words) in groups:
        current_group = group_idx
        group_idx += 1

        n = len(left_words)  # 5

        # Лівий блок (непарні) на всю групу
        left_block_img = make_column_image_fixed_rows(left_words, n, font_size, col_w, text_color)

        # КРОК 0: показати ліві 5; справа порожньо; таймер stage
        right_empty_img = make_column_image_fixed_rows([], n, font_size, col_w, text_color)

        left_block_clip0  = rgba_image_to_clip(left_block_img,  (left_x, cols_top_y), stage)
        right_block_clip0 = rgba_image_to_clip(right_empty_img, (right_x, cols_top_y), stage)
        avatar_clip0      = rgba_image_to_clip(avatar_img,      avatar_pos,          stage)
        ring0, num0       = make_timer_clip(
            W,H,stage, timer_center, timer_radius, timer_thickness,
            accent_rgb, timer_bg_opacity, number_center_y, num_font_size, text_color
        )

        comp0 = CompositeVideoClip([
            bg.set_duration(stage),
            avatar_clip0, left_block_clip0, right_block_clip0,
            ring0, num0
        ]).set_duration(stage)
        clips.append(comp0)

        t_cursor += stage
        t_audio  += stage  # перший TTS піде зі кроку 1

        # КРОКИ 1..5: додаємо переклади справа (і вони ЗАЛИШАЮТЬСЯ)
        for i in range(n):
            is_last = (i == n - 1)
            # 1..4 — повні 5 с із таймером; 5-е — миттєво (0.001 с), без таймера
            dur_video = stage if not is_last else 0.001

            accumulated = right_words[:i+1]
            right_block_img = make_column_image_fixed_rows(accumulated, n, font_size, col_w, text_color)
            right_block_clip = rgba_image_to_clip(right_block_img, (right_x, cols_top_y), dur_video)
            left_block_clip  = rgba_image_to_clip(left_block_img,  (left_x, cols_top_y), dur_video)
            avatar_clipi     = rgba_image_to_clip(avatar_img,      avatar_pos,          dur_video)

            layers = [bg.set_duration(dur_video), avatar_clipi, left_block_clip, right_block_clip]
            if not is_last:
                ringi, numi = make_timer_clip(
                    W, H, dur_video, timer_center, timer_radius, timer_thickness,
                    accent_rgb, timer_bg_opacity, number_center_y, num_font_size, text_color
                )
                layers += [ringi, numi]

            compi = CompositeVideoClip(layers).set_duration(dur_video)
            clips.append(compi)

            # Озвучка перекладу (праве слово) — унікальне ім'я, щоб не збивалося між групами
            wav_path = os.path.join(tmpdir, f"uk_g{current_group}_i{i}.wav")
            try:
                tts_espeak_uk(right_words[i], wav_path, rate=150)
                seg = AudioFileClip(wav_path).set_start(t_audio)
                audio_segments.append(seg)
            except Exception:
                pass

            # час
            t_cursor += dur_video
            t_audio  += stage  # аудіо завжди рівними кроками 5с

        # ПІСЛЯ 5-Ї ПАРИ: показати ВСІ 10 слів ще hold_dur (без таймера), потім одразу нова п’ятірка
        right_full_img  = make_column_image_fixed_rows(right_words, n, font_size, col_w, text_color)
        right_full_clip = rgba_image_to_clip(right_full_img, (right_x, cols_top_y), hold_dur)
        left_full_clip  = rgba_image_to_clip(left_block_img, (left_x, cols_top_y), hold_dur)
        avatar_hold     = rgba_image_to_clip(avatar_img,     avatar_pos,           hold_dur)

        comp_hold = CompositeVideoClip(
            [bg.set_duration(hold_dur), avatar_hold, left_full_clip, right_full_clip]
        ).set_duration(hold_dur)
        clips.append(comp_hold)

        t_cursor += hold_dur
        t_audio  += hold_dur  # 5-те слово може ще звучати в межах цієї «утримуючої» паузи

    # Фінал: відео + звук
    video = concatenate_videoclips(clips, method="compose")
    if audio_segments:
        final_audio = CompositeAudioClip(audio_segments).set_duration(video.duration)
        video = video.set_audio(final_audio)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    video.write_videofile(args.out, fps=FPS, codec="libx264", audio_codec="aac", preset="medium", threads=4)


if __name__ == "__main__":
    print(">>> MODE ACTIVE: odd->left, even->right, 1s intro, 5s stages, 1.5s hold of all 10, unique TTS per step.")
    main()
