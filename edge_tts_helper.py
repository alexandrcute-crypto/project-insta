# edge_tts_helper.py
import asyncio
import edge_tts

# Базова мапа голосів (можеш міняти пізніше)
VOICE_MAP = {
    "uk": "uk-UA-PolinaNeural",   # або uk-UA-OstapNeural
    "es": "es-ES-AlvaroNeural",   # або es-ES-ElviraNeural
    "ru": "ru-RU-SvetlanaNeural", # або ru-RU-DmitryNeural
    "de": "de-DE-KillianNeural",  # або de-DE-KatjaNeural
    "fr": "fr-FR-DeniseNeural",   # або fr-FR-HenriNeural
    "pt": "pt-PT-DuarteNeural",   # або pt-BR-AntonioNeural
    "tr": "tr-TR-AhmetNeural",    # або tr-TR-EmelNeural
    "en": "en-US-GuyNeural"
}

def tts_edge(text: str, lang_code: str, out_path: str, voice_override: str = ""):
    """
    Генерує MP3 з тексту.
    text          — текст для озвучки
    lang_code     — короткий код мови: 'uk' | 'es' | 'en'
    out_path      — шлях до вихідного MP3 (наприклад 'outputs\\word_01.mp3')
    voice_override— (необов’язково) повна назва голосу Edge TTS
    """
    voice = voice_override or VOICE_MAP.get(lang_code.lower(), VOICE_MAP["uk"])

    async def _run():
        tts = edge_tts.Communicate(text, voice)
        await tts.save(out_path)

    asyncio.run(_run())
