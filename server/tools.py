"""
MCP tool tanımları.

Üç tool:
- generate_music_prompt: Tam pipeline (analyzer + lyricist)
- analyze_emotion: Sadece analyzer
- generate_lyrics_only: Sadece lyricist (kullanıcı metadata verir)
"""

from __future__ import annotations

import logging
import re
from typing import Any

from . import content_guard, copyright_guard, inference
from .inference import MusicMetadata, analyze, generate_lyrics, suggest_titles

log = logging.getLogger(__name__)

MAX_COPYRIGHT_RETRIES = 3


# ── generate_music_prompt ─────────────────────────────────────────────────────

def generate_music_prompt(text: str, mode: str = "custom", language: str = "tr") -> dict[str, Any]:
    """
    Türkçe metin → Suno/Udio için tam müzik promptu (stil + sözler + başlıklar).

    Args:
        text: Analiz edilecek Türkçe metin (hikaye/konsept/şarkı sözü, 10-800 kelime)
        mode: "custom" = tam çıktı, "prompt_only" = sadece tek satır style prompt
        language: Şu an sadece "tr"

    Returns:
        custom modda: {style_prompt, structured_lyrics, title_suggestions, parameters}
        prompt_only modda: {suno_prompt}
    """
    # 1) Input validation
    ok, err = _validate_input(text)
    if not ok:
        return {"error": err}

    # 2) Content moderation
    allowed, reason = content_guard.check_input(text)
    if not allowed:
        return {"error": reason}

    # 3) Analyzer: metin → metadata
    try:
        meta = analyze(text)
    except Exception as e:
        log.exception("Analyzer hatası")
        return {"error": f"Analiz başarısız: {e}"}

    # 4) prompt_only modda kısa devre
    if mode == "prompt_only":
        return {"suno_prompt": _pick_suno_prompt(meta)}

    # 5) Lyricist: metadata → şarkı sözü (telif kontrolüyle)
    lyrics, copyright_score = _generate_with_copyright_guard(meta)
    if lyrics is None:
        return {
            "error": "Üretilen şarkı sözleri eğitim setine çok benziyor, 3 deneme başarısız."
        }

    # 6) Output moderation
    out_ok, out_reason = content_guard.check_output(lyrics)
    if not out_ok:
        return {"error": f"Çıktı politika ihlali: {out_reason}"}

    # 7) Title suggestion
    titles = suggest_titles(meta, lyrics)

    return {
        "style_prompt": _pick_suno_prompt(meta),
        "structured_lyrics": lyrics,
        "title_suggestions": titles,
        "parameters": meta.to_dict(),
        "copyright_similarity": round(copyright_score, 3),
    }


# ── analyze_emotion ───────────────────────────────────────────────────────────

def analyze_emotion(text: str) -> dict[str, Any]:
    """
    Sadece duygu/stil analizi. Hafif kullanım.
    """
    ok, err = _validate_input(text)
    if not ok:
        return {"error": err}

    allowed, reason = content_guard.check_input(text)
    if not allowed:
        return {"error": reason}

    try:
        meta = analyze(text)
    except Exception as e:
        log.exception("Analyzer hatası")
        return {"error": f"Analiz başarısız: {e}"}

    return meta.to_dict()


# ── generate_lyrics_only ──────────────────────────────────────────────────────

def generate_lyrics_only(
    emotion: str,
    energy: int = 5,
    bpm: int = 90,
    key: str = "A minor",
    instruments: list[str] | None = None,
    vocal_style: str = "",
) -> dict[str, Any]:
    """
    Kullanıcı metadata verirse sadece şarkı sözü üret. Analyzer atlanır.
    """
    meta = MusicMetadata(
        emotion=emotion,
        energy=int(energy),
        bpm=int(bpm),
        key=key,
        instruments=list(instruments) if instruments else [],
        vocal_style=vocal_style,
    )

    lyrics, copyright_score = _generate_with_copyright_guard(meta)
    if lyrics is None:
        return {"error": "Üretilen şarkı sözleri eğitim setine çok benziyor."}

    out_ok, out_reason = content_guard.check_output(lyrics)
    if not out_ok:
        return {"error": f"Çıktı politika ihlali: {out_reason}"}

    return {
        "structured_lyrics": lyrics,
        "copyright_similarity": round(copyright_score, 3),
        "parameters": meta.to_dict(),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_input(text: str) -> tuple[bool, str]:
    if not text or not text.strip():
        return False, "Metin boş olamaz."
    words = text.split()
    if len(words) < 5:
        return False, "Metin çok kısa (min 5 kelime)."
    if len(words) > 800:
        return False, "Metin çok uzun (max 800 kelime)."
    return True, ""


def _generate_with_copyright_guard(meta: MusicMetadata) -> tuple[str | None, float]:
    """Max 3 deneme; eşik aşılmıyorsa lyrics döndür."""
    best_score = 1.0
    for attempt in range(MAX_COPYRIGHT_RETRIES):
        lyrics = generate_lyrics(meta)
        is_safe, score, _ = copyright_guard.check_copyright(lyrics)
        if is_safe:
            return lyrics, score
        if score < best_score:
            best_score = score
        log.warning(
            f"Copyright eşiği aşıldı (deneme {attempt+1}/{MAX_COPYRIGHT_RETRIES}, "
            f"jaccard={score:.3f})"
        )
    return None, best_score


_EMOTION_EN = {
    "hüzün": "melancholic", "öfke": "angry", "aşk": "romantic", "özlem": "yearning",
    "neşe": "joyful", "isyan": "rebellious", "korku": "fearful", "umut": "hopeful",
}

_INSTRUMENT_EN = {
    "ney": "ney flute", "keman": "violin", "ud": "oud", "saz": "saz",
    "bağlama": "baglama", "piyano": "piano", "gitar": "guitar",
    "elektro gitar": "electric guitar", "davul": "drums", "bas": "bass",
    "klarinet": "clarinet", "kanun": "qanun", "darbuka": "darbuka",
}


_TURKISH_LIKELY_WORDS = {
    # Türkçe karakter içermese de tipik Türkçe sızıntı kelimeleri
    "nostaljik", "romantik", "dramatik", "klasik", "modern", "akustik",
    "elektronik", "ritmik", "melodik", "lirik", "epik",
    "ve", "ile", "bir", "sarki", "soz",
    "ney", "saz", "ud", "baglama", "kemence", "darbuka", "kanun",
    "erkek", "kadin", "vokal",
}


def _looks_english(s: str) -> bool:
    """Yarı-Türkçe yarı-İngilizce çıktıları tespit et. Şüphe varsa False."""
    if not s or len(s) < 15:
        return False
    # Türkçe-spesifik karakter varsa muhtemelen Türkçe sızmış
    if any(c in s for c in "çğıöşüÇĞİÖŞÜ"):
        return False
    # Türkçe-likely kelimeler
    words = set(re.findall(r"[a-z]+", s.lower()))
    if words & _TURKISH_LIKELY_WORDS:
        return False
    # İngilizce stil kelimeleri içermeli (sanity check)
    en_style_words = {
        "song", "music", "ballad", "track", "tune",
        "vocal", "vocals", "voice", "instrument", "instruments",
        "guitar", "violin", "drums", "piano", "bass", "flute",
        "slow", "fast", "moderate", "upbeat", "tempo", "rhythm",
        "minor", "major", "key", "bpm",
        "melancholic", "joyful", "romantic", "energetic", "calm",
        "turkish", "arabesque", "folk", "pop", "rock",
    }
    return bool(words & en_style_words)


def _fallback_suno_prompt(meta: MusicMetadata) -> str:
    """Metadata'dan deterministik İngilizce Suno prompt."""
    emotion = _EMOTION_EN.get(meta.emotion.lower(), meta.emotion or "atmospheric")
    if meta.energy <= 3:
        energy_word = "slow, melancholic"
    elif meta.energy <= 5:
        energy_word = "moderate tempo"
    elif meta.energy <= 7:
        energy_word = "upbeat"
    else:
        energy_word = "energetic, intense"

    instr_en = [_INSTRUMENT_EN.get(i.lower(), i) for i in meta.instruments]
    instr_str = ", ".join(instr_en) if instr_en else "acoustic instruments"

    vocal = meta.vocal_style.replace("erkek", "male").replace("kadın", "female")
    vocal = vocal.replace("dramatik", "dramatic").replace("yumuşak", "soft")
    vocal = vocal.replace("kısık", "raspy").replace("ağır", "deep")
    vocal = vocal.replace("lirik", "lyrical").replace("isyankar", "rebellious")
    vocal = vocal.replace("sert", "harsh").replace("dokunaklı", "emotional")
    vocal_str = f"{vocal} vocal" if vocal else "vocal"

    return f"Turkish {emotion} song, {energy_word}, {instr_str}, {meta.bpm} BPM, {meta.key}, {vocal_str}"


def _pick_suno_prompt(meta: MusicMetadata) -> str:
    """Analyzer çıktısı İngilizce ise kullan, değilse fallback üret."""
    raw = meta.suno_style_prompt.strip()
    if raw and _looks_english(raw):
        return raw
    return _fallback_suno_prompt(meta)
