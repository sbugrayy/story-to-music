"""
Story-to-Music Inference Engine.

İki fine-tuned mT5-small modeli yükler ve yönetir:
- Analyzer: Türkçe metin → {emotion, energy, bpm, key, instruments, vocal_style, suno_style_prompt}
- Lyricist: metadata → [Verse]/[Chorus]/[Bridge]/[Outro] etiketli Türkçe şarkı sözü

Modeller singleton; tek seferde yüklenir, sonraki çağrılar cache'den kullanır.
CPU/GPU otomatik tespit; CPU'da Lyricist ~30-60s, Analyzer ~2-5s.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

log = logging.getLogger(__name__)

# ── Path Resolution ────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_model_path(env_var: str, candidates: list[str]) -> Path:
    """Env var önce, yoksa fallback path'leri sırayla dene."""
    if (env_val := os.getenv(env_var)):
        p = Path(env_val)
        if p.exists():
            return p
    for c in candidates:
        p = _PROJECT_ROOT / c
        if p.exists():
            return p
    raise FileNotFoundError(
        f"{env_var} model bulunamadı. Denenmiş: {candidates}. "
        f"Env var {env_var} ile path verebilirsin."
    )


ANALYZER_PATH = _resolve_model_path(
    "ANALYZER_PATH",
    ["model/story-to-music-analyzer", "data/story-to-music-analyzer"],
)
LYRICIST_PATH = _resolve_model_path(
    "LYRICIST_PATH",
    ["model/story-to-music-lyricist", "data/story-to-music-lyricist"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f"Inference cihazı: {DEVICE}")


# ── Veri Tipleri ──────────────────────────────────────────────────────────────

@dataclass
class MusicMetadata:
    """Analyzer çıktısı — Suno/Udio için müzik parametreleri."""
    emotion: str = ""
    energy: int = 5
    bpm: int = 90
    key: str = "A minor"
    instruments: list[str] = field(default_factory=list)
    vocal_style: str = ""
    suno_style_prompt: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_lyricist_prompt(self) -> str:
        """Lyricist modeline verilecek prompt formatı."""
        instr_str = ", ".join(self.instruments) if self.instruments else ""
        return (
            f"Türkçe şarkı sözü yaz: "
            f"duygu={self.emotion}, "
            f"enerji={self.energy}, "
            f"tempo={self.bpm} BPM, "
            f"ton={self.key}, "
            f"enstrümanlar={instr_str}, "
            f"vokal={self.vocal_style}"
        )


# ── Model Wrapper ─────────────────────────────────────────────────────────────

class _ModelHolder:
    """Lazy-loaded singleton wrapper."""

    def __init__(self, path: Path, label: str):
        self.path = path
        self.label = label
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        log.info(f"[{self.label}] yükleniyor: {self.path}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.path)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.path).to(DEVICE)
        self._model.eval()
        # Generation sırasında gradient_checkpointing kapalı olmalı
        try:
            self._model.gradient_checkpointing_disable()
        except Exception:
            pass
        log.info(f"[{self.label}] hazır ({self._model.num_parameters():,} param)")

    @property
    def tokenizer(self):
        self._ensure_loaded()
        return self._tokenizer

    @property
    def model(self):
        self._ensure_loaded()
        return self._model


_analyzer = _ModelHolder(ANALYZER_PATH, "Analyzer")
_lyricist = _ModelHolder(LYRICIST_PATH, "Lyricist")


# ── Analyzer ──────────────────────────────────────────────────────────────────

_ANALYZER_PROMPT_PREFIX = "Şu metni analiz et ve müzik promptu üret: "


def analyze(text: str) -> MusicMetadata:
    """
    Türkçe metin → MusicMetadata.

    Modelin JSON çıktısı parse edilir; alan eksikse default değer kullanılır.
    """
    prompt = _ANALYZER_PROMPT_PREFIX + text.strip()
    tok = _analyzer.tokenizer
    mdl = _analyzer.model

    inputs = tok(prompt, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            max_new_tokens=192,
            num_beams=4,
            early_stopping=True,
        )
    decoded = tok.decode(out[0], skip_special_tokens=True).strip()

    return _parse_metadata(decoded)


def _parse_metadata(raw: str) -> MusicMetadata:
    """Analyzer çıktısını MusicMetadata'ya çevir; bozuk JSON'ı toleranslı parse et."""
    # Doğrudan JSON dene
    try:
        d = json.loads(raw)
        return _from_dict(d)
    except json.JSONDecodeError:
        pass

    # Süslü parantezler içinden tek satır JSON çıkar
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            return _from_dict(d)
        except json.JSONDecodeError:
            pass

    log.warning(f"Analyzer JSON parse başarısız: {raw[:200]!r}")
    return MusicMetadata()


def _from_dict(d: dict) -> MusicMetadata:
    instruments = d.get("instruments", [])
    if isinstance(instruments, str):
        instruments = [i.strip() for i in instruments.split(",") if i.strip()]
    return MusicMetadata(
        emotion=str(d.get("emotion", "")),
        energy=int(d.get("energy", 5)) if str(d.get("energy", "5")).strip() else 5,
        bpm=int(d.get("bpm", 90)) if str(d.get("bpm", "90")).strip() else 90,
        key=str(d.get("key", "A minor")),
        instruments=list(instruments) if isinstance(instruments, list) else [],
        vocal_style=str(d.get("vocal_style", "")),
        suno_style_prompt=str(d.get("suno_style_prompt", "")),
    )


# ── Lyricist ──────────────────────────────────────────────────────────────────

# Sampling parametreleri — beam search mode collapse'e sebep oluyor, sampling kullan
_LYRICIST_GEN_KWARGS = dict(
    max_new_tokens=350,
    do_sample=True,
    num_beams=1,
    temperature=0.9,
    top_p=0.9,
    top_k=50,
    no_repeat_ngram_size=3,
    repetition_penalty=1.3,
)


def generate_lyrics(meta: MusicMetadata, seed: Optional[int] = None) -> str:
    """
    Metadata → yapılandırılmış Türkçe şarkı sözü.

    Sampling kullanılır; her çağrı farklı çıktı verir. Deterministic istersen
    seed=42 geç.
    """
    if seed is not None:
        torch.manual_seed(seed)

    prompt = meta.to_lyricist_prompt()
    tok = _lyricist.tokenizer
    mdl = _lyricist.model

    inputs = tok(prompt, return_tensors="pt", max_length=64, truncation=True).to(DEVICE)
    with torch.no_grad():
        out = mdl.generate(**inputs, **_LYRICIST_GEN_KWARGS)
    decoded = tok.decode(out[0], skip_special_tokens=True).strip()

    return _format_lyrics(decoded)


def _format_lyrics(raw: str) -> str:
    """
    Lyricist tek satırda üretiyor — okunabilir hale getir:
    1) Bozuk etiketleri normalize et ("Verses 2" → "Verse 2")
    2) Yapısal etiketler kendi satırında dursun
    3) Bölüm içinde, büyük harfle başlayan yeni cümleleri ayrı satıra al
       (mT5'in token-by-token üretiminde newline'lar kaybolmuş oluyor)
    """
    # 1) Bozuk etiket normalizasyonu — model "Versed", "Versen", "Verses" gibi sapmalar üretiyor
    #    "Verse" prefiksi + herhangi bir sonek → "Verse N"
    raw = re.sub(
        r"\[Vers\w*\s*(\d*)\s*\]",
        lambda m: f"[Verse{(' ' + m.group(1)) if m.group(1).strip() else ''}]",
        raw, flags=re.IGNORECASE,
    )
    raw = re.sub(r"\[Pre[\s-]?Chor\w*\]", "[Pre-Chorus]", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\[Chor\w*\]", "[Chorus]", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\[Brid\w*\]", "[Bridge]", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\[Outr\w*\]", "[Outro]", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\[Intr\w*\]", "[Intro]", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\[Hook\w*\]", "[Hook]", raw, flags=re.IGNORECASE)

    # 2) Yapısal etiketler kendi paragrafında
    formatted = re.sub(
        r"\s*(\[(?:Verse(?:\s*\d+)?|Chorus|Bridge|Outro|Pre-Chorus|Hook|Intro)\])\s*",
        r"\n\n\1\n", raw,
    )
    # Birden fazla boşluğu tek boşluğa indir
    formatted = re.sub(r"[ \t]+", " ", formatted)

    # 3) Bölüm içinde satır kırılımı:
    #    " Kelime" pattern'i (boşluk + büyük harf) genelde yeni dize başlangıcıdır
    #    Etiketleri koru, sadece bölüm içeriklerinde uygula
    lines = formatted.split("\n")
    out_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            out_lines.append("")
            continue
        # Etiket ise olduğu gibi
        if re.match(r"^\[", line):
            out_lines.append(line)
            continue
        # İçerikte boşluk + büyük harf öncesi kırılım (Türkçe karakterler dahil)
        # Ancak çok kısa parça kalmasın diye min 3 kelime önde olsun
        broken = re.sub(
            r"(?<=[a-zçğıöşü?!,.])  ?([A-ZÇĞİÖŞÜ][a-zçğıöşü])",
            r"\n\1",
            line,
        )
        # Tek satıra eğer ki uzunsa, kırılan parçaları tek tek ekle
        for sub in broken.split("\n"):
            sub = sub.strip()
            if sub:
                out_lines.append(sub)

    # Fazla boş satırları azalt (3+ → 2)
    result = "\n".join(out_lines)
    result = re.sub(r"\n{3,}", "\n\n", result).strip()
    return result


# ── Title Suggestion ──────────────────────────────────────────────────────────

_TITLE_STOPWORDS = {
    # Yapısal etiket kelimeleri (sızdığında başlığa girmesin)
    "verse", "verses", "versed", "versen", "chorus", "bridge", "outro",
    "intro", "hook", "pre",
    # Yaygın Türkçe stop-words
    "için", "olan", "olarak", "gibi", "ama", "fakat", "veya", "ile",
    "kadar", "daha", "çok", "şey", "ben", "sen", "bir", "biz", "her",
    "bana", "sana", "bizim", "sizin", "bunu", "şunu",
}


def suggest_titles(meta: MusicMetadata, lyrics: str) -> list[str]:
    """
    Basit heuristic title üretimi. Lyrics'ten anahtar kelimeler + duygu kombinasyonu.

    Eğitilmiş model yok, deterministic kural-tabanlı.
    """
    # Yapısal etiketleri tamamen kaldır (içerikten çek)
    lyrics_clean = re.sub(r"\[[^\]]*\]", " ", lyrics)
    words = re.findall(r"[A-Za-zçğıöşüÇĞİÖŞÜ]+", lyrics_clean)
    # 4+ harfli, stopword olmayan kelimeler, ilk geçen sıra
    interesting = []
    seen = set()
    for w in words:
        wl = w.lower()
        if len(wl) >= 4 and wl not in seen and wl not in _TITLE_STOPWORDS:
            seen.add(wl)
            interesting.append(w)
        if len(interesting) >= 6:
            break

    emotion_titles = {
        "hüzün": ["Sessiz Geceler", "Yarım Kalan"],
        "öfke": ["Kırgın", "Ses Verme"],
        "aşk": ["Seninle", "Bir Bakış"],
        "özlem": ["Uzaktan", "Hatıra"],
        "neşe": ["Bugün", "Birlikte"],
        "isyan": ["Boyun Eğmem", "Karşı Dur"],
        "korku": ["Karanlık", "Gölge"],
        "umut": ["Yarın", "Yeniden"],
    }
    base = emotion_titles.get(meta.emotion.lower(), ["Yansıma", "Bir An"])
    candidates = list(base)
    if interesting:
        candidates.append(" ".join(interesting[:2]).title())
    return candidates[:3]
