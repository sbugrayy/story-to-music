"""
Telif hakkı koruması — üretilen şarkı sözünün eğitim setindekilerle benzerliğini kontrol et.

Eşik (CLAUDE.md): Jaccard > 0.35 → çok benzer, ret.

Eğitim setindeki şarkı sözleri startup'ta yüklenir; her inference'da yeni çıktı bunlarla
karşılaştırılır. N-gram bazlı (kelime), kelime kümesi → Jaccard.

Performans: 854 ham şarkı + 3720 augmented = ~4500 referans. Her biri ~200 kelime →
hızlı set comparison. CPU'da <100ms.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Referans dataset — augment + cleaned hepsi
_DATASET_CANDIDATES = [
    "data/training/dataset_lyrics_v2.jsonl",
    "data/training/dataset_lyrics.jsonl",
    "data/training/dataset.jsonl",
]

DEFAULT_THRESHOLD = 0.35


# ── Tokenization ──────────────────────────────────────────────────────────────

def _tokenize(text: str) -> set[str]:
    """Şarkı sözünü kelime kümesine çevir; etiketleri (Verse, Chorus) at."""
    # Yapısal etiketleri kaldır
    cleaned = re.sub(r"\[(?:Verse[^]]*|Chorus|Bridge|Outro|Pre-Chorus|Hook|Intro)\]",
                     " ", text)
    # Türkçe + Latin alfabesi
    words = re.findall(r"[a-zçğıöşü]{3,}", cleaned.lower())
    return set(words)


# ── Referans Veri ─────────────────────────────────────────────────────────────

class _ReferenceCorpus:
    """Lazy-loaded singleton."""

    def __init__(self):
        self._refs: list[set[str]] | None = None

    def _load(self):
        if self._refs is not None:
            return
        refs: list[set[str]] = []
        for candidate in _DATASET_CANDIDATES:
            path = _PROJECT_ROOT / candidate
            if path.exists():
                log.info(f"[copyright_guard] referans yükleniyor: {candidate}")
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        try:
                            d = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        # dataset_lyrics_v2 → target_text; dataset.jsonl → structured_lyrics
                        lyrics = d.get("target_text") or d.get("structured_lyrics") or ""
                        if lyrics:
                            tokens = _tokenize(lyrics)
                            if len(tokens) >= 10:  # çok kısa şarkıları atla
                                refs.append(tokens)
                break  # ilk bulduğu kaynağı kullan, dur
        self._refs = refs
        log.info(f"[copyright_guard] {len(refs)} referans yüklendi")

    @property
    def refs(self) -> list[set[str]]:
        self._load()
        return self._refs  # type: ignore


_corpus = _ReferenceCorpus()


# ── API ───────────────────────────────────────────────────────────────────────

def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def check_copyright(
    generated: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[bool, float, int]:
    """
    Üretilen şarkıyı tüm referanslarla karşılaştır, en yüksek Jaccard'ı bul.

    Returns:
        (is_safe, max_jaccard, ref_index)
        is_safe=True ise eşiğin altında, üretim güvenli.
    """
    gen_tokens = _tokenize(generated)
    if len(gen_tokens) < 5:
        return True, 0.0, -1

    max_score = 0.0
    max_idx = -1
    for i, ref in enumerate(_corpus.refs):
        score = jaccard(gen_tokens, ref)
        if score > max_score:
            max_score = score
            max_idx = i
            # Eşiği erken aşarsa kısa devre
            if max_score > threshold:
                return False, max_score, max_idx

    return True, max_score, max_idx
