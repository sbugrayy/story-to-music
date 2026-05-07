"""
Şarkı sözü temizleme pipeline'ı.
data/raw/*.jsonl → data/cleaned/*.jsonl

Adımlar:
  1. Genius başlık satırını kaldır  ("["X" için şarkı sözleri]")
  2. Yapısal etiketleri kaldır      ([Verse], [Nakarat], [Bölüm 1] vb.)
  3. Metadata artıklarını kaldır    (You might also like, Embed vb.)
  4. Tekrar eden satırları normalize et
  5. Dil tespiti                    (%70+ Türkçe zorunlu)
  6. Uzunluk filtresi               (50–600 kelime)
  7. Güvenlik taraması              (nefret söylemi, açık zararlı içerik)
"""

import re
import sys
import json
import logging
import argparse
from pathlib import Path
from collections import Counter

from langdetect import detect_langs, DetectorFactory, LangDetectException

# Windows konsol encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Langdetect sonuçlarını tutarlı yap
DetectorFactory.seed = 42

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
))
logging.basicConfig(level=logging.INFO, handlers=[_handler])
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
RAW_DIR     = Path(__file__).parent.parent / "data" / "raw"
CLEANED_DIR = Path(__file__).parent.parent / "data" / "cleaned"

MIN_WORDS = 50
MAX_WORDS = 600
MIN_TR_PROB = 0.70   # Türkçe olasılık eşiği

# Genius başlık satırı: ["Şarkı Adı" için şarkı sözleri]
GENIUS_HEADER = re.compile(
    r'^\[.*?için\s+şarkı\s+sözleri\]',
    re.IGNORECASE | re.MULTILINE,
)

# Yapısal etiketler — Türkçe ve İngilizce
SECTION_TAG = re.compile(
    r'^\s*\['
    r'(?:'
    # Türkçe etiketler
    r'[Bb]ölüm|[Nn]akarat|[Kk]öprü|[Gg]iriş|[Çç]ıkış'
    r'|[Ee]nstrümantal|[Ss]ol(?:o)?|[Kk]onuşma|[Tt]ekrar'
    r'|[Ss]on|[Bb]aşlangıç|[Aa]ra|[Rr]efren'
    # İngilizce etiketler
    r'|[Vv]erse|[Cc]horus|[Bb]ridge|[Oo]utro|[Ii]ntro'
    r'|[Pp]re[\s\-][Cc]horus|[Pp]ost[\s\-][Cc]horus'
    r'|[Hh]ook|[Rr]efrain|[Ii]nterlude|[Ss]olo|[Ss]poken'
    r'|[Oo]utro|[Bb]reak|[Cc]oda|[Ee]nd'
    r')'
    r'[\s\d\:\-–]*'
    r'\]\s*$',
    re.MULTILINE,
)

# Genius metadata artıkları
METADATA_LINES = re.compile(
    r'^(?:You might also like|Embed|See\s|Contributors|Translations'
    r'|\d+\s*Embed|\d+\s*Contributors)\s*$',
    re.IGNORECASE | re.MULTILINE,
)

# Güvenlik: eğitim verisinden çıkarılacak içerik kalıpları
# NOT: Küfür tek başına engel değil; yaratıcı bağlamdaki kaba dil normaldir.
# Sadece doğrudan şiddet çağrısı, istismar veya nefret söylemi engellenir.
HARD_BLOCK_PATTERNS = [
    # Belirli gruplara yönelik doğrudan şiddet çağrısı
    re.compile(
        r'\b(?:öldür(?:ün|elim)|katlet(?:in|elim)|imha\s+et(?:in|elim))\s+'
        r'(?:kürt|ermeni|rum|yahudi|müslüman|hristiyan|alevi|sünni)\b',
        re.IGNORECASE,
    ),
    # Çocuk istismarı
    re.compile(
        r'\b(?:çocuk|küçük)\s+(?:istismar|taciz|tecavüz)\b',
        re.IGNORECASE,
    ),
    # Terör örgütü yüceltmesi (spesifik örgüt adıyla eylem övgüsü)
    re.compile(
        r'\b(?:pkk|deaş|işid|isis)\s+(?:yaşa(?:sın)?|kazan(?:dı|acak)|zafer)\b',
        re.IGNORECASE,
    ),
]

# ---------------------------------------------------------------------------
# Temizleme fonksiyonları
# ---------------------------------------------------------------------------

def remove_genius_header(text: str) -> str:
    """İlk satırdaki Genius başlık bilgisini kaldır."""
    lines = text.splitlines()
    if lines and GENIUS_HEADER.match(lines[0].strip()):
        lines = lines[1:]
    return "\n".join(lines)


def remove_section_tags(text: str) -> str:
    """[Verse 1], [Nakarat], [Bölüm 2] gibi yapısal etiketleri kaldır."""
    return SECTION_TAG.sub("", text)


def remove_metadata(text: str) -> str:
    """Genius'un eklediği metadata satırlarını kaldır."""
    return METADATA_LINES.sub("", text)


def normalize_whitespace(text: str) -> str:
    """Üçten fazla ardışık boş satırı ikiye indir, baş/son boşlukları kaldır."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def normalize_repeated_lines(text: str) -> str:
    """
    Aynı satırın 3+ kez arka arkaya tekrarını 2'ye indir.
    Nakarat blokları normalde 2 kez geçer, 3+ kez geçmesi veri gürültüsüdür.
    """
    lines = text.splitlines()
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Kaç kez tekrar ediyor?
        j = i
        while j < len(lines) and lines[j] == line:
            j += 1
        repeat_count = j - i
        # Boş satırlar için max 2, dolu satırlar için max 2
        keep = min(repeat_count, 2)
        result.extend([line] * keep)
        i = j
    return "\n".join(result)


def clean_text(raw: str) -> str:
    """Tüm temizleme adımlarını sırayla uygula."""
    text = remove_genius_header(raw)
    text = remove_section_tags(text)
    text = remove_metadata(text)
    text = normalize_repeated_lines(text)
    text = normalize_whitespace(text)
    return text


# ---------------------------------------------------------------------------
# Filtreleme fonksiyonları
# ---------------------------------------------------------------------------

def word_count(text: str) -> int:
    return len(text.split())


def turkish_probability(text: str) -> float:
    """langdetect ile Türkçe olasılığını döndür. Hata durumunda 0.0."""
    try:
        langs = detect_langs(text)
        for lang in langs:
            if lang.lang == "tr":
                return lang.prob
        return 0.0
    except LangDetectException:
        return 0.0


def passes_security(text: str) -> tuple[bool, str]:
    """
    Güvenlik taramasından geçirse True döner.
    Geçemezse (False, sebep) döner.
    """
    for pattern in HARD_BLOCK_PATTERNS:
        if pattern.search(text):
            return False, f"hard_block: {pattern.pattern[:60]}"
    return True, ""


# ---------------------------------------------------------------------------
# Ana pipeline
# ---------------------------------------------------------------------------

def process_file(jsonl_path: Path) -> tuple[list[dict], dict]:
    """
    Tek bir raw JSONL dosyasını işle.
    Temiz kayıtları ve istatistikleri döndür.
    """
    stats: dict[str, int] = Counter()
    clean_records: list[dict] = []

    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    stats["total"] = len(lines)

    for line in lines:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            stats["json_error"] += 1
            continue

        raw_lyrics = rec.get("lyrics", "")
        if not raw_lyrics:
            stats["empty"] += 1
            continue

        # 1-4: Temizleme
        cleaned = clean_text(raw_lyrics)

        # 5: Uzunluk filtresi
        wc = word_count(cleaned)
        if wc < MIN_WORDS:
            stats["too_short"] += 1
            continue
        if wc > MAX_WORDS:
            stats["too_long"] += 1
            continue

        # 6: Dil tespiti
        tr_prob = turkish_probability(cleaned)
        if tr_prob < MIN_TR_PROB:
            stats["not_turkish"] += 1
            log.debug("  Türkçe değil (%.2f): %s — %s",
                      tr_prob, rec.get("artist"), rec.get("title"))
            continue

        # 7: Güvenlik taraması
        ok, reason = passes_security(cleaned)
        if not ok:
            stats["blocked"] += 1
            log.warning("  Engellendi [%s]: %s — %s", reason,
                        rec.get("artist"), rec.get("title"))
            continue

        stats["passed"] += 1
        clean_records.append({
            "artist":   rec["artist"],
            "title":    rec["title"],
            "url":      rec.get("url", ""),
            "category": rec.get("category", ""),
            "lyrics":   cleaned,
            "word_count": wc,
            "collected_at": rec.get("collected_at", ""),
        })

    return clean_records, stats


def run(categories: list[str] | None = None) -> None:
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(RAW_DIR.glob("*.jsonl"))
    if categories:
        jsonl_files = [f for f in jsonl_files if f.stem in categories]

    if not jsonl_files:
        log.error("data/raw/ altında işlenecek dosya bulunamadı.")
        return

    grand_total = grand_passed = 0

    for raw_file in jsonl_files:
        log.info("=" * 60)
        log.info("İşleniyor: %s", raw_file.name)
        log.info("=" * 60)

        records, stats = process_file(raw_file)

        # Kategori bazlı dosyaya kaydet
        out_path = CLEANED_DIR / raw_file.name
        with out_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        total   = stats["total"]
        passed  = stats["passed"]
        dropped = total - passed

        log.info("  Toplam : %d", total)
        log.info("  Geçti  : %d (%.1f%%)", passed, 100 * passed / total if total else 0)
        log.info("  Düştü  : %d", dropped)
        if stats["too_short"]:   log.info("    Çok kısa      : %d", stats["too_short"])
        if stats["too_long"]:    log.info("    Çok uzun       : %d", stats["too_long"])
        if stats["not_turkish"]: log.info("    Türkçe değil   : %d", stats["not_turkish"])
        if stats["blocked"]:     log.info("    Güvenlik engeli: %d", stats["blocked"])
        if stats["empty"]:       log.info("    Boş içerik     : %d", stats["empty"])
        log.info("  Kaydedildi: %s", out_path)

        grand_total  += total
        grand_passed += passed

    log.info("=" * 60)
    log.info(
        "Tamamlandı. Genel: %d / %d şarkı geçti (%.1f%%).",
        grand_passed, grand_total,
        100 * grand_passed / grand_total if grand_total else 0,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ham şarkı sözlerini temizle ve filtrele."
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        metavar="KATEGORİ",
        help="Sadece belirtilen kategorileri işle (ör: arabesk rock).",
    )
    args = parser.parse_args()
    run(categories=args.categories)


if __name__ == "__main__":
    main()
