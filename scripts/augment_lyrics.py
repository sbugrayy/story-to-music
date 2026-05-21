"""
Sentetik Türkçe şarkı sözü üretimi (data augmentation) — Model 2 (Söz Yazarı) için.

dataset.jsonl içindeki her şarkının yalnızca METADATA'sını referans alarak Ollama
(qwen2.5:7b) üzerinden 5 ayrı 'lens' direktifiyle özgün varyantlar üretir.
Orijinal sözler prompt'a verilmez — telif riski yapısal olarak sıfır ve train/test
distribution birebir Model 2 inference görevine eşleşir (metadata → lyrics).

Çıktı: data/training/lyrics_augmented.jsonl

Kullanım:
    # Smoke test (10 varyant, ~1-2 dk)
    python scripts/augment_lyrics.py --limit 5 --variants 2

    # Full run (854 × 5 = 4270 varyant)
    python scripts/augment_lyrics.py
"""

import os
import re
import sys
import json
import time
import logging
import argparse
from pathlib import Path

import requests

# Windows konsol encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
TRAINING_DIR = Path(__file__).parent.parent / "data" / "training"
INPUT_PATH   = TRAINING_DIR / "dataset.jsonl"
OUTPUT_PATH  = TRAINING_DIR / "lyrics_augmented.jsonl"

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL     = "http://localhost:11434/api/tags"
MODEL       = "qwen2.5:7b"
NUM_CTX     = 2048          # few-shot örneklerini sığdırmak için orta seviye
NUM_PREDICT = 512           # çıktı tavanı (~180-220 kelime)
NUM_GPU     = 99            # tüm layer'ları GPU'ya zorla (qwen2.5:7b ~28 layer)
TEMPERATURE = 0.85          # varyant çeşitliliği için yüksek
HTTP_TIMEOUT = 240

MAX_RETRIES = 3
RETRY_DELAY = 5.0

# Kalite kontrolü
MIN_WORDS         = 80
MIN_TAGS          = 3
MIN_TR_CHAR_RATIO = 0.03
TURKISH_CHARS     = set("çğıöşüÇĞİÖŞÜ")

# 5 lens directive — variant_idx ile deterministik eşleşir (resume güvenliği)
LENSES = [
    "Birinci kişi ağzından, kişisel ve içsel sahneler kullan.",
    "Doğa ve mevsim imgeleri kullan (yağmur, rüzgar, sonbahar, gece, deniz vb.).",
    "Şehir ve sokak sahneleri kullan (kalabalık, ışıklar, otobüs, kafe, tren vb.).",
    "Geçmişe dönüş ve anı imgeleri kullan (fotoğraflar, eski mektuplar, çocukluk, kayıp dostlar vb.).",
    "Soyut ve metaforik dil kullan (zaman, yıldızlar, ayna, gölge, sessizlik gibi semboller).",
]

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
Sen Türkçe şarkı sözü yazarısın. Sana verilen müzik parametrelerine uygun, özgün bir Türkçe şarkı sözü yazacaksın.

KURALLAR:
- ÇIKTIN SADECE şarkı sözü olsun. Hiçbir açıklama, başlık, JSON veya markdown ekleme.
- Yapı: [Verse 1] / [Chorus] / [Verse 2] / [Chorus] / [Bridge] / [Outro] etiketlerini kullan.
- En az 3 yapısal etiket bulunsun. Etiketleri köşeli parantezle yaz: [Verse 1] gibi.
- Sözler tamamen Türkçe olsun. Türkçe karakterleri (ç, ğ, ı, ö, ş, ü) doğal şekilde kullan.
- Min 100 kelime üret. Chorus tekrarlanabilir, ama her satırı kelimesi kelimesine birden fazla kez kopyalama.
- Verilen duygu, enerji ve enstrüman atmosferine uygun imgeler ve kelimeler seç.
- Müzik teorisi terimlerini (BPM, ton/key, enstrüman adı, "A minor", "ney", "keman" vb.) şarkı sözlerine ASLA yazma. Bunlar sadece atmosfer referansı, lyric'e katma.
- İmgeler somut ve Türkçeye yakışan deyişlerle olsun. Anlamsız/garip kelime üretme.

Aşağıdaki iki örnek istenen stil ve kalitenin referansıdır.

═══════ ÖRNEK 1 ═══════
Müzik parametreleri:
- Duygu: özlem
- Enerji: 5/10
- Tempo: 70 BPM
- Tonalite: A minor
- Enstrümanlar: ney, saz
- Vokal stili: kısık, dramatik

Yaratıcı yönelim: Birinci kişi ağzından, kişisel ve içsel sahneler kullan.

ÇIKTI:
[Verse 1]
Ne böyle seninle, ne de sensiz
Yazık, yaşanmıyor çaresiz
Ne bir arada, ne de ayrı
Olmak imkansız, hiç sebepsiz

[Chorus]
Ne hayallerle, ümitlerle
Mutlu olmaktı dileğimiz
Suçlu ne sensin ne de benim
Şimdi sensizim, sen de bensiz

[Bridge]
Etrafımızı sarıverecek
Bir boşluk ki asla bitmeyecek
Her şey bir anda anlamsız gelecek
İşte biz o gün tükeneceğiz

[Outro]
Tükeneceğiz, sen de bensiz

═══════ ÖRNEK 2 ═══════
Müzik parametreleri:
- Duygu: aşk
- Enerji: 7/10
- Tempo: 90 BPM
- Tonalite: A minor
- Enstrümanlar: ney, saz
- Vokal stili: dramatik, kısık

Yaratıcı yönelim: Geçmişe dönüş ve anı imgeleri kullan.

ÇIKTI:
[Verse 1]
Nihayet aşk kapımı araladı
Usul usul yanıma sokulup özüme daldı
Sen daha önceleri nerelerdeydin
Demlendim kollarında, ateşinde

[Chorus]
Teslim oldum sana bile bile
Kapıldım yine, göz göre göre
Başım dönüyor, kanım kaynıyor
Tiryakin oldum, yarim, sen olmadan duramıyorum

[Verse 2]
Yıllar öncesi solmuş bir fotoğraf
Senin gülüşün üstüne tutuyorum şimdi
Kayıp gittim aşkına, zalim
Düştüm yine ağına

[Bridge]
O eski mektupları yaktım hepsini
Ama küllerinden yine sen çıktın
Yıllar geçti, hiçbir şey değişmedi
İçimde aynı kor, aynı yangın

[Outro]
Tiryakin oldum, yarim

═══════ ÖRNEKLER BİTTİ ═══════

Şimdi sıradaki parametreler için aynı kalitede, ÖZGÜN bir şarkı sözü yaz.\
"""

USER_TEMPLATE = """\
Müzik parametreleri:
- Duygu: {emotion}
- Enerji: {energy}/10
- Tempo: {bpm} BPM
- Tonalite: {key}
- Enstrümanlar: {instruments}
- Vokal stili: {vocal_style}

Yaratıcı yönelim: {lens}

Şimdi sadece şarkı sözünü yaz."""

# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------

def fmt_instruments(v) -> str:
    if isinstance(v, list):
        return ", ".join(str(x) for x in v) or "akustik gitar"
    return str(v) if v else "akustik gitar"


def call_ollama(system: str, user: str) -> str | None:
    """HTTP /api/generate çağrısı. Başarısızsa None."""
    payload = {
        "model": MODEL,
        "system": system,
        "prompt": user,
        "stream": False,
        "options": {
            "num_ctx":     NUM_CTX,
            "num_predict": NUM_PREDICT,
            "num_gpu":     NUM_GPU,
            "temperature": TEMPERATURE,
        },
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            response = (data.get("response") or "").strip()
            if response:
                return response
            log.warning("  Boş yanıt (deneme %d/%d)", attempt, MAX_RETRIES)
        except requests.exceptions.RequestException as e:
            log.warning("  HTTP hata (deneme %d/%d): %s", attempt, MAX_RETRIES, e)
        except (json.JSONDecodeError, KeyError) as e:
            log.warning("  Yanıt parse hata (deneme %d/%d): %s", attempt, MAX_RETRIES, e)

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)
    return None


def strip_markdown(text: str) -> str:
    """ ``` blokları varsa içeriği koru, çevresini at."""
    text = re.sub(r"```(?:\w+)?\n?(.*?)\n?```", r"\1", text, flags=re.DOTALL)
    return text.strip()


TAG_RE = re.compile(r"\[[A-Za-zÇĞİÖŞÜçğıöşü][A-Za-zÇĞİÖŞÜçğıöşü\s\-0-9]*\]")


def count_tags(text: str) -> int:
    """Toplam yapısal etiket sayısı (instance bazlı, unique değil)."""
    return len(TAG_RE.findall(text))


def turkish_char_ratio(text: str) -> float:
    alpha = sum(1 for c in text if c.isalpha())
    if alpha == 0:
        return 0.0
    tr = sum(1 for c in text if c in TURKISH_CHARS)
    return tr / alpha


def word_count_lyrics(text: str) -> int:
    """Etiketleri çıkardıktan sonraki saf söz kelime sayısı."""
    stripped = TAG_RE.sub(" ", text)
    return len(stripped.split())


def validate_lyrics(text: str) -> tuple[bool, str]:
    if not text:
        return False, "boş çıktı"
    wc = word_count_lyrics(text)
    if wc < MIN_WORDS:
        return False, f"kelime sayısı: {wc} < {MIN_WORDS}"
    tc = count_tags(text)
    if tc < MIN_TAGS:
        return False, f"etiket sayısı: {tc} < {MIN_TAGS}"
    tr = turkish_char_ratio(text)
    if tr < MIN_TR_CHAR_RATIO:
        return False, f"Türkçe karakter oranı: {tr:.2%} < {MIN_TR_CHAR_RATIO:.0%}"
    return True, ""


def build_input_text(rec: dict) -> str:
    """Model 2'nin inference input formatıyla birebir aynı."""
    return (
        f"Türkçe şarkı sözü yaz: "
        f"duygu={rec.get('emotion','')}, "
        f"enerji={rec.get('energy',5)}, "
        f"tempo={rec.get('bpm',90)} BPM, "
        f"ton={rec.get('key','A minor')}, "
        f"enstrümanlar={fmt_instruments(rec.get('instruments',[]))}, "
        f"vokal={rec.get('vocal_style','')}"
    )


def load_processed_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    if not path.exists():
        return keys
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
            keys.add(f"{rec.get('artist','')}|||{rec.get('title','')}|||{rec.get('variant_idx',0)}")
        except json.JSONDecodeError:
            pass
    return keys


def check_ollama() -> bool:
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=5)
        r.raise_for_status()
        models = [m.get("name", "") for m in r.json().get("models", [])]
        if not any(MODEL in m for m in models):
            log.error("Model '%s' Ollama'da yok. Mevcut: %s", MODEL, models)
            log.error("Yükle: ollama pull %s", MODEL)
            return False
        log.info("Ollama OK — model=%s, num_ctx=%d, num_predict=%d, num_gpu=%d, temp=%.2f",
                 MODEL, NUM_CTX, NUM_PREDICT, NUM_GPU, TEMPERATURE)
        return True
    except Exception as e:
        log.error("Ollama erişilemez: %s", e)
        log.error("Çalıştır: ollama serve")
        return False


# ---------------------------------------------------------------------------
# Ana pipeline
# ---------------------------------------------------------------------------

def run(limit: int | None, variants: int) -> None:
    if variants < 1 or variants > len(LENSES):
        log.error("variants 1-%d arasında olmalı (lens sayısı).", len(LENSES))
        return

    if not check_ollama():
        return

    if not INPUT_PATH.exists():
        log.error("Girdi dosyası yok: %s", INPUT_PATH)
        return

    records: list[dict] = []
    for line in INPUT_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    log.info("Kaynak kayıt: %d", len(records))

    if limit:
        records = records[:limit]
        log.info("Limit: ilk %d kayıt", limit)

    processed = load_processed_keys(OUTPUT_PATH)
    if processed:
        log.info("Daha önce işlenmiş %d varyant atlanacak.", len(processed))

    total_jobs = len(records) * variants
    log.info("Hedef varyant: %d (kayıt %d × varyant %d)", total_jobs, len(records), variants)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_fh = OUTPUT_PATH.open("a", encoding="utf-8")
    success = fail = skipped = 0
    t0 = time.time()

    try:
        for ri, rec in enumerate(records, 1):
            artist = rec.get("artist", "")
            title  = rec.get("title", "")

            for vi in range(variants):
                key = f"{artist}|||{title}|||{vi}"
                if key in processed:
                    skipped += 1
                    continue

                lens = LENSES[vi]
                user_prompt = USER_TEMPLATE.format(
                    emotion=rec.get("emotion", ""),
                    energy=rec.get("energy", 5),
                    bpm=rec.get("bpm", 90),
                    key=rec.get("key", "A minor"),
                    instruments=fmt_instruments(rec.get("instruments", [])),
                    vocal_style=rec.get("vocal_style", ""),
                    lens=lens,
                )

                tstart = time.time()
                raw = call_ollama(SYSTEM_PROMPT, user_prompt)
                dt = time.time() - tstart

                if raw is None:
                    log.error("[%d/%d v%d] %s — %s | FAIL no-response (%.1fs)",
                              ri, len(records), vi, artist, title, dt)
                    fail += 1
                    continue

                text = strip_markdown(raw)
                ok, reason = validate_lyrics(text)
                if not ok:
                    log.warning("[%d/%d v%d] %s — %s | INVALID %s (%.1fs)",
                                ri, len(records), vi, artist, title, reason, dt)
                    fail += 1
                    continue

                entry = {
                    "artist":      artist,
                    "title":       title,
                    "category":    rec.get("category", ""),
                    "variant_idx": vi,
                    "lens":        lens,
                    "input_text":  build_input_text(rec),
                    "target_text": text,
                }
                out_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                out_fh.flush()
                os.fsync(out_fh.fileno())

                success += 1
                log.info("[%d/%d v%d] %s — %s | OK %.1fs, %d words",
                         ri, len(records), vi, artist, title, dt, word_count_lyrics(text))

        elapsed = time.time() - t0
        log.info("=" * 60)
        log.info("Başarılı: %d | Başarısız: %d | Atlanan: %d", success, fail, skipped)
        log.info("Süre: %.1fs (%.2f saat)", elapsed, elapsed / 3600)
        if success > 0:
            log.info("Ortalama: %.1fs/varyant", elapsed / max(success, 1))
        log.info("Çıktı: %s", OUTPUT_PATH)
    finally:
        out_fh.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ollama (HTTP API) ile Türkçe şarkı sözü augmentation."
    )
    parser.add_argument("--limit", type=int, help="İşlenecek kayıt sayısı (smoke test).")
    parser.add_argument("--variants", type=int, default=5,
                        help="Kayıt başına varyant sayısı (default: 5, max: %d)." % len(LENSES))
    args = parser.parse_args()
    run(limit=args.limit, variants=args.variants)


if __name__ == "__main__":
    main()
