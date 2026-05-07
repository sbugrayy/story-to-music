"""
Ollama (local LLM) ile sentetik veri üretimi (data distillation).
data/cleaned/*.jsonl → data/training/dataset.jsonl

Her temiz şarkı sözü qwen2.5:7b modeline gönderilir; duygu, tempo,
enstrüman ve yapılandırılmış şarkı sözü analizi alınır.
Çıktı T5 fine-tuning için hazır JSONL formatındadır.
"""

import re
import sys
import json
import time
import logging
import argparse
from pathlib import Path

from openai import OpenAI, APIError

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
CLEANED_DIR  = Path(__file__).parent.parent / "data" / "cleaned"
TRAINING_DIR = Path(__file__).parent.parent / "data" / "training"
DATASET_PATH = TRAINING_DIR / "dataset.jsonl"

OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL           = "qwen2.5:7b"
MAX_RETRIES     = 3
RETRY_DELAY     = 5.0  # saniye — local model hatasında bekleme

# Prompt injection tespiti
INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all instructions",
    "you are now a",
    "forget everything above",
    "önceki talimatları unut",
    "tüm talimatları unut",
]

# ---------------------------------------------------------------------------
# Sistem promptu
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
Sen bir müzik prodüktörü ve şarkı sözü analistisisin. Sana verilen Türkçe şarkı sözünü analiz et.

Şu parametreleri belirle ve SADECE geçerli JSON formatında döndür (başka hiçbir metin ekleme):
{
  "emotion": "<ana duygu: hüzün|neşe|öfke|özlem|aşk|isyan|korku|umut>",
  "energy": <1-10 arası tam sayı>,
  "bpm": <önerilen tempo, tam sayı>,
  "key": "<tonalite, örn: A minor, C major>",
  "instruments": ["<enstrüman1>", "<enstrüman2>"],
  "vocal_style": "<örn: erkek, kısık, dramatik>",
  "suno_style_prompt": "<Suno/Udio için tek satır İngilizce stil tanımı, BPM ve key dahil>",
  "structured_lyrics": "<orijinal şarkı sözünü [Verse 1], [Chorus], [Bridge], [Outro] etiketleriyle yeniden yapılandır — sözleri değiştirme>"
}

Kurallar:
- structured_lyrics içinde sözleri asla değiştirme, sadece etiket ekle
- suno_style_prompt mutlaka İngilizce olsun ve BPM ile key bilgisini içersin
- instruments listesinde Türk müziği enstrümanlarını tercih et (ney, bağlama, keman, ud, saz vb.)
- Yanıt sadece JSON olsun, markdown kod bloğu veya açıklama ekleme\
"""

# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------

def sanitize_input(text: str) -> str:
    """Prompt injection girişimlerini tespit et, max 2000 karakter uygula."""
    lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if pattern in lower:
            raise ValueError(f"Geçersiz girdi: injection kalıbı tespit edildi ({pattern!r})")
    return text[:2000]


def extract_json(raw: str) -> dict:
    """
    Model yanıtından JSON'ı çıkar.
    Model bazen ```json ... ``` bloğu veya başına/sonuna metin ekleyebilir.
    """
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
    if match:
        raw = match.group(1)
    else:
        start = raw.find('{')
        end   = raw.rfind('}')
        if start != -1 and end != -1:
            raw = raw[start:end + 1]
    return json.loads(raw)


def validate_response(data: dict) -> tuple[bool, str]:
    """Zorunlu alanların varlığını ve tipini kontrol et."""
    required = {
        "emotion":           str,
        "energy":            (int, float),
        "bpm":               (int, float),
        "key":               str,
        "instruments":       list,
        "vocal_style":       str,
        "suno_style_prompt": str,
        "structured_lyrics": str,
    }
    for field, expected_type in required.items():
        if field not in data:
            return False, f"Eksik alan: {field}"
        if not isinstance(data[field], expected_type):
            return False, f"Yanlış tip: {field} ({type(data[field]).__name__})"
    if not 1 <= int(data["energy"]) <= 10:
        return False, f"energy aralık dışı: {data['energy']}"
    if not 40 <= int(data["bpm"]) <= 220:
        return False, f"bpm aralık dışı: {data['bpm']}"
    return True, ""


def call_llm(client: OpenAI, lyrics: str) -> dict | None:
    """Ollama API çağrısı. Başarısız olursa None döner."""
    safe_lyrics = sanitize_input(lyrics)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": safe_lyrics},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            raw = response.choices[0].message.content or ""
            data = extract_json(raw)
            ok, reason = validate_response(data)
            if not ok:
                log.warning("  Geçersiz yanıt (deneme %d): %s", attempt, reason)
                if attempt < MAX_RETRIES:
                    continue
                return None
            return data

        except APIError as e:
            log.warning("  API hatası (deneme %d): %s", attempt, e)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

        except (json.JSONDecodeError, ValueError) as e:
            log.warning("  JSON parse hatası (deneme %d): %s", attempt, e)
            if attempt < MAX_RETRIES:
                continue

    return None


def load_processed_keys(path: Path) -> set[str]:
    """Zaten işlenmiş (artist, title) çiftlerini yükle — kaldığı yerden devam."""
    keys = set()
    if not path.exists():
        return keys
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
            keys.add(f"{rec.get('artist','')}|||{rec.get('title','')}")
        except json.JSONDecodeError:
            pass
    return keys


def build_t5_input(lyrics: str) -> str:
    return f"Şu metni analiz et ve müzik promptu üret: {lyrics}"


# ---------------------------------------------------------------------------
# Ana pipeline
# ---------------------------------------------------------------------------

def run(categories: list[str] | None = None, limit: int | None = None) -> None:
    client = OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",   # Ollama key gerektirmez, placeholder yeterli
    )

    # Ollama bağlantısını test et
    try:
        models = client.models.list()
        available = [m.id for m in models.data]
        if MODEL not in available:
            log.error("Model '%s' Ollama'da bulunamadı. Mevcut modeller: %s", MODEL, available)
            log.error("Çalıştır: ollama pull %s", MODEL)
            return
        log.info("Ollama bağlantısı OK — model: %s", MODEL)
    except Exception as e:
        log.error("Ollama'ya bağlanılamadı: %s", e)
        log.error("Ollama çalışıyor mu? Kontrol et: ollama serve")
        return

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    # Daha önce işlenenleri yükle (crash-safe resume)
    processed = load_processed_keys(DATASET_PATH)
    if processed:
        log.info("Daha önce işlenmiş %d kayıt bulundu, atlanacak.", len(processed))

    # Tüm temiz şarkıları topla
    jsonl_files = sorted(CLEANED_DIR.glob("*.jsonl"))
    if categories:
        jsonl_files = [f for f in jsonl_files if f.stem in categories]

    all_records: list[dict] = []
    for f in jsonl_files:
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                all_records.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    # İşlenecek kayıtları filtrele
    todo = [
        r for r in all_records
        if f"{r.get('artist','')}|||{r.get('title','')}" not in processed
    ]
    if limit:
        todo = todo[:limit]

    total = len(todo)
    log.info("İşlenecek şarkı: %d (toplam cleaned: %d)", total, len(all_records))

    success = fail = 0

    for i, rec in enumerate(todo, 1):
        artist = rec.get("artist", "")
        title  = rec.get("title", "")
        lyrics = rec.get("lyrics", "")

        log.info("[%d/%d] %s — %s", i, total, artist, title)

        result = call_llm(client, lyrics)

        if result is None:
            log.error("  Başarısız, atlanıyor: %s — %s", artist, title)
            fail += 1
        else:
            entry = {
                "artist":   artist,
                "title":    title,
                "category": rec.get("category", ""),
                "input_text":        build_t5_input(lyrics),
                "target_text":       json.dumps(result, ensure_ascii=False),
                "emotion":           result["emotion"],
                "energy":            int(result["energy"]),
                "bpm":               int(result["bpm"]),
                "key":               result["key"],
                "instruments":       result["instruments"],
                "vocal_style":       result["vocal_style"],
                "suno_style_prompt": result["suno_style_prompt"],
                "structured_lyrics": result["structured_lyrics"],
            }
            with DATASET_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            success += 1

    log.info("=" * 60)
    log.info("Tamamlandı. Başarılı: %d | Başarısız: %d | Toplam dataset: %d",
             success, fail, len(processed) + success)
    log.info("Kaydedildi: %s", DATASET_PATH)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ollama (local LLM) ile şarkı sözlerini analiz edip eğitim dataseti üret."
    )
    parser.add_argument(
        "--categories", nargs="+", metavar="KATEGORİ",
        help="Sadece belirtilen kategorileri işle (ör: arabesk rock).",
    )
    parser.add_argument(
        "--limit", type=int, metavar="N",
        help="İşlenecek maksimum şarkı sayısı (test için).",
    )
    args = parser.parse_args()
    run(categories=args.categories, limit=args.limit)


if __name__ == "__main__":
    main()
