"""
Genius API'den Türkçe şarkı sözü toplama scripti.
Her sanatçıdan max 20-30 ünlü şarkı çeker, data/raw/ klasörüne JSONL olarak kaydeder.
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

import lyricsgenius
from dotenv import load_dotenv

# Windows konsolunda Türkçe karakterlerin doğru görünmesi için
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
# Sanatçı Listesi
# ---------------------------------------------------------------------------
ARTISTS: dict[str, list[str]] = {
    "turk_pop": [
        "Sezen Aksu",
        "Tarkan",
        "Hadise",
        "Kenan Doğulu",
        "Ajda Pekkan",
        "Nilüfer",
        "Kayahan",
        "Yıldız Tilbe",
        "Hande Yener",
    ],
    "arabesk": [
        "Müslüm Gürses",
        "İbrahim Tatlıses",
        "Ferdi Tayfur",
        "Orhan Gencebay",
        "Bülent Ersoy",
    ],
    "rock": [
        "Erkin Koray",
        "Cem Karaca",
        "Duman",
        "Mor ve Ötesi",
        "Athena",
        "Pinhani",
        "Manga",
        "Teoman",
        "Gripin",
    ],
    "folk_halk": [
        "Aşık Veysel",
        "Zülfü Livaneli",
        "Barış Manço",
        "Selda Bağcan",
    ],
    "turku": [
        "Neşet Ertaş",
        "Yavuz Bingöl",
        "Belkıs Akkale",
        "Mahsuni Şerif",
        "Muazzez Ersoy",
    ],
    "sanat_muzigi": [
        "Zeki Müren",
        "Münir Nurettin Selçuk",
    ],
    "hip_hop": [
        "Ceza",
        "Ezhel",
        "Norm Nar",
        "Ben Fero",
        "Sagopa Kajmer",
        "Şanışer",
    ],
}

MAX_SONGS_PER_ARTIST = 25
RATE_LIMIT_DELAY = 5.0      # saniye — sanatçılar arası bekleme (rate limit koruması)
RETRY_DELAY = 20.0          # saniye — hata sonrası bekleme (exponential backoff tabanı)
DNS_BLOCK_DELAY = 180.0     # saniye — DNS hatası = IP engeli, uzun bekle
MAX_RETRIES = 4

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------

def make_genius_client(token: str) -> lyricsgenius.Genius:
    genius = lyricsgenius.Genius(
        token,
        timeout=30,              # 15'ten artırıldı — yavaş bağlantılar için
        retries=0,               # kendi retry mantığımızı kullanıyoruz, çift retry'ı önler
        verbose=False,           # kütüphane kendi print'ini kapatıyoruz
        remove_section_headers=False,  # [Verse] gibi etiketleri koruyoruz
        skip_non_songs=True,
        excluded_terms=["(Remix)", "(Live)", "(Acoustic)", "(Cover)", "(Karaoke)"],
    )
    genius.sleep_time = RATE_LIMIT_DELAY
    return genius


def sanitize_lyrics(raw: str) -> str:
    """Genius metadata artıklarını kaldır, metni temizle."""
    lines = raw.splitlines()
    clean = []
    skip_phrases = [
        "You might also like",
        "Embed",
        "See ",
        "Contributors",
        "Translations",
    ]
    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(p) for p in skip_phrases):
            continue
        # "123Embed" gibi sayı+Embed kalıplarını at
        if stripped.endswith("Embed") and stripped[:-5].strip().isdigit():
            continue
        clean.append(line)
    return "\n".join(clean).strip()


def fetch_artist_songs(
    genius: lyricsgenius.Genius,
    artist_name: str,
    max_songs: int,
) -> list[dict]:
    """Tek bir sanatçının şarkılarını çeker. Liste içinde sözlük döner."""
    results = []
    log.info("  Aranıyor: %s (max %d şarkı)", artist_name, max_songs)

    artist = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            artist = genius.search_artist(
                artist_name,
                max_songs=max_songs,
                sort="popularity",
                include_features=False,
            )
            break
        except Exception as exc:
            exc_str = str(exc).lower()
            is_dns = any(k in exc_str for k in ("getaddrinfo", "nameresolution", "failed to resolve"))
            # DNS hatası = IP geçici engellendi, çok daha uzun bekle
            wait = DNS_BLOCK_DELAY if is_dns else RETRY_DELAY * (2 ** (attempt - 1))
            log.warning("  Deneme %d başarısız (%s): %s", attempt, artist_name, exc)
            if attempt < MAX_RETRIES:
                log.info("  %.0f saniye bekleniyor%s...", wait,
                         " (DNS/IP engeli tespit edildi)" if is_dns else "")
                time.sleep(wait)
            else:
                log.error("  %s için tüm denemeler tükendi, atlanıyor.", artist_name)
                return results

    if artist is None:
        log.warning("  Sanatçı bulunamadı: %s", artist_name)
        return results

    for song in artist.songs:
        if song.lyrics is None:
            continue
        lyrics = sanitize_lyrics(song.lyrics)
        if not lyrics:
            continue
        results.append({
            "artist": artist_name,
            "title": song.title,
            "url": song.url,
            "lyrics": lyrics,
            "collected_at": datetime.utcnow().isoformat() + "Z",
        })

    log.info("  %s → %d şarkı bulundu.", artist_name, len(results))
    return results


def append_jsonl(records: list[dict], path: Path) -> None:
    """Kayıtları dosyaya ekler (overwrite değil, append). Crash-safe ilerleme için."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_existing_artists(raw_dir: Path) -> set[str]:
    """Daha önce çekilmiş sanatçıların adını döner (yeniden çekmemek için)."""
    seen = set()
    for jsonl_file in raw_dir.glob("*.jsonl"):
        with jsonl_file.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    seen.add(rec.get("artist", ""))
                except json.JSONDecodeError:
                    pass
    return seen


# ---------------------------------------------------------------------------
# Ana mantık
# ---------------------------------------------------------------------------

def collect(
    categories: list[str] | None = None,
    skip_existing: bool = True,
    dry_run: bool = False,
) -> None:
    load_dotenv()
    token = os.getenv("GENIUS_ACCESS_TOKEN")
    if not token:
        raise EnvironmentError(
            "GENIUS_ACCESS_TOKEN bulunamadı. "
            ".env dosyasına ekleyin veya ortam değişkeni olarak tanımlayın."
        )

    genius = make_genius_client(token)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    existing_artists = load_existing_artists(RAW_DIR) if skip_existing else set()
    if existing_artists:
        log.info("Daha önce çekilmiş %d sanatçı atlanacak.", len(existing_artists))

    target_categories = categories or list(ARTISTS.keys())
    total_songs = 0
    total_artists = 0

    for category in target_categories:
        if category not in ARTISTS:
            log.warning("Bilinmeyen kategori: %s, atlanıyor.", category)
            continue

        artist_list = ARTISTS[category]
        log.info("=" * 60)
        log.info("Kategori: %s (%d sanatçı)", category, len(artist_list))
        log.info("=" * 60)

        out_path = RAW_DIR / f"{category}.jsonl"

        for artist_name in artist_list:
            if skip_existing and artist_name in existing_artists:
                log.info("  Atlanıyor (zaten mevcut): %s", artist_name)
                continue

            if dry_run:
                log.info("  [DRY RUN] %s çekilecekti.", artist_name)
                continue

            songs = fetch_artist_songs(genius, artist_name, MAX_SONGS_PER_ARTIST)
            for song in songs:
                song["category"] = category

            if songs:
                # Her sanatçı çekildikten hemen sonra diske yaz — crash-safe
                append_jsonl(songs, out_path)
                existing_artists.add(artist_name)  # aynı çalışmada tekrar atla
                log.info("  Kaydedildi: %s (%d şarkı → %s)",
                         artist_name, len(songs), out_path.name)

            total_artists += 1
            total_songs += len(songs)

            # Her sanatçıdan sonra bekleme (Genius rate limit)
            time.sleep(RATE_LIMIT_DELAY)

    log.info("=" * 60)
    log.info(
        "Tamamlandı. Toplam: %d sanatçı, %d şarkı.",
        total_artists,
        total_songs,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genius API'den Türkçe şarkı sözü topla."
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=list(ARTISTS.keys()),
        metavar="KATEGORİ",
        help=(
            "Sadece belirtilen kategorileri çek. "
            f"Seçenekler: {', '.join(ARTISTS.keys())}"
        ),
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Daha önce çekilmiş sanatçıları da yeniden çek.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="API'ye bağlanmadan sadece neyin çekileceğini göster.",
    )
    args = parser.parse_args()

    collect(
        categories=args.categories,
        skip_existing=not args.no_skip,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
