# Story-to-Music MCP Sunucusu

Türkçe metinlerin (hikaye, senaryo, konsept veya şarkı sözü) duygu haritasını analiz ederek **Suno** ve **Udio** gibi yapay zeka müzik platformları için kullanıma hazır, yapılandırılmış müzik promptları ve Türkçe şarkı sözleri üreten **Model Context Protocol (MCP)** tabanlı otonom bir yapay zeka aracı.

---

## Nasıl Çalışır?

```
Türkçe Metin → Fine-Tuned T5-small Model → Müzik Parametreleri + Yapılandırılmış Şarkı Sözü
```

### Mod 1 — Custom (Tam Çıktı)
Suno/Udio'da "Custom" sekmesi açıkken kullanılır. Şunları üretir:
- **Style prompt** → `melancholic Turkish arabesque, 75 BPM, A minor, slow strings, ney flute, male vocal`
- **Yapılandırılmış şarkı sözü** → `[Verse]`, `[Chorus]`, `[Bridge]`, `[Outro]` etiketleriyle Türkçe sözler
- **Başlık önerileri** → 2-3 alternatif başlık

### Mod 2 — Prompt Only (Hızlı Prompt)
Tek satırlık, platforma optimize edilmiş prompt üretir:
> `sad Turkish pop ballad with ney flute and piano, melancholic female vocals, 70 BPM`

---

## Kurulum (Docker — Önerilen)

```bash
docker run --rm -i story-to-music-mcp:latest
```

Claude, Cursor veya başka bir MCP istemcisinin config dosyasına bu satırı ekleyin. Başka bir kurulum adımı yoktur.

---

## Geliştirici Kurulumu

```bash
git clone https://github.com/YOUR_USERNAME/story-to-music.git
cd story-to-music
cp .env.example .env
# .env dosyasına API anahtarlarınızı ekleyin
```

### Ortam Değişkenleri

| Değişken | Açıklama | Zorunlu |
|---|---|---|
| `GENIUS_ACCESS_TOKEN` | Genius API token (veri toplama) | Hayır (sadece data pipeline) |
| `GROQ_API_KEY` | Groq API key (prompt refinement) | Hayır (prompt_only modu için) |
| `MODEL_PATH` | Fine-tuned model dizini | Evet |

---

## Proje Yapısı

```
story-to-music/
├── scripts/                # Veri toplama ve temizleme
│   ├── collect_lyrics.py   # Genius API scraper
│   ├── clean_lyrics.py     # Filtreleme ve normalizasyon
│   └── distill_data.py     # Groq ile sentetik veri üretimi
│
├── training/               # Model eğitimi
│   ├── train.py
│   └── evaluate.py
│
├── server/                 # MCP sunucu
│   ├── main.py
│   ├── tools.py
│   ├── inference.py
│   ├── groq_client.py
│   ├── content_guard.py
│   └── copyright_guard.py
│
├── docker/
│   └── Dockerfile
│
├── data/                   # Veri (git'e dahil edilmez)
│   ├── raw/
│   ├── cleaned/
│   └── training/
│
└── model/                  # Model ağırlıkları (git'e dahil edilmez)
    └── story-to-music-t5/
```

---

## Tech Stack

| Katman | Teknoloji |
|---|---|
| MCP Sunucu | Python + `mcp` SDK |
| NLP Modeli | `google/flan-t5-small` (fine-tuned) |
| Prompt Refinement | Groq API (Llama 3.1) |
| Veri Toplama | `lyricsgenius` + Genius API |
| Konteyner | Docker |

---

## Geliştirme Aşamaları

- [x] Proje planı ve mimari
- [x] `collect_lyrics.py` — Genius API scraper
- [ ] `clean_lyrics.py` — Temizleme pipeline'ı
- [ ] `distill_data.py` — Groq ile veri distillation
- [ ] `train.py` — T5 fine-tuning
- [ ] MCP sunucu (`server/`)
- [ ] Docker imajı
- [ ] Docker Hub yayını

---

## Lisans

MIT
