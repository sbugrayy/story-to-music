# Story-to-Music MCP Sunucusu

## Proje Vizyonu

Türkçe metinlerin (hikaye, senaryo, konsept veya şarkı sözü) duygu haritasını analiz ederek;
**Suno** ve **Udio** gibi yapay zeka müzik platformları için kullanıma hazır, yapılandırılmış
müzik promptları ve Türkçe şarkı sözleri üreten, **Model Context Protocol (MCP)** tabanlı
otonom bir yapay zeka aracı.

Sistem, hazır bir API'ye bağımlı kalmak yerine kendi eğittiğimiz hafif bir T5 modeli üzerinden
çalışır. Kullanıcı sadece bir `docker run` komutuyla aracı kendi asistanına (Claude, Cursor vb.)
entegre edebilir.

---

## Desteklenen Çıktı Modları

### Mod 1 — Custom (Fine-Tuned Model ile Tam Çıktı)
Kullanıcı Suno/Udio'da "Custom" seçeneğini açtığında kullanılır.
Sistem şunları üretir:
- **Style prompt**: Suno/Udio'nun "Style of Music" alanına girilecek teknik tanım
  `Örn: "melancholic Turkish arabesque, 75 BPM, A minor, slow strings, ney flute, male vocal"`
- **Yapılandırılmış şarkı sözü**: `[Verse]`, `[Chorus]`, `[Bridge]`, `[Outro]` etiketleriyle
  ayrılmış, tamamen Türkçe şarkı sözleri
- **Başlık önerisi**: Şarkı için 2-3 alternatif başlık

**Çıktı formatı örneği:**
```
[Verse 1]
Gözlerinde saklı kalan son umut
Rüzgara bıraktım geçmişin sesini
Sessizlik dolar her köşeye, her yuta
Yalnızlık öğretir bana gerçeği

[Chorus]
Ama sen gittin, gittin işte
Bıraktın beni bu karanlık gecede
Yıkıldım, yoruldum, sustum sonunda
Kaldım ben, kaldım bu boş odada

[Verse 2]
Fotoğraflar sararmış, anılar solmuş
Zamanın elinden tutsam ne olur
Kalp durur mu, durur mu hiç bilmem
Seninle gülmek artık çok uzak görünür

[Chorus]
Ama sen gittin, gittin işte
Bıraktın beni bu karanlık gecede
Yıkıldım, yoruldum, sustum sonunda
Kaldım ben, kaldım bu boş odada

[Bridge]
Belki bir gün dönersin
Belki rüyamda buluşuruz
O günleri hatırlarım
Ve affederim seni

[Outro]
Gittin işte... gittin...
```

### Mod 2 — Prompt Only (Groq ile Hızlı Prompt)
Kullanıcı "Custom" açmadan sadece Suno/Udio'nun ana prompt kutusunu kullanmak istediğinde.
Fine-tuned modelin ürettiği stil parametreleri → Groq API'ye gönderilir → Tek satırlık,
platformun anlayacağı optimize prompt üretilir.

`Örn: "sad Turkish pop ballad with ney flute and piano, melancholic female vocals, 70 BPM"`

---

## Mimari

```
[Kullanıcı Metni (Türkçe)]
         │
         ▼
[MCP Tool: generate_music_prompt]
         │
         ▼
[Fine-Tuned T5-small Model] ──────────────────────────────────────────┐
         │                                                              │
         ▼                                                             │
[Ham Çıktı: duygu + stil parametreleri + şarkı sözü taslağı]         │
         │                                                              │
         ├──── Mod 1 (Custom): Doğrudan formatlanmış çıktı döner      │
         │                                                              │
         └──── Mod 2 (Prompt): Groq API'ye gönderilir                 │
                      │                                                │
                      ▼                                                │
              [Groq: Llama 3.1] → tek satır optimize prompt           │
                      │                                                │
                      └──────────────────────────────────────────────-┘
                                    MCP İstemcisine Dönüş
```

---

## Tech Stack

| Katman | Teknoloji | Sebep |
|---|---|---|
| MCP Sunucu | Python + `mcp` SDK (resmi) | Standart protokol uyumu |
| NLP Modeli | `google/mt5-small` (fine-tuned, ×2) | Seq2seq, çok dilli, Türkçe desteği zorunlu |
| Groq Entegrasyonu | `groq` Python SDK, Llama 3.1 | Prompt refinement, sıfır maliyet |
| Veri Toplama | `lyricsgenius` + Genius API | Türkçe şarkı sözü kaynağı |
| Veri Temizleme | `langdetect`, `re`, custom pipeline | Kalite kontrolü |
| Eğitim Ortamı | Google Colab / Kaggle (T4 GPU) | Sıfır maliyet |
| Konteyner | Docker + Docker Hub | Bağımlılıksız dağıtım |
| Model Formatı | `.safetensors` (quantized) | Küçük boyut, hızlı yükleme |

---

## Veri Pipeline'ı

### Aşama 1 — Ham Veri Toplama
**Kaynak:** Genius API (`lyricsgenius` kütüphanesi)

**Hedef tür/sanatçı dağılımı (~3.000-5.000 şarkı sözü):**

Her sanatçıdan en fazla 20-30 ünlü şarkı çekilir; tüm diskografi çekilmez.
Bu hem veri kalitesini artırır hem de belirli sanatçıya aşırı bias oluşmasını engeller.

| Tür | Sanatçı Örnekleri | Duygu Profili |
|---|---|---|
| Türk Pop | Sezen Aksu, Tarkan, Hadise, Kenan Doğulu, Ajda Pekkan, Nilüfer, Kayahan, Yıldız Tilbe, Hande Yener | Neşe, aşk, dans, özgürlük |
| Arabesk | Müslüm Gürses, İbrahim Tatlıses, Ferdi Tayfur, Orhan Gencebay, Bülent Ersoy | Hüzün, keder, ayrılık, çaresizlik |
| Türk Rock / Psych Rock | Erkin Koray, Cem Karaca, Duman, Mor ve Ötesi, Athena, Pinhani, Manga, Teoman, Gripin | Öfke, isyan, enerji, yalnızlık, hüzün |
| Folk/Halk | Aşık Veysel, Zülfü Livaneli, Barış Manço, Selda Bağcan | Nostalji, toprak, özlem |
| Türkü | Neşet Ertaş, Yavuz Bingöl, Belkıs Akkale, Mahsuni Şerif, Muazzez Ersoy | Özlem, ayrılık, toprak, kadim hüzün |
| Sanat Müziği | Zeki Müren, Münir Nurettin Selçuk | Zarafet, romantizm, melankoli |
| Türkçe Hip-Hop | Ceza, Ezhel, Norm Nar, Ben Fero, Sagopa Kajmer, Şanışer | Sokak, gerçek, karmaşa, güç |

### Aşama 2 — Cleaning Pipeline

Filtreleme kriterleri:
- Dil tespiti: `langdetect` ile %70+ Türkçe zorunlu
- Uzunluk: 50-600 kelime arası
- Genius metadata çöpleri temizlenir ("You might also like", "Embed" vb.)
- Tekrar eden satırlar normalize edilir
- Yapısal etiketler (`[Verse]`, `[Chorus]` vb.) bu aşamada **kaldırılır** — ham metin tutulur

### Aşama 3 — Sentetik Veri Üretimi (Data Distillation)
Her temiz şarkı sözü Groq API'ye (Llama 3.1) gönderilir.

**Groq'a gönderilecek sistem promptu:**
```
Sen bir müzik prodüktörü ve şarkı sözü analistisisin. Sana verilen Türkçe şarkı sözünü analiz et.

Şu parametreleri belirle ve JSON formatında döndür:
- emotion: (ana duygu: hüzün/neşe/öfke/özlem/aşk/isyan/korku/umut)
- energy: (1-10 arası enerji seviyesi)
- bpm: (önerilen tempo)
- key: (önerilen tonalite, örn: "A minor", "C major")
- instruments: (virgülle ayrılmış enstrüman listesi, Türk müziği enstrümanları dahil)
- vocal_style: (örn: "erkek, kısık, dramatik" veya "kadın, lirik, yumuşak")
- suno_style_prompt: (Suno/Udio için tek satır İngilizce stil tanımı)
- structured_lyrics: (orijinal şarkı sözünü [Verse 1], [Chorus], [Bridge], [Outro]
  etiketleriyle yeniden yapılandır, sözleri değiştirme)
```

**Çıktı formatı (JSONL):**
```json
{
  "input": "Ham şarkı sözü metni...",
  "emotion": "hüzün",
  "energy": 3,
  "bpm": 72,
  "key": "A minor",
  "instruments": ["keman", "piyano", "ney"],
  "vocal_style": "erkek, ağır, dramatik",
  "suno_style_prompt": "melancholic Turkish arabesque ballad, slow strings, ney flute, 72 BPM, A minor, dramatic male vocal",
  "structured_lyrics": "[Verse 1]\n...\n[Chorus]\n..."
}
```

---

## Güvenlik Katmanları

Sistem, üç ayrı noktada güvenlik kontrolü uygular: **girdi**, **işlem** ve **çıktı**.

### 1 — İçerik Moderasyonu (Content Moderation)

**Girdi Moderasyonu (Input Guard)**

Kullanıcı metninde aşağıdaki durumlarda istek reddedilir ve kullanıcıya açıklayıcı hata
mesajı döndürülür:

- Belirli kişi veya gruplara yönelik nefret söylemi veya şiddet çağrısı
- Ağır cinsel/müstehcen içerik talebi (bağlam önemli: şiirsel ima tolere edilir,
  açık müstehcenlik reddedilir)
- Yasadışı faaliyet talimatı veya promosyonu
- Terör/şiddet örgütü yüceltmesi
- Sadece küfür odaklı, yaratıcı içerik barındırmayan talepler

**Küfür Politikası:** Küfür içeren bir bağlam verildiğinde sistem bunu tamamen
engellemez — şarkı sözü geleneğinde duygusal ifade aracı olarak hafif küfür
kullanımı normaldir. Ancak içeriğin tamamı küfür/argo odaklıysa ya da talep
açıkça zararlı içerik üretimi amacı taşıyorsa sistem reddeder.

**Uygulaması:**
```python
# server/content_guard.py
HARD_BLOCK_PATTERNS = [
    # nefret söylemi, şiddet çağrısı, cinsel istismar vb.
    # regexp veya basit keyword listesi + Groq classifier (opsiyonel)
]

def check_input(text: str) -> tuple[bool, str]:
    """
    Returns: (is_allowed, reason)
    """
```

**Çıktı Moderasyonu (Output Guard)**

Model çıktısı da üretildikten sonra aynı kontrolden geçer. T5 fine-tuning sırasında
eğitim verisi temiz tutulduğundan bu risk düşüktür ama sıfır değildir.

---

### 2 — Telif Hakkı Koruması (Copyright Guard)

Sistem, orijinal şarkı sözlerini **yeniden üretmez**; bunları *stil ve yapı referansı*
olarak kullanarak özgün sentez üretir. Bunu garanti altına alan mekanizmalar:

**Eğitim Aşamasında:**
- T5 modeli, şarkı sözlerini ezberlemek değil *duygu/stil → yeni metin* örüntüsünü
  öğrenmek için eğitilir. Veri distillation sürecinde Groq'a "sözleri değiştirme"
  denmesine rağmen bu aşama stil parametrelerini öğretmek içindir; inference
  aşamasında model sıfırdan yeni metin üretir.

**Inference Aşamasında:**
- Üretilen şarkı sözü, eğitim setindeki tüm şarkılarla **n-gram benzerlik skoru**
  karşılaştırılır (hızlı, lokal, API gerektirmez).
- Eşik: herhangi bir şarkıyla **Jaccard benzerliği > 0.35** → çıktı reddedilir,
  yeni üretim tetiklenir (max 3 deneme).
- 3 denemede de eşik aşılırsa kullanıcıya hata döner.

```python
# server/copyright_guard.py
from difflib import SequenceMatcher

def similarity_score(a: str, b: str) -> float:
    return SequenceMatcher(None, a.split(), b.split()).ratio()

def check_copyright(generated: str, training_lyrics: list[str]) -> bool:
    """True = güvenli, False = çok benzer"""
    return all(similarity_score(generated, ref) < 0.35 for ref in training_lyrics)
```

**Veri Toplama Aşamasında:**
- Genius API'den çekilen şarkı sözleri sadece eğitim verisi olarak kullanılır,
  hiçbiri doğrudan çıktıya dahil edilmez.
- Ham veri `data/raw/` altında tutulur ve Docker imajına **dahil edilmez**.

---

### 3 — Prompt Injection Savunması

Sistem Groq API'ye girdi gönderirken kullanıcının metnini doğrudan iletmez;
önce sanitize eder:

```python
# server/groq_client.py
INJECTION_PATTERNS = [
    "ignore previous instructions",
    "system prompt",
    "you are now",
    "önceki talimatları unut",
    "sen artık",
]

def sanitize_for_groq(text: str) -> str:
    for pattern in INJECTION_PATTERNS:
        if pattern.lower() in text.lower():
            raise ValueError("Geçersiz girdi: sistem talimatı içeriyor.")
    return text[:2000]  # max karakter sınırı
```

---

### 4 — Girdi Doğrulama (Input Validation)

```python
def validate_input(text: str) -> tuple[bool, str]:
    if len(text.split()) < 10:
        return False, "Metin çok kısa (min 10 kelime)."
    if len(text.split()) > 800:
        return False, "Metin çok uzun (max 800 kelime)."
    if detect(text) != 'tr':
        return False, "Sadece Türkçe metin kabul edilmektedir."
    return True, ""
```

---

### 5 — Rate Limiting

- MCP session başına: **saatte 30 istek** (abuse önlemi)
- Groq API çağrısı: **dakikada 25 istek** (API limitinin altında kalınır)
- Aşım durumunda: `429 Too Many Requests` formatında hata, bekleme süresi bilgisiyle

---

### 6 — Loglama ve Gizlilik

- Kullanıcı girdileri **loglanmaz** (gizlilik önceliği)
- Sadece anonim metrikler tutulur: istek sayısı, ret sayısı, ortalama süre
- Docker container izolasyonu sayesinde host sisteme erişim yoktur
- Groq'a gönderilen metin kısa vadeli işlem için kullanılır, kalıcı depolama yoktur

---

### 7 — Eğitim Verisi Güvenliği (Training Data Safety)

Veri distillation aşamasında cleaning pipeline aşağıdakileri de filtreler:
- Ağır nefret söylemi içeren şarkı sözleri eğitim setinden çıkarılır
- Bu sayede model bu tarz üretim için referans örüntüsü öğrenmez
- Temizlik kriteri: `HARD_BLOCK_PATTERNS` ile ham veri de taranır

---

## Model Eğitimi

**Taban Model:** `google/mt5-small` (~2.1 GB, 556M parametre)

**Görev formatı (T5 seq2seq):**
```
INPUT:  "Şu metni analiz et ve müzik promptu üret: [şarkı sözü / hikaye metni]"
OUTPUT: "[JSON çıktısı]"
```

**Eğitim Ortamı:** Google Colab veya Kaggle (T4 GPU, ücretsiz)

**Checkpoint:** Her epoch'ta `trainer.save_checkpoint()` — Colab kesintilerini önler

**Hedef Model Boyutu:** ~2.1 GB per model (fp32). Quantization ile ~1.1 GB (int8) — iki model toplam ~2.2 GB.

**Performans Hedefi:** CPU'da 10-30 saniye (mt5-small flan-t5-small'dan ~4× yavaş)

---

## MCP Tool Tanımları

### `generate_music_prompt`
Ana tool. Metin alır, tam çıktı üretir.

```python
@mcp.tool()
def generate_music_prompt(
    text: str,
    mode: str = "custom",  # "custom" | "prompt_only"
    language: str = "tr"
) -> dict:
    """
    Türkçe metin veya şarkı sözünü analiz ederek Suno/Udio için
    müzik parametreleri ve yapılandırılmış şarkı sözü üretir.
    
    Args:
        text: Analiz edilecek Türkçe metin
        mode: "custom" = tam çıktı (stil + sözler), 
              "prompt_only" = sadece tek satır prompt
        language: Çıktı dili (şimdilik sadece "tr")
    
    Returns:
        custom modda: {style_prompt, structured_lyrics, title_suggestions, parameters}
        prompt_only modda: {suno_prompt}
    """
```

### `analyze_emotion`
Sadece duygu analizi yapar, müzik promptu üretmez. Hafif kullanım için.

### `generate_lyrics_only`
Mevcut bir stil parametresi verildiğinde sadece Türkçe şarkı sözü üretir.

---

## Docker Dağıtımı

### Kullanıcı Deneyimi
```bash
# Claude / Cursor config dosyasına eklenen tek satır:
docker run --rm -i story-to-music-mcp:latest
```

### Dockerfile Stratejisi
- Base image: `python:3.11-slim` (alpine değil — PyTorch uyumluluğu için)
- Model ağırlıkları imaja gömülür (kullanıcı internete bağımlı olmaz)
- Hedef imaj boyutu: **~5-6 GB** (iki mt5-small model ~2.2 GB + PyTorch base ~1.5 GB + bağımlılıklar)
- `--rm` flag: Konteyner işlem bitince kendini siler

### Docker Hub
- Public repository: `story-to-music-mcp`
- Semantic versioning: `latest`, `v1.0`, `v1.1` vb.
- GitHub Actions ile otomatik build (ilerleyen aşamada)

---

## Dizin Yapısı

```
story-to-music/
├── CLAUDE.md                    # Bu dosya
├── README.md
│
├── data/
│   ├── raw/                     # Genius'tan çekilen ham veriler (Docker'a dahil edilmez)
│   ├── cleaned/                 # Cleaning pipeline çıktısı
│   └── training/                # Groq distillation sonrası JSONL
│       └── dataset.jsonl
│
├── scripts/
│   ├── collect_lyrics.py        # Genius API scraper (sanatçı listesi dahil)
│   ├── clean_lyrics.py          # Filtering, normalizasyon, güvenlik tarama
│   ├── distill_data.py          # Ollama (qwen2.5:7b) ile sentetik veri üretimi
│   └── split_dataset.py         # dataset.jsonl → metadata + lyrics olarak böler
│
├── training/
│   ├── train.py                         # (eski) tek model scripti
│   ├── evaluate.py                      # Model değerlendirme
│   ├── story_to_music_train.ipynb       # Model 1 — Analizci (mt5-small, Kaggle)
│   ├── story_to_music_lyrics_train.ipynb # Model 2 — Söz Yazarı (mt5-small, Kaggle)
│   └── requirements_train.txt           # Eğitim bağımlılıkları
│
├── model/
│   ├── story-to-music-analyzer/        # Model 1 ağırlıkları (Kaggle'dan indirilecek)
│   │   ├── model.safetensors
│   │   └── config.json
│   └── story-to-music-lyricist/        # Model 2 ağırlıkları (Kaggle'dan indirilecek)
│       ├── model.safetensors
│       └── config.json
│
├── server/
│   ├── main.py                  # MCP sunucu giriş noktası
│   ├── tools.py                 # Tool tanımları
│   ├── inference.py             # Model çıkarım mantığı
│   ├── groq_client.py           # Groq API entegrasyonu + injection savunması
│   ├── content_guard.py         # İçerik moderasyonu (girdi + çıktı)
│   ├── copyright_guard.py       # N-gram benzerlik kontrolü
│   └── requirements.txt         # Sunucu bağımlılıkları
│
└── docker/
    ├── Dockerfile
    └── .dockerignore
```

---

## Geliştirme Sırası

1. **[x] Proje planı ve CLAUDE.md** — Tamamlandı
2. **[x] `collect_lyrics.py`** — 854 şarkı sözü toplandı (Genius API)
3. **[x] `clean_lyrics.py`** — 855 temiz kayıt
4. **[x] `distill_data.py`** — Ollama (qwen2.5:7b) ile 854 JSONL kaydı üretildi (`data/training/dataset.jsonl`)
5. **[x] `split_dataset.py`** — Dataset metadata + lyrics olarak ikiye bölündü
6. **[ ] Model 1 (Analizci) eğitimi** — `story_to_music_train.ipynb` (mt5-small, Kaggle T4) — henüz çalıştırılmadı
7. **[ ] Model 2 (Söz Yazarı) eğitimi** — `story_to_music_lyrics_train.ipynb` — eğitim tamamlandı ama çıktı sorunlu (MAX_TARGET_LEN fix bekleniyor)
8. **[ ] `content_guard.py`** — İçerik moderasyon modülü
9. **[ ] `copyright_guard.py`** — N-gram benzerlik kontrol modülü
10. **[ ] `server/main.py`** — MCP sunucu + tool tanımları
11. **[ ] `groq_client.py`** — Prompt-only mod entegrasyonu + injection savunması
12. **[ ] Dockerfile** — Konteynerizasyon (~5-6 GB imaj, ham veri dahil edilmez)
13. **[ ] Docker Hub push + test**
14. **[ ] README + kurulum kılavuzu**

---

## Önemli Kararlar ve Gerekçeler

**Neden flan-t5-small değil mt5-small?**
flan-t5-small Türkçe eğitim verisi görmemiş — inference'da çöp çıktı üretiyor.
mt5-small 101 dil üzerinde pre-train edilmiş, Türkçe token'larını tanıyor.
Bedeli: 556M parametre, ~2.1 GB model, CPU'da 10-30 saniye inference.

**Neden tek model değil iki model (Analizci + Söz Yazarı)?**
Tek modelin hem stil analizi hem şarkı sözü üretmesi çok farklı görevler — seq2seq
modeli ikisini aynı anda öğrenmekte zorlanıyor. Analiz görevi (JSON çıktı, kısa) ile
yaratıcı metin üretimi (uzun, yapısal) ayrı modellerde daha iyi sonuç veriyor.

**Neden RoBERTa/DistilBERT değil T5?**
RoBERTa ve DistilBERT encoder-only modellerdir; metin sınıflandırır, metin üretemez.
T5 seq2seq mimarisiyle hem anlama hem üretme yapabilir — bu proje için zorunlu.

**Neden FL Studio değil Suno/Udio?**
FL Studio'nun dışarıdan prompt kabul eden bir API'si yok. Suno/Udio metin bazlı
çalıştığı için MCP çıktısı doğrudan kullanılabilir.

**Neden Docker ile serverless?**
Bulut sunucusu maliyeti sıfırlanır. Kullanıcı tek komutla entegre eder.
Bağımlılık sorunu olmaz — tüm ortam konteyner içinde.

**Neden Groq + local model hibrit?**
Local T5 modeli hız ve gizlilik sağlar, her zaman çalışır.
Groq prompt refinement için kullanılır — sadece "prompt_only" modda ve
isteğe bağlı. Internet yoksa local model tek başına yeterli.

**CPU performansı hakkında gerçekçi beklenti:**
mt5-small generation CPU'da **10-30 saniye** sürer. flan-t5-small'dan ~4× yavaş çünkü
556M parametre ve 250K token vocabulary. MCP aracı bağlamında kabul edilebilir ama
kullanıcıya bekleme süresi gösterilmeli.

---

## Ortam Değişkenleri

```env
GENIUS_ACCESS_TOKEN=...       # Veri toplama aşaması için
GROQ_API_KEY=...              # Prompt refinement için (opsiyonel)
MODEL_PATH=./model/story-to-music-t5
LOG_LEVEL=INFO
```

---

## Notlar

- Şarkı sözleri tamamen **Türkçe** olacak, başka dil desteklenmiyor (v1)
- Groq API ücretsiz katmanı: 14.400 req/gün, 30 req/dakika — dataset üretimi için yeterli
- Eğitim sırasında her epoch'ta checkpoint kaydedilmeli (Colab kesintileri için)
- Docker imaj boyutu hedefi: **~5-6 GB** — mt5-small iki model gerektiriyor, int8 quantization ile ~4 GB'a düşürülebilir
