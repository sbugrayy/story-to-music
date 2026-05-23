# HTTP MCP Transport — Deployment Plan

> v1.1 hedefi: Mevcut stdio MCP'ye ek olarak HTTP/SSE transport ekle, HF Space üzerinden remote MCP server sun. Docker yolu (lokal/stdio) korunur. Bu doküman sıfırdan execute edilebilir; her şey burada yazılı, ek context gerektirmez.

## Hedef

İki transport, iki kullanıcı senaryosu:

| Kullanıcı | Transport | Erişim |
|---|---|---|
| Local kullanıcı | stdio (mevcut) | `docker run --rm -i sbugrayy/story-to-music-mcp:latest` |
| Claude Desktop/Code Pro kullanıcısı | HTTP/SSE (yeni) | `https://bugrayildirim-story-to-music.hf.space/sse` |
| Web demo | Gradio UI (mevcut) | `https://huggingface.co/spaces/bugrayildirim/story-to-music` |

Üç dağıtım da tek HF Space'te çalışır — model'ler bir kez yüklenir, hem Gradio hem MCP onları paylaşır.

---

## Mimari Karar: Gradio + FastAPI Mount

Gradio aslında FastAPI üstüne kurulu. `mcp.server.sse` Starlette/FastAPI uyumlu. Tek process:

```
HF Space (tek container)
├── gradio_blocks  →  /          (UI)
├── /sse           →  MCP SSE GET (server→client)
└── /messages/     →  MCP POST   (client→server)
```

`gr.mount_gradio_app(fastapi_app, demo, path="/")` ile mount edilir. Model cache (`_models` dict) iki endpoint arasında paylaşılır — RAM verimli.

**Alternatif (reddedildi):** Ayrı HF Space açmak. Model'ler 2× yüklenir, 16 GB RAM yetmez.

---

## Dosya Değişiklikleri

### 1. `huggingface_space/app.py` — MCP HTTP endpoint ekle

Mevcut Gradio + model loading kodu **olduğu gibi kalır**. Sona MCP server kurulumu eklenir:

```python
# ── MCP Server (HTTP/SSE Transport) ───────────────────────────────────────────

from mcp.server import Server as MCPServer
from mcp.server.sse import SseServerTransport
from mcp.types import TextContent, Tool
from starlette.applications import Starlette
from starlette.routing import Route, Mount

mcp_app = MCPServer("story-to-music")


@mcp_app.list_tools()
async def _list_tools() -> list[Tool]:
    return [
        Tool(
            name="generate_music_prompt",
            description="Türkçe metin → Suno/Udio için tam müzik promptu...",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "mode": {"type": "string", "enum": ["custom", "prompt_only"], "default": "custom"},
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="analyze_emotion",
            description="Sadece duygu/stil analizi.",
            inputSchema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        ),
        Tool(
            name="generate_lyrics_only",
            description="Metadata verildiğinde sadece şarkı sözü üret.",
            inputSchema={
                "type": "object",
                "properties": {
                    "emotion": {"type": "string"},
                    "energy": {"type": "integer", "default": 5},
                    "bpm": {"type": "integer", "default": 90},
                    "key": {"type": "string", "default": "A minor"},
                    "instruments": {"type": "array", "items": {"type": "string"}, "default": []},
                    "vocal_style": {"type": "string", "default": ""},
                },
                "required": ["emotion"],
            },
        ),
    ]


@mcp_app.call_tool()
async def _call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "generate_music_prompt":
            text = arguments.get("text", "")
            mode = arguments.get("mode", "custom")
            style, lyrics, titles, params_json = run_pipeline(text, mode)
            if style.startswith("❌"):
                result = {"error": style}
            elif mode == "prompt_only":
                result = {"suno_prompt": style}
            else:
                result = {
                    "style_prompt": style,
                    "structured_lyrics": lyrics,
                    "title_suggestions": [t.strip("• ").strip() for t in titles.split("\n") if t.strip()],
                    "parameters": json.loads(params_json) if params_json else {},
                }
        elif name == "analyze_emotion":
            text = arguments.get("text", "")
            result = analyze(text)
        elif name == "generate_lyrics_only":
            meta = {
                "emotion": arguments["emotion"],
                "energy": arguments.get("energy", 5),
                "bpm": arguments.get("bpm", 90),
                "key": arguments.get("key", "A minor"),
                "instruments": arguments.get("instruments", []),
                "vocal_style": arguments.get("vocal_style", ""),
                "suno_style_prompt": "",
            }
            result = {"structured_lyrics": generate_lyrics(meta), "parameters": meta}
        else:
            result = {"error": f"Bilinmeyen tool: {name}"}
    except Exception as e:
        result = {"error": f"İç hata: {type(e).__name__}: {e}"}

    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]


# ── Starlette + Gradio Mount ──────────────────────────────────────────────────

sse_transport = SseServerTransport("/messages/")


async def handle_sse(request):
    async with sse_transport.connect_sse(
        request.scope, request.receive, request._send,
    ) as (read_stream, write_stream):
        await mcp_app.run(read_stream, write_stream, mcp_app.create_initialization_options())


starlette_app = Starlette(
    routes=[
        Route("/sse", endpoint=handle_sse),
        Mount("/messages/", app=sse_transport.handle_post_message),
    ],
)

# Gradio'yu Starlette app'e mount et — Gradio UI "/" altında, MCP "/sse" ve "/messages/" altında
app = gr.mount_gradio_app(starlette_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
```

**Önemli:** `demo.queue(...).launch()` çağrısını **kaldır**, Gradio mount edildiği için ayrıca launch'a gerek yok. Uvicorn `app`'i çalıştırır.

### 2. `huggingface_space/requirements.txt` — dependencies ekle

Mevcut içeriğe ekle:
```
mcp>=1.0.0
starlette>=0.37.0
uvicorn[standard]>=0.30.0
```

`fastapi` ZATEN Gradio bağımlılığı, ayrıca eklemeye gerek yok. `starlette` daha düşük seviyeli ve daha az çakışma yaratır.

### 3. `huggingface_space/README.md` — sdk değiştir

YAML frontmatter'da:
```yaml
sdk: docker  # gradio'dan docker'a değiştir
```

Ve aynı dosyaya `Dockerfile` ekle (HF Space root'a):

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

**Neden Docker SDK?** Gradio SDK runtime'ı `gradio.cli`'yi çalıştırır, custom uvicorn launch yapamazsın. Docker SDK ise tam kontrol verir. Build süresi biraz uzar (~3 dk) ama tek seferlik.

### 4. README.md (proje root) — kullanım örnekleri ekle

Hızlı Başlangıç bölümüne 3. seçenek olarak:

```markdown
### Remote (zero-install, Claude Desktop / Code)

```json
{
  "mcpServers": {
    "story-to-music": {
      "url": "https://bugrayildirim-story-to-music.hf.space/sse",
      "transport": "sse"
    }
  }
}
```

İlk istek 60-90 saniye (HF Space cold start + model yükleme). Sonraki istekler hızlı.
HF Space free tier 48 saat inaktiviteden sonra uyur, ilk istekte uyanır.
```

---

## Adım Adım Implementation

### Adım 1: Local'de test edilebilir hale getir
```powershell
cd C:\Users\bugra\Documents\Claude\Projects\Story-to-Music\huggingface_space
pip install mcp starlette "uvicorn[standard]"
python app.py
```
Tarayıcıdan `http://localhost:7860/` → Gradio UI görmeli.
`http://localhost:7860/sse` → text/event-stream response vermeli (curl ile test).

### Adım 2: MCP client ile local test
Local Claude Desktop config:
```json
{
  "mcpServers": {
    "story-to-music-local-http": {
      "url": "http://localhost:7860/sse",
      "transport": "sse"
    }
  }
}
```
Claude Desktop restart. Tool listesinde görmeli, çağrı denenebilir.

### Adım 3: HF Space'e deploy
```powershell
Remove-Item Env:SSL_CERT_FILE -ErrorAction SilentlyContinue
hf upload bugrayildirim/story-to-music huggingface_space --repo-type space
```
Space rebuild başlar. Dockerfile build ~3-5 dakika.

### Adım 4: Remote test
Build "Running" geçince:
```powershell
curl -N https://bugrayildirim-story-to-music.hf.space/sse
```
SSE event stream görmeli.

Claude Desktop config'i remote URL ile güncelle, restart, tool çağır.

### Adım 5: Gradio UI hâlâ çalışıyor mu kontrol et
`https://huggingface.co/spaces/bugrayildirim/story-to-music` — UI normal görünmeli, "Üret" butonu çalışmalı.

---

## Test Senaryoları

| Test | Beklenen |
|---|---|
| `curl https://.../sse` | `event: endpoint\ndata: /messages/?session_id=...` |
| Claude Desktop tool list | `story-to-music` görünür, 3 tool listelenir |
| `generate_music_prompt` çağrısı | 60-120s sonra JSON yanıt (style + lyrics + titles) |
| Gradio UI hâlâ erişilebilir mi | `https://...hf.space/` Gradio yüklenir |
| Cold start (48h inaktivite sonrası) | İlk istek 90-120s, sonra hızlı |

---

## Olası Sorunlar ve Çözümleri

### A. Build hatası: "ModuleNotFoundError: mcp"
`requirements.txt`'e `mcp>=1.0.0` eklenmemiş. Düzelt ve yeniden upload.

### B. Cold start çok uzun (>3 dakika)
Model'ler HF Hub'dan iniyor (2.5 GB). Çözüm: Modelleri build sırasında preload et — Dockerfile'a ekle:
```dockerfile
RUN python -c "from transformers import AutoModelForSeq2SeqLM, AutoTokenizer; \
  AutoTokenizer.from_pretrained('bugrayildirim/story-to-music-analyzer'); \
  AutoModelForSeq2SeqLM.from_pretrained('bugrayildirim/story-to-music-analyzer'); \
  AutoTokenizer.from_pretrained('bugrayildirim/story-to-music-lyricist'); \
  AutoModelForSeq2SeqLM.from_pretrained('bugrayildirim/story-to-music-lyricist')"
```
Build süresi ~10 dakikaya çıkar ama runtime cold start ~30s'ye iner.

### C. MCP client bağlanamıyor: "Connection refused"
HF Space "Sleeping" durumunda olabilir. Web UI'a bir GET at, uyanır. Veya HF Space'i "Always-on" yap (ücretli).

### D. Claude Desktop tool çağırıyor ama timeout
Default timeout düşük olabilir. Config'e ekle:
```json
{
  "mcpServers": {
    "story-to-music": {
      "url": "https://...",
      "transport": "sse",
      "timeout": 180000
    }
  }
}
```

### E. Gradio mount sonrası UI bozuk
`gr.mount_gradio_app` parametre sırası kritik:
```python
app = gr.mount_gradio_app(starlette_app, demo, path="/")
```
İlk arg Starlette app, ikinci Gradio Blocks, üçüncü path. Karıştırma.

### F. CORS hatası (browser MCP client kullanıyorsa)
Starlette'e CORS middleware ekle:
```python
from starlette.middleware.cors import CORSMiddleware
starlette_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])
```

---

## Yapma Listesi

- Sampling parametrelerine (_LYRICIST_GEN_KWARGS) dokunma — özenle ayarlandı
- `_load()` fonksiyonunu değiştirme — `low_cpu_mem_usage` zaten doğru ayarlı (kullanılmıyor)
- Gradio'yu kaldırma — UI dağıtım kanalı olarak önemli
- Birden fazla HF Space açma — tek Space'te iki transport çalışıyor, RAM verimli
- requirements.txt'te transformers pin'ini bozma (`transformers==4.46.3` çalışıyor)

---

## Toplam İş Saati Tahmini

- Kod yazımı: 30 dk (yukarıdaki snippet'leri app.py'a yapıştır)
- Local test: 20 dk
- HF deploy + rebuild: 10 dk
- Claude Desktop entegrasyon + test: 15 dk
- README güncellemesi: 10 dk

**Toplam: ~1.5 saat.** Çoğunlukla bekleme (build + cold start).

---

## Commit Stratejisi

İki commit'le:
1. `Add HTTP/SSE MCP transport to HF Space` — kod değişikliği
2. `Document remote MCP usage in README` — kullanıcı dokümantasyonu

Bu sayede bir şey patlarsa hangi commit'e revert edileceği net.

---

## Bittiğinde

README'ye:
- Üçüncü kullanım modu (remote) eklendi
- Yeni badge: `![Remote MCP](https://img.shields.io/badge/MCP-remote-green)`

CLAUDE.md'ye:
- Geliştirme sırası 18. madde: `[x] HTTP MCP transport — HF Space remote endpoint`

Lansman duyurusunda öne çıkar:
> "Zero-install option: Claude Desktop/Code Pro kullanıcıları sadece config'e URL eklesin, Docker gerekmiyor."
