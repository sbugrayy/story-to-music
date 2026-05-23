---
title: Story-to-Music
emoji: 🎵
colorFrom: purple
colorTo: red
sdk: gradio
sdk_version: 5.0.0
python_version: "3.12"
app_file: app.py
pinned: true
license: mit
short_description: Türkçe metin → Suno/Udio için müzik promptu ve şarkı sözü
models:
  - bugrayildirim/story-to-music-analyzer
  - bugrayildirim/story-to-music-lyricist
tags:
  - turkish
  - music-generation
  - suno
  - udio
  - mcp
  - lyrics
---

# 🎵 Story-to-Music

Türkçe metni **Suno** ve **Udio** için hazır müzik promptu + yapılandırılmış Türkçe şarkı sözüne çeviren açık kaynaklı sistem.

İki fine-tuned `mT5-small` modeli:
- **Analyzer:** Türkçe metin → `{emotion, energy, bpm, key, instruments, vocal_style}`
- **Lyricist:** Metadata → `[Verse]/[Chorus]/[Bridge]/[Outro]` etiketli Türkçe sözler

## Kullanım

Sol panelden Türkçe bir metin gir, **Üret** butonuna bas. CPU'da ~30-60 saniye.

## Daha Fazlası

- 🔗 [GitHub](https://github.com/sbugrayy/story-to-music) — kod, eğitim notebookları, MCP server
- 🐳 [Docker Hub](https://hub.docker.com/r/sbugrayy/story-to-music-mcp) — `docker run -i sbugrayy/story-to-music-mcp:latest`
- 🤗 Modeller: [analyzer](https://huggingface.co/bugrayildirim/story-to-music-analyzer) · [lyricist](https://huggingface.co/bugrayildirim/story-to-music-lyricist)

## MCP Entegrasyonu

Claude Code, Cursor, veya Claude Desktop'a MCP server olarak ekleyebilirsin — kendi makinende çalışır, harici API yok.

```json
{
  "mcpServers": {
    "story-to-music": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "sbugrayy/story-to-music-mcp:latest"]
    }
  }
}
```

MIT License · [@sbugrayy](https://github.com/sbugrayy)
