"""
Story-to-Music MCP Server giriş noktası.

Stdio transport ile çalışır; Claude Desktop / Cursor config'inde
`docker run --rm -i story-to-music-mcp:latest` ile entegre edilir.

Çalıştırma:
    python -m server.main
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from . import tools

# ── Loglama ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,  # stdout MCP protokolüne ayrılmış
)
log = logging.getLogger(__name__)

# ── MCP Server ────────────────────────────────────────────────────────────────

app = Server("story-to-music")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="generate_music_prompt",
            description=(
                "Türkçe metin (hikaye/konsept/şarkı sözü) → Suno/Udio için tam müzik "
                "promptu: stil parametreleri, [Verse]/[Chorus] etiketli Türkçe şarkı sözü "
                "ve başlık önerileri. mode='custom' tam çıktı, mode='prompt_only' tek satır prompt."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Analiz edilecek Türkçe metin (10-800 kelime).",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["custom", "prompt_only"],
                        "default": "custom",
                        "description": "'custom' = tam çıktı (stil+sözler), 'prompt_only' = tek satır.",
                    },
                    "language": {
                        "type": "string",
                        "enum": ["tr"],
                        "default": "tr",
                    },
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="analyze_emotion",
            description=(
                "Sadece duygu/stil analizi yap. Şarkı sözü üretmez, hafif kullanım için. "
                "Türkçe metin → {emotion, energy, bpm, key, instruments, vocal_style, suno_style_prompt}."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="generate_lyrics_only",
            description=(
                "Stil parametreleri verildiğinde sadece Türkçe şarkı sözü üret. "
                "Metin analizi atlanır, doğrudan lyricist modeli çalışır."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "emotion": {"type": "string", "description": "Örn: hüzün, aşk, öfke, özlem"},
                    "energy": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                    "bpm": {"type": "integer", "default": 90},
                    "key": {"type": "string", "default": "A minor"},
                    "instruments": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "vocal_style": {"type": "string", "default": ""},
                },
                "required": ["emotion"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    log.info(f"Tool çağrısı: {name}")
    try:
        if name == "generate_music_prompt":
            result = tools.generate_music_prompt(**arguments)
        elif name == "analyze_emotion":
            result = tools.analyze_emotion(**arguments)
        elif name == "generate_lyrics_only":
            result = tools.generate_lyrics_only(**arguments)
        else:
            result = {"error": f"Bilinmeyen tool: {name}"}
    except Exception as e:
        log.exception(f"Tool execution hatası: {name}")
        result = {"error": f"İç hata: {e}"}

    # MCP TextContent — JSON pretty print
    import json
    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    log.info("Story-to-Music MCP server başlatılıyor")
    log.info("Modelleri ilk tool çağrısında lazy-load eder")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
