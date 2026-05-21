# Story-to-Music MCP Server
# Multi-stage build — final image ~5-6 GB (iki mt5-small model + PyTorch)

FROM python:3.11-slim AS builder

WORKDIR /build

# Build deps (sadece pip için)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY server/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ── Runtime image ─────────────────────────────────────────────────────────────

FROM python:3.11-slim

LABEL maintainer="bugrayildirim"
LABEL description="Story-to-Music MCP server: Türkçe metin → Suno/Udio müzik promptu"

WORKDIR /app

# Builder'dan Python paketlerini kopyala
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

# Server kodu
COPY server/ /app/server/

# Modeller — gerçek build sırasında bu klasörler dolu olmalı
# .dockerignore'da exclude EDİLMEMİŞ olduğundan emin ol
COPY model/story-to-music-analyzer/ /app/model/story-to-music-analyzer/
COPY model/story-to-music-lyricist/ /app/model/story-to-music-lyricist/

# Copyright guard için referans veri
COPY data/training/dataset_lyrics_v2.jsonl /app/data/training/dataset_lyrics_v2.jsonl

# Çevre değişkenleri
ENV ANALYZER_PATH=/app/model/story-to-music-analyzer
ENV LYRICIST_PATH=/app/model/story-to-music-lyricist
ENV LOG_LEVEL=INFO

# MCP stdio transport — interactive
ENTRYPOINT ["python", "-m", "server.main"]
