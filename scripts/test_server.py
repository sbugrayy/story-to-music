"""
Server pipeline test — MCP wrap'ini atlayarak doğrudan tools.py'ı çağırır.

Kullanım:
    python scripts/test_server.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import tools


SAMPLE_TEXT = """
Bir zamanlar büyük bir aşk yaşamıştım. Sonra o aşk gitti, geride sadece anılar kaldı.
Şimdi her gece pencereden dışarı bakıyorum, yağmurun sesi hatırlatıyor onu bana.
Geri dönmesini bekliyorum ama biliyorum ki dönmeyecek. Yine de umut etmekten vazgeçemiyorum.
"""


def test_analyze():
    print("=" * 60)
    print("TEST: analyze_emotion")
    print("=" * 60)
    result = tools.analyze_emotion(SAMPLE_TEXT)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def test_full():
    print("\n" + "=" * 60)
    print("TEST: generate_music_prompt (mode=custom)")
    print("=" * 60)
    result = tools.generate_music_prompt(SAMPLE_TEXT, mode="custom")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def test_lyrics_only():
    print("\n" + "=" * 60)
    print("TEST: generate_lyrics_only")
    print("=" * 60)
    result = tools.generate_lyrics_only(
        emotion="özlem",
        energy=3,
        bpm=65,
        key="D minor",
        instruments=["bağlama", "ney"],
        vocal_style="erkek, ağır, dokunaklı",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    test_analyze()
    test_lyrics_only()
    test_full()
