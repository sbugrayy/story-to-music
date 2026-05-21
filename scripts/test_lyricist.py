"""
Model 2 (Söz Yazarı) local smoke test.

Kaggle'a girmeden, indirilen model.safetensors ile generation parametrelerini
test eder. CPU'da çalışır (~30-60s/üretim).

Kullanım:
    python scripts/test_lyricist.py
"""

import random
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_DIR = Path("data/story-to-music-lyricist")
MAX_INPUT_LEN = 64

TURKISH_CHARS = set("çğıöşüÇĞİÖŞÜ")

GEN_KWARGS = dict(
    max_new_tokens=350,
    do_sample=True,                  # beam search yerine sampling
    num_beams=1,
    temperature=0.9,
    top_p=0.9,
    top_k=50,
    no_repeat_ngram_size=3,
    repetition_penalty=1.3,
)

TEST_INPUTS = [
    "Türkçe şarkı sözü yaz: duygu=hüzün, enerji=4, tempo=72 BPM, ton=A minor, "
    "enstrümanlar=ney, keman, piyano, vokal=erkek, kısık, dramatik",

    "Türkçe şarkı sözü yaz: duygu=aşk, enerji=7, tempo=110 BPM, ton=C major, "
    "enstrümanlar=gitar, davul, bas, vokal=kadın, lirik, yumuşak",

    "Türkçe şarkı sözü yaz: duygu=öfke, enerji=9, tempo=140 BPM, ton=E minor, "
    "enstrümanlar=elektro gitar, davul, bas, vokal=erkek, sert, isyankar",

    "Türkçe şarkı sözü yaz: duygu=özlem, enerji=3, tempo=65 BPM, ton=D minor, "
    "enstrümanlar=bağlama, ney, vokal=erkek, ağır, dokunaklı",
]


def turkce_skor(text):
    if not text:
        return 0.0
    tr = sum(1 for c in text if c in TURKISH_CHARS)
    alpha = sum(1 for c in text if c.isalpha())
    return tr / alpha if alpha > 0 else 0.0


def tekrar_orani(text):
    words = [w.lower() for w in text.split() if w.isalpha()]
    if len(words) < 10:
        return 0.0
    counts = Counter(words)
    return counts.most_common(1)[0][1] / len(words)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cihaz: {device}")
    print(f"Model: {MODEL_DIR}")

    print("\nModel ve tokenizer yükleniyor...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(device)
    model.eval()
    print(f"Parametre: {model.num_parameters():,}")

    nan_count = sum(p.isnan().any().item() for p in model.parameters())
    print(f"NaN tensör: {nan_count} (0 olmalı)")

    # Sampling deterministic değil — her input için 2 çıktı üret, varyans gör
    torch.manual_seed(42)

    for i, inp in enumerate(TEST_INPUTS, 1):
        print(f"\n{'=' * 60}")
        print(f"TEST {i}/{len(TEST_INPUTS)}")
        print(f"{'=' * 60}")
        print(f"GIRDI: {inp}")

        inputs = tokenizer(
            inp, return_tensors="pt",
            max_length=MAX_INPUT_LEN, truncation=True
        ).to(device)

        # Aynı girdi için 2 farklı sampling — varyans testi
        for run in (1, 2):
            with torch.no_grad():
                out = model.generate(**inputs, **GEN_KWARGS)
            decoded = tokenizer.decode(out[0], skip_special_tokens=True)
            print(f"\n--- Sample {run} ---")
            print(decoded[:400])

        # Metrikler son sample için
        print(f"\nMetrikler (son):")
        print(f"  Kelime sayısı     : {len(decoded.split())}")
        print(f"  Türkçe karakter   : {turkce_skor(decoded):.1%}")
        print(f"  En sık kelime oranı: {tekrar_orani(decoded):.1%}")
if __name__ == "__main__":
    main()
