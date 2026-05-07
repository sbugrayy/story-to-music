"""
Eğitilmiş modeli değerlendirir.
Kullanım:
    python evaluate.py --model story-to-music-t5 --dataset dataset.jsonl --samples 50
"""

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast

REQUIRED_KEYS = {"emotion", "energy", "bpm", "key", "instruments",
                 "vocal_style", "suno_style_prompt", "structured_lyrics"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   default="story-to-music-t5")
    p.add_argument("--dataset", default="dataset.jsonl")
    p.add_argument("--samples", type=int, default=50)
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


def load_samples(path: str, n: int, seed: int) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    random.seed(seed)
    return random.sample(records, min(n, len(records)))


def generate(model, tokenizer, text: str, device) -> str:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    ).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_length=512,
            num_beams=4,
            early_stopping=True,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def evaluate_sample(generated: str, reference: str) -> dict:
    result = {"valid_json": False, "key_coverage": 0.0, "field_match": {}}

    try:
        gen = json.loads(generated)
        result["valid_json"] = True
    except json.JSONDecodeError:
        return result

    try:
        ref = json.loads(reference)
    except json.JSONDecodeError:
        return result

    present = REQUIRED_KEYS & set(gen.keys())
    result["key_coverage"] = len(present) / len(REQUIRED_KEYS)

    # Skaler alanlar için tam eşleşme
    for field in ("emotion", "energy", "bpm", "key"):
        result["field_match"][field] = gen.get(field) == ref.get(field)

    return result


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Cihaz: {device}")

    print(f"[+] Model yükleniyor: {args.model}")
    tokenizer = T5TokenizerFast.from_pretrained(args.model)
    model = T5ForConditionalGeneration.from_pretrained(args.model).to(device)
    model.eval()

    samples = load_samples(args.dataset, args.samples, args.seed)
    print(f"[+] {len(samples)} örnek değerlendiriliyor...")

    stats = {
        "valid_json": 0,
        "key_coverage": [],
        "field_exact": {f: 0 for f in ("emotion", "energy", "bpm", "key")},
    }

    for i, sample in enumerate(samples, 1):
        generated = generate(model, tokenizer, sample["input_text"], device)
        r = evaluate_sample(generated, sample["target_text"])

        if r["valid_json"]:
            stats["valid_json"] += 1
        stats["key_coverage"].append(r["key_coverage"])
        for field, match in r.get("field_match", {}).items():
            if match:
                stats["field_exact"][field] += 1

        if i <= 3:  # İlk 3 örneği göster
            print(f"\n── Örnek {i} ──")
            print(f"  Üretilen (200 karakter): {generated[:200]}")
            print(f"  Geçerli JSON: {r['valid_json']}  |  Alan kapsamı: {r['key_coverage']:.0%}")

    n = len(samples)
    avg_coverage = sum(stats["key_coverage"]) / n if n else 0

    print("\n══ Sonuçlar ══")
    print(f"  Geçerli JSON oranı  : {stats['valid_json']}/{n} ({stats['valid_json']/n:.0%})")
    print(f"  Ortalama alan kapsamı: {avg_coverage:.0%}")
    print("  Alan tam eşleşme:")
    for field, count in stats["field_exact"].items():
        print(f"    {field:12s}: {count}/{n} ({count/n:.0%})")

    # Kabul kriteri (basit)
    json_rate = stats["valid_json"] / n
    if json_rate >= 0.80 and avg_coverage >= 0.85:
        print("\n[✓] Model sunucu entegrasyonuna hazır.")
    else:
        print(f"\n[!] Model henüz yeterli değil — JSON oranı {json_rate:.0%}, kapsam {avg_coverage:.0%}.")
        print("    Daha fazla epoch veya daha büyük model deneyebilirsin.")


if __name__ == "__main__":
    main()
