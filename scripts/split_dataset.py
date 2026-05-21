"""
dataset.jsonl → dataset_metadata.jsonl + dataset_lyrics.jsonl + dataset_lyrics_v2.jsonl

dataset_lyrics_v2.jsonl: orijinal lyrics + lyrics_augmented.jsonl birleştirilmiş hâl

Kullanım:
    python scripts/split_dataset.py
"""

import json
from pathlib import Path

INPUT      = Path("data/training/dataset.jsonl")
AUGMENTED  = Path("data/training/lyrics_augmented.jsonl")
OUTDIR     = Path("data/training")


def fmt_instruments(v):
    if isinstance(v, list):
        return ", ".join(v)
    return str(v)


def main():
    meta_records, lyrics_records = [], []

    with open(INPUT, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)

            # ── Model 1: Analizci ──────────────────────────────────────────
            meta_target = {
                "emotion":          d.get("emotion", ""),
                "energy":           d.get("energy", 5),
                "bpm":              d.get("bpm", 90),
                "key":              d.get("key", "A minor"),
                "instruments":      d.get("instruments", []),
                "vocal_style":      d.get("vocal_style", ""),
                "suno_style_prompt": d.get("suno_style_prompt", ""),
            }
            meta_records.append({
                "input_text":  d["input_text"],
                "target_text": json.dumps(meta_target, ensure_ascii=False),
            })

            # ── Model 2: Söz Yazarı ────────────────────────────────────────
            lyrics = d.get("structured_lyrics", "").strip()
            if not lyrics or len(lyrics.split()) < 20:
                continue

            lyrics_input = (
                f"Türkçe şarkı sözü yaz: "
                f"duygu={d.get('emotion','')}, "
                f"enerji={d.get('energy',5)}, "
                f"tempo={d.get('bpm',90)} BPM, "
                f"ton={d.get('key','A minor')}, "
                f"enstrümanlar={fmt_instruments(d.get('instruments',[]))}, "
                f"vokal={d.get('vocal_style','')}"
            )
            lyrics_records.append({
                "input_text":  lyrics_input,
                "target_text": lyrics,
            })

    OUTDIR.mkdir(parents=True, exist_ok=True)

    meta_path = OUTDIR / "dataset_metadata.jsonl"
    with open(meta_path, "w", encoding="utf-8") as f:
        for r in meta_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lyrics_path = OUTDIR / "dataset_lyrics.jsonl"
    with open(lyrics_path, "w", encoding="utf-8") as f:
        for r in lyrics_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── Birleşik v2: orijinal lyrics + augmented ──────────────────────────────
    v2_records = list(lyrics_records)
    aug_count = 0
    if AUGMENTED.exists():
        with open(AUGMENTED, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if "input_text" not in d or "target_text" not in d:
                    continue
                v2_records.append({
                    "input_text":  d["input_text"],
                    "target_text": d["target_text"],
                })
                aug_count += 1

    v2_path = OUTDIR / "dataset_lyrics_v2.jsonl"
    with open(v2_path, "w", encoding="utf-8") as f:
        for r in v2_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Analizci dataset   : {len(meta_records):>4} kayit -> {meta_path}")
    print(f"Soz yazari dataset : {len(lyrics_records):>4} kayit -> {lyrics_path}")
    print(f"V2 (lyrics+aug)    : {len(v2_records):>4} kayit -> {v2_path}  (augmented: {aug_count})")

    # Örnek göster
    print("\n── Model 1 örnek target ──")
    print(meta_records[0]["target_text"])
    print("\n── Model 2 örnek input ──")
    print(lyrics_records[0]["input_text"])
    print("\n── Model 2 örnek target (ilk 200 karakter) ──")
    print(lyrics_records[0]["target_text"][:200])


if __name__ == "__main__":
    main()
