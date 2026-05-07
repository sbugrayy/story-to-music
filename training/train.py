"""
flan-t5-small fine-tuning — Story-to-Music
Çalışma ortamı: Google Colab / Kaggle (T4 GPU, 16 GB VRAM)

Colab kurulumu:
    1. Sol panel > Files > dataset.jsonl yükle  (ya da Drive'a mount et)
    2. !pip install -r requirements_train.txt
    3. !python train.py
"""

import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

# ── Konfigürasyon ──────────────────────────────────────────────────────────────

BASE_MODEL      = "google/flan-t5-small"
DATASET_PATH    = os.getenv("DATASET_PATH", "dataset.jsonl")
OUTPUT_DIR      = os.getenv("OUTPUT_DIR",   "story-to-music-t5")
CHECKPOINT_DIR  = os.getenv("CHECKPOINT_DIR", "checkpoints")

MAX_INPUT_LEN   = 512
MAX_TARGET_LEN  = 512
TRAIN_RATIO     = 0.90

# Hiperparametreler
EPOCHS          = 15
BATCH_SIZE      = 8       # T4 için güvenli; OOM olursa 4'e düşür
GRAD_ACCUM      = 2       # efektif batch = 16
LR              = 5e-4
WARMUP_RATIO    = 0.10
WEIGHT_DECAY    = 0.01
SAVE_TOTAL      = 3       # en iyi 3 checkpoint tutulur
PATIENCE        = 3       # eval_loss artmazsa erken dur

# ── Veri ──────────────────────────────────────────────────────────────────────

def load_dataset(path: str) -> tuple[Dataset, Dataset]:
    records = []
    missing = []

    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if "input_text" not in d or "target_text" not in d:
                    missing.append(i)
                    continue
                records.append({
                    "input_text":  d["input_text"],
                    "target_text": d["target_text"],
                })
            except json.JSONDecodeError:
                missing.append(i)

    if missing:
        print(f"[!] {len(missing)} satır atlandı (eksik alan veya bozuk JSON): {missing[:5]}")

    if not records:
        sys.exit("[HATA] Geçerli kayıt bulunamadı. DATASET_PATH'i kontrol et.")

    ds = Dataset.from_list(records)
    split = ds.train_test_split(test_size=1 - TRAIN_RATIO, seed=42)
    print(f"[+] Veri: {len(split['train'])} eğitim / {len(split['test'])} doğrulama")
    return split["train"], split["test"]


# ── Tokenizasyon ──────────────────────────────────────────────────────────────

def make_tokenize_fn(tokenizer):
    def tokenize(batch):
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=MAX_INPUT_LEN,
            truncation=True,
            padding=False,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["target_text"],
                max_length=MAX_TARGET_LEN,
                truncation=True,
                padding=False,
            )
        # -100 → loss hesabında padding token'ları yok say
        model_inputs["labels"] = [
            [(t if t != tokenizer.pad_token_id else -100) for t in ids]
            for ids in labels["input_ids"]
        ]
        return model_inputs
    return tokenize


# ── Ana Eğitim ────────────────────────────────────────────────────────────────

def main():
    # GPU kontrol
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Cihaz: {device}")
    if device == "cpu":
        print("[!] UYARI: CPU eğitimi çok yavaş olacak. Colab/Kaggle GPU kullan.")

    # Veri yükle
    train_ds, eval_ds = load_dataset(DATASET_PATH)

    # Model ve tokenizer
    print(f"[+] Model yükleniyor: {BASE_MODEL}")
    tokenizer = T5TokenizerFast.from_pretrained(BASE_MODEL)
    model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL)
    print(f"[+] Parametre sayısı: {model.num_parameters():,}")

    # Tokenize
    tokenize_fn = make_tokenize_fn(tokenizer)
    train_tokenized = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
    eval_tokenized  = eval_ds.map(tokenize_fn,  batched=True, remove_columns=eval_ds.column_names)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    # Eğitim argümanları
    total_steps = (len(train_tokenized) // (BATCH_SIZE * GRAD_ACCUM)) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    args = Seq2SeqTrainingArguments(
        output_dir=CHECKPOINT_DIR,

        # Döngü
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,

        # Optimizer
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",

        # Değerlendirme ve kaydetme
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=SAVE_TOTAL,

        # Hız / bellek
        fp16=(device == "cuda"),
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LEN,

        # Log
        logging_steps=20,
        report_to="none",

        # Seed
        seed=42,
        data_seed=42,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    )

    # Önceki checkpoint varsa devam et (Colab kesintisi sonrası)
    last_checkpoint = _find_last_checkpoint(CHECKPOINT_DIR)
    if last_checkpoint:
        print(f"[+] Checkpoint bulundu, devam ediliyor: {last_checkpoint}")
    else:
        print("[+] Sıfırdan başlanıyor.")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    # En iyi modeli kaydet
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n[✓] Model kaydedildi: {OUTPUT_DIR}/")

    # Boyut raporu
    safetensors_path = Path(OUTPUT_DIR) / "model.safetensors"
    if safetensors_path.exists():
        size_mb = safetensors_path.stat().st_size / (1024 ** 2)
        print(f"[+] Model boyutu: {size_mb:.1f} MB")

    # Hızlı smoke test
    _smoke_test(model, tokenizer, eval_ds[0]["input_text"])


def _find_last_checkpoint(checkpoint_dir: str) -> str | None:
    p = Path(checkpoint_dir)
    if not p.exists():
        return None
    checkpoints = sorted(p.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[-1]))
    return str(checkpoints[-1]) if checkpoints else None


def _smoke_test(model, tokenizer, sample_input: str):
    print("\n── Smoke Test ──")
    model.eval()
    device = next(model.parameters()).device
    inputs = tokenizer(
        sample_input[:512],
        return_tensors="pt",
        max_length=MAX_INPUT_LEN,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            early_stopping=True,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Çıktı (ilk 300 karakter):")
    print(decoded[:300])
    print("────────────────")


if __name__ == "__main__":
    main()
