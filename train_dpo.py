#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, argparse, random
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import DPOConfig, DPOTrainer


def set_global_seed(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--train_jsonl", type=str, default="data/sample_tiny.jsonl")
    ap.add_argument("--eval_jsonl", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default="models/qwen25-0_5b-dpo")

    # Eğitim hiperparametreleri
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--per_device_train_batch_size", type=int, default=2)   # CPU için küçültüldü
    ap.add_argument("--per_device_eval_batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)   # efektif batch’i korur
    ap.add_argument("--max_steps", type=int, default=200)                   # duman testi için kısa
    ap.add_argument("--eval_steps", type=int, default=50)
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)

    # Sayısal tip/cihaz
    ap.add_argument("--bf16", action="store_true")

    # LoRA
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # Çeşitli
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_global_seed(args.seed)

    # ---------------- Tokenizer ----------------
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Uzun sekansları kes: TRL 0.21 internal processing_class bu ayarları kullanır
    tokenizer.model_max_length = 512
    tokenizer.truncation = True
    tokenizer.truncation_side = "left"

    # ---------------- Model (4-bit mümkünse) ----------------
    dtype = (
        torch.bfloat16 if (args.bf16 and torch.cuda.is_available())
        else (torch.float16 if torch.cuda.is_available() else torch.float32)
    )
    device_map = "auto" if torch.cuda.is_available() else None

    bnb_config = None
    try:
        if torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map=device_map,
            torch_dtype=dtype,
            quantization_config=bnb_config,
        )
    except Exception as e:
        print(f"[WARN] 4-bit yükleme başarısız: {e}\nTam hassasiyet (FP) ile yükleniyor.")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, device_map=device_map, torch_dtype=dtype
        )

    # Bellek: gradient checkpointing (CPU/GPU fark etmeksizin RAM tüketimini düşürür)
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    # ---------------- LoRA (PEFT) ----------------
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # ---------------- Dataset ----------------
    train_dataset = load_dataset("json", data_files=args.train_jsonl, split="train")
    eval_dataset = None
    if args.eval_jsonl and os.path.exists(args.eval_jsonl):
        eval_dataset = load_dataset("json", data_files=args.eval_jsonl, split="train")

    # ---------------- DPOConfig (TRL 0.21) ----------------
    cfg = DPOConfig(
        output_dir=args.output_dir,

        # Eğitim
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,

        # Kayıt / Log / Değerlendirme
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=1,
        save_safetensors=True,
        logging_steps=10,
        eval_strategy=("steps" if eval_dataset is not None else "no"),
        eval_steps=(args.eval_steps if eval_dataset is not None else 0),

        # Sayısal tipler
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),

        # LR scheduler & ısınma
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,

        # Optimizer
        optim=("paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch"),

        # Dataloader & RAM
        dataloader_pin_memory=False,
        dataloader_num_workers=0,

        # DPO özgü
        loss_type="sigmoid",
        beta=0.1,
    )

    # ---------------- DPOTrainer ----------------
    # TRL 0.21: tokenizer yerine processing_class kullanılır; max_* argümanları yok.
    dpo = DPOTrainer(
        model=model,
        args=cfg,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        ref_model=None,
    )

    # ---------------- Train & Save ----------------
    dpo.train()
    dpo.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[OK] Model ve tokenizer kaydedildi: {args.output_dir}")


if __name__ == "__main__":
    main()
