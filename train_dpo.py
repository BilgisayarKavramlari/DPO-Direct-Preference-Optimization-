#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--train_jsonl", type=str, default="data/sample_tiny.jsonl")
    ap.add_argument("--eval_jsonl", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default="models/qwen25-0_5b-dpo")
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--per_device_train_batch_size", type=int, default=4)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--eval_steps", type=int, default=100)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else (torch.float16 if torch.cuda.is_available() else torch.float32)
    device_map = "auto" if torch.cuda.is_available() else None

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
        print(f"[WARN] 4-bit yükleme başarısız: {e}\nTam hassasiyetli yükleniyor.")
        model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map=device_map, torch_dtype=dtype)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    train_dataset = load_dataset("json", data_files=args.train_jsonl, split="train")
    eval_dataset = load_dataset("json", data_files=args.eval_jsonl, split="train") if (args.eval_jsonl and os.path.exists(args.eval_jsonl)) else None

    targs = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        save_steps=args.save_steps,
        logging_steps=10,
        bf16=(dtype==torch.bfloat16),
        fp16=(dtype==torch.float16),
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        report_to="none",
    )

    dpo = DPOTrainer(
        model=model,
        ref_model=None,
        args=targs,
        beta=0.1,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=1024,
        max_prompt_length=512,
        max_target_length=512,
    )

    dpo.train()
    dpo.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[OK] Kaydedildi: {args.output_dir}")

if __name__ == "__main__":
    main()
