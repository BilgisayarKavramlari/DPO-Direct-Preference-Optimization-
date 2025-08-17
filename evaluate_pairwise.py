#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, json, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model_tokenizer(base_model: str, adapter_path: str = None, dtype=None):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path or base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype or (torch.float16 if torch.cuda.is_available() else torch.float32), device_map="auto" if torch.cuda.is_available() else None)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer

def sequence_logprob(model, tokenizer, prompt: str, response: str, max_len=1024):
    text = (prompt + " " + response).strip() if prompt else response
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"]
    attn_mask = enc["attention_mask"]
    if input_ids.size(1) > max_len:
        input_ids = input_ids[:, -max_len:]
        attn_mask = attn_mask[:, -max_len:]
    input_ids = input_ids.to(model.device)
    attn_mask = attn_mask.to(model.device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask, labels=input_ids)
        logits = out.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        logprobs = torch.log_softmax(logits, dim=-1)
        token_lp = logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    if prompt:
        p_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
        prompt_len = p_ids.size(1)
    else:
        prompt_len = 0

    resp_lp = token_lp[:, prompt_len-1:].sum().item() if prompt_len > 0 else token_lp.sum().item()
    return resp_lp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--adapter_path", type=str, default=None)
    ap.add_argument("--eval_jsonl", type=str, required=True)
    ap.add_argument("--max_eval_samples", type=int, default=500)
    ap.add_argument("--report_path", type=str, default="outputs/eval_report.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model, tokenizer = load_model_tokenizer(args.base_model, args.adapter_path, dtype=dtype)

    samples = []
    with open(args.eval_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    if len(samples) > args.max_eval_samples:
        samples = samples[:args.max_eval_samples]

    wins, ties = 0, 0
    for ex in samples:
        p = ex.get("prompt", "")
        c = ex["chosen"]
        r = ex["rejected"]
        lp_c = sequence_logprob(model, tokenizer, p, c)
        lp_r = sequence_logprob(model, tokenizer, p, r)
        if lp_c > lp_r:
            wins += 1
        elif lp_c == lp_r:
            ties += 1

    total = len(samples)
    win_rate = wins / total if total > 0 else 0.0
    report = {"total": total, "wins": wins, "ties": ties, "win_rate": win_rate}
    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[EVAL] win_rate={win_rate:.3f}  ({wins}/{total}, ties={ties})")
    print(f"[OK] Rapor: {args.report_path}")

if __name__ == "__main__":
    main()
