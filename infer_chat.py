#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SYSTEM_PROMPT = "Aşağıda kullanıcı ile yardımsever, doğru ve zararsız bir asistan arasındaki sohbet yer alıyor."

def load_model_tokenizer(base_model: str, adapter_path: str = None, dtype=None):
    tok = AutoTokenizer.from_pretrained(adapter_path or base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32), device_map="auto" if torch.cuda.is_available() else None)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter_path", type=str, default="models/qwen25-0_5b-dpo")
    args = ap.parse_args()

    model, tok = load_model_tokenizer(args.base_model, args.adapter_path)

    print("\n[Interactive Chat] Çıkmak için Ctrl+C")
    while True:
        try:
            user = input("\n[Kullanıcı] ")
        except KeyboardInterrupt:
            print("\nÇıkılıyor.")
            break
        if not user.strip():
            continue

        prompt = f"{SYSTEM_PROMPT}\n\nKullanıcı: {user}\nAsistan:"
        enc = tok(prompt, return_tensors="pt").to(model.device)
        gen = model.generate(
            **enc,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
        out_text = tok.decode(gen[0], skip_special_tokens=True)
        if "Asistan:" in out_text:
            out_text = out_text.split("Asistan:")[-1].strip()
        print(f"[Asistan] {out_text}")

if __name__ == "__main__":
    main()
