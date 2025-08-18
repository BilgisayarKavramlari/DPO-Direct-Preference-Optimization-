#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pairwise evaluation for DPO-style datasets (Transformers 4.55.x + TRL 0.21 + PEFT 0.17 uyumlu)

- Tokenizer her zaman BASE modelden yüklenir (adapter klasöründe tokenizer yoktur).
- LoRA adaptörü yalnızca modelin üzerine bind edilir.
- stdout flush ile detaylı günlük: her adımda çıktı görürsünüz.
- prompt tokenları -100 ile maskelenir; yalnızca response tokenları üzerinden log-olabilirlik hesaplanır.
"""

import os, json, argparse, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def log(msg: str):
    print(msg, flush=True)


def load_model_tokenizer(base_model: str, adapter_path: str | None, dtype):
    log(f"[LOAD] base_model={base_model}")
    # FAST tokenizer => protobuf bağımlılığına düşmeyelim
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto" if torch.cuda.is_available() else None
    log(f"[LOAD] model (dtype={dtype}, device_map={device_map})")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=device_map,
    )

    if adapter_path:
        if not os.path.isdir(adapter_path):
            log(f"[WARN] adapter_path klasörü yok: {adapter_path}")
        else:
            log(f"[LOAD] attaching LoRA adapter: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


@torch.no_grad()
def response_logprob(model, tokenizer, prompt: str, response: str, max_len: int = 1024) -> float:
    """
    Log P(response | prompt) ~ - average CE over response tokens * (#response_tokens)
    """
    device = model.device

    enc_p = tokenizer(prompt or "", add_special_tokens=False, return_tensors="pt")
    enc_r = tokenizer(response, add_special_tokens=False, return_tensors="pt")

    input_ids = torch.cat([enc_p["input_ids"], enc_r["input_ids"]], dim=1)
    attn_mask = torch.ones_like(input_ids)

    # Sol kesim: en yeni tokenlar kalsın
    if input_ids.size(1) > max_len:
        input_ids = input_ids[:, -max_len:]
        attn_mask = attn_mask[:, -max_len:]

    prompt_len = enc_p["input_ids"].size(1)
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100  # prompt'u ignore et

    num_resp_tokens = int((labels != -100).sum().item())
    if num_resp_tokens == 0:
        # boş cevap gibi durumlar
        return float("-inf")

    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)
    labels = labels.to(device)

    out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
    avg_nll = float(out.loss.item())  # CE ortalaması sadece response tokenları üzerinde
    total_logprob = -avg_nll * num_resp_tokens
    return total_logprob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--adapter_path", type=str, default=None)
    ap.add_argument("--eval_jsonl", type=str, required=True)
    ap.add_argument("--max_eval_samples", type=int, default=500)
    ap.add_argument("--report_path", type=str, default="outputs/eval_report.json")
    ap.add_argument("--max_length", type=int, default=1024, help="max total sequence length for scoring")
    args = ap.parse_args()

    # Çalışma dizinini ve yolları yaz
    log(f"[INFO] CWD={os.getcwd()}")
    log(f"[INFO] eval_jsonl={args.eval_jsonl}")
    log(f"[INFO] report_path={args.report_path}")

    # Rapor klasörünü hazırla
    out_dir = os.path.dirname(args.report_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Veri var mı?
    if not os.path.exists(args.eval_jsonl):
        log(f"[ERROR] Değerlendirme dosyası bulunamadı: {args.eval_jsonl}")
        # Yine de boş rapor bırak
        with open(args.report_path, "w", encoding="utf-8") as f:
            json.dump({"total": 0, "wins": 0, "ties": 0, "win_rate": 0.0, "error": "eval_jsonl not found"}, f, ensure_ascii=False, indent=2)
        sys.exit(1)

    # Model/Tokenizer yükle
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model, tokenizer = load_model_tokenizer(args.base_model, args.adapter_path, dtype=dtype)

    # Örnekleri oku
    log("[READ] loading eval samples...")
    samples = []
    with open(args.eval_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    total_raw = len(samples)
    if total_raw == 0:
        log("[WARN] eval_jsonl boş.")
    if total_raw > args.max_eval_samples:
        samples = samples[:args.max_eval_samples]
    log(f"[READ] total={total_raw}, eval_count={len(samples)}")

    # Değerlendir
    wins, ties, total = 0, 0, 0
    for i, ex in enumerate(samples, 1):
        if "chosen" not in ex or "rejected" not in ex:
            # DPO formatı değilse atla
            continue
        p = ex.get("prompt", "") or ""
        c = ex["chosen"]
        r = ex["rejected"]

        lp_c = response_logprob(model, tokenizer, p, c, max_len=args.max_length)
        lp_r = response_logprob(model, tokenizer, p, r, max_len=args.max_length)

        if lp_c > lp_r:
            wins += 1
        elif lp_c == lp_r:
            ties += 1
        total += 1

        if i % 50 == 0:
            log(f"[PROGRESS] {i}/{len(samples)} processed... wins={wins}, ties={ties}")

    win_rate = (wins / total) if total else 0.0
    report = {"total": total, "wins": wins, "ties": ties, "win_rate": win_rate}

    # Raporu yaz
    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    log(f"[EVAL] win_rate={win_rate:.3f}  (wins={wins}, ties={ties}, total={total})")
    log(f"[OK] Rapor kaydedildi: {args.report_path}")


if __name__ == "__main__":
    main()
