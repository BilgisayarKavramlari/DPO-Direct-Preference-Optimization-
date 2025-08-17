#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, os, json, re
from datasets import load_dataset

def longest_common_prefix(a: str, b: str) -> str:
    a_tokens = a.split()
    b_tokens = b.split()
    i = 0
    for x, y in zip(a_tokens, b_tokens):
        if x == y:
            i += 1
        else:
            break
    return " ".join(a_tokens[:i])

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", type=str, required=True)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--max_samples", type=int, default=50000)
    ap.add_argument("--output_jsonl", type=str, required=True)
    args = ap.parse_args()

    ds = load_dataset(args.dataset_name, split=args.split)
    n = min(len(ds), args.max_samples)

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    out_f = open(args.output_jsonl, "w", encoding="utf-8")
    kept = 0

    for i in range(n):
        ex = ds[i]
        if all(k in ex for k in ["prompt", "chosen", "rejected"]):
            prompt = normalize_text(ex["prompt"])
            chosen = normalize_text(ex["chosen"])
            rejected = normalize_text(ex["rejected"])
        elif all(k in ex for k in ["chosen", "rejected"]):
            c = normalize_text(ex["chosen"]); r = normalize_text(ex["rejected"])
            lcp = longest_common_prefix(c, r)
            if lcp and c.startswith(lcp) and r.startswith(lcp):
                prompt = lcp.strip()
                chosen, rejected = c[len(lcp):].strip(), r[len(lcp):].strip()
            else:
                prompt, chosen, rejected = "", c, r
        elif all(k in ex for k in ["question", "response_j", "response_k", "label"]):
            prompt = normalize_text(ex["question"])
            j = normalize_text(ex["response_j"]); k = normalize_text(ex["response_k"]); label = int(ex["label"])
            chosen, rejected = (j, k) if label == 1 else (k, j)
        else:
            continue

        if not chosen or not rejected or chosen == rejected:
            continue

        out_f.write(json.dumps({"prompt": prompt, "chosen": chosen, "rejected": rejected}, ensure_ascii=False) + "\n")
        kept += 1

    out_f.close()
    print(f"Wrote {kept} examples to {args.output_jsonl}")

if __name__ == "__main__":
    main()
