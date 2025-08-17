# Gerçek DPO İnce Ayar Projesi (OSS Modellerle)

Bu proje, **Direct Preference Optimization (DPO)** ile **açık kaynak** bir dil modelini gerçekten
eğitmek için **uçtan uca çalışan** bir kurulum sağlar. Kodlar **simülasyon** yapmaz; gerçek model,
gerçek veri ve gerçek kayıplarla eğitilir. Varsayılan temel model: **Qwen2.5-0.5B-Instruct** (Apache-2.0).

## Hızlı Başlangıç
1) Colab/Kaggle GPU açın. 
2) `pip install -r requirements.txt`
3) Veri dönüştürme (gerçek veri): 
```
python scripts/convert_hhrlhf_to_dpo.py --dataset_name Anthropic/hh-rlhf --split train --max_samples 20000 --output_jsonl data/hh_rlhf_train.jsonl
python scripts/convert_hhrlhf_to_dpo.py --dataset_name Anthropic/hh-rlhf --split test  --max_samples 2000  --output_jsonl data/hh_rlhf_eval.jsonl
```
4) Eğitim:
```
python train_dpo.py --base_model Qwen/Qwen2.5-0.5B-Instruct --train_jsonl data/hh_rlhf_train.jsonl --eval_jsonl data/hh_rlhf_eval.jsonl --output_dir models/qwen25-0_5b-dpo --learning_rate 1e-5 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 4 --max_steps 1000 --eval_steps 100 --save_steps 200 --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --bf16
```
5) Değerlendirme:
```
python evaluate_pairwise.py --base_model Qwen/Qwen2.5-0.5B-Instruct --adapter_path models/qwen25-0_5b-dpo --eval_jsonl data/hh_rlhf_eval.jsonl --max_eval_samples 500 --report_path outputs/eval_report.json
```
6) İnferans:
```
python infer_chat.py --base_model Qwen/Qwen2.5-0.5B-Instruct --adapter_path models/qwen25-0_5b-dpo
```

### Notlar
- 4-bit QLoRA GPU var ise otomatik devreye girer. CPU'da da çalışır (yavaş).
- Alternatif temel modeller: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, `Qwen/Qwen2.5-1.5B-Instruct`.

