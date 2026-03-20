#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

bash experiments/vlm_finetuning/datasphere/bin/run_build_datasets.sh
run_with_torchrun experiments/vlm_finetuning/scripts/train_vlm_sft.py \
  --model-id Qwen/Qwen3-VL-4B-Instruct \
  --train-file data/derived/training/vlm_sft_train.jsonl \
  --eval-file data/derived/training/vlm_sft_eval.jsonl \
  --output-dir outputs/sft_pilot_qwen3vl_4b_lora \
  --train-mode text \
  --use-lora \
  --bf16 \
  --gradient-checkpointing \
  --learning-rate 1e-4 \
  --num-train-epochs 2 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --save-steps 100 \
  --eval-steps 100 \
  --logging-steps 10 \
  --save-adapter-only

tar -czf outputs/sft_pilot_qwen3vl_4b_lora.tar.gz outputs/sft_pilot_qwen3vl_4b_lora
