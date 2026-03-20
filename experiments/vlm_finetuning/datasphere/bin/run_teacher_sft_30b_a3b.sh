#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

bash experiments/vlm_finetuning/datasphere/bin/run_build_datasets.sh
run_with_torchrun experiments/vlm_finetuning/scripts/train_vlm_sft.py \
  --model-id Qwen/Qwen3-VL-30B-A3B-Instruct \
  --train-file data/derived/training/vlm_sft_train.jsonl \
  --eval-file data/derived/training/vlm_sft_eval.jsonl \
  --output-dir outputs/teacher_sft_qwen3vl_30b_a3b_lora \
  --train-mode text \
  --use-lora \
  --bf16 \
  --gradient-checkpointing \
  --learning-rate 5e-5 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 32 \
  --save-steps 100 \
  --eval-steps 100 \
  --logging-steps 10 \
  --save-adapter-only

tar -czf outputs/teacher_sft_qwen3vl_30b_a3b_lora.tar.gz outputs/teacher_sft_qwen3vl_30b_a3b_lora
