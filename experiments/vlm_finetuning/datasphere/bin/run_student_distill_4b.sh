#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if [ ! -f data/derived/training/vlm_distill_train.jsonl ]; then
  echo "Missing data/derived/training/vlm_distill_train.jsonl. Build or upload the teacher-generated silver corpus first." >&2
  exit 2
fi
EXTRA_EVAL=()
if [ -f data/derived/training/vlm_distill_eval.jsonl ]; then
  EXTRA_EVAL+=(--eval-file data/derived/training/vlm_distill_eval.jsonl)
fi
run_with_torchrun experiments/vlm_finetuning/scripts/train_vlm_sft.py \
  --model-id Qwen/Qwen3-VL-4B-Instruct \
  --train-file data/derived/training/vlm_distill_train.jsonl \
  "${EXTRA_EVAL[@]}" \
  --output-dir outputs/student_distill_qwen3vl_4b \
  --train-mode text \
  --use-lora \
  --bf16 \
  --gradient-checkpointing \
  --learning-rate 8e-5 \
  --num-train-epochs 2 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --save-steps 100 \
  --logging-steps 10 \
  --save-adapter-only

tar -czf outputs/student_distill_qwen3vl_4b.tar.gz outputs/student_distill_qwen3vl_4b
