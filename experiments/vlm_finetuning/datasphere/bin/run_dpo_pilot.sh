#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

bash experiments/vlm_finetuning/datasphere/bin/run_build_datasets.sh
EXTRA_ARGS=()
if [ -n "${SFT_ADAPTER_PATH:-}" ]; then
  EXTRA_ARGS+=(--sft-adapter-path "$SFT_ADAPTER_PATH")
fi
run_with_torchrun experiments/vlm_finetuning/scripts/train_vlm_dpo.py \
  --model-id Qwen/Qwen3-VL-4B-Instruct \
  --train-file data/derived/training/vlm_dpo_train.jsonl \
  --eval-file data/derived/training/vlm_dpo_eval.jsonl \
  --output-dir outputs/dpo_pilot_qwen3vl_4b_lora \
  --train-mode text \
  --use-lora \
  --bf16 \
  --gradient-checkpointing \
  --learning-rate 5e-6 \
  --beta 0.1 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --save-steps 100 \
  --eval-steps 100 \
  --logging-steps 10 \
  "${EXTRA_ARGS[@]}"

tar -czf outputs/dpo_pilot_qwen3vl_4b_lora.tar.gz outputs/dpo_pilot_qwen3vl_4b_lora
