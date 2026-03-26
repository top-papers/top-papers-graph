#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

bash experiments/vlm_finetuning/datasphere/bin/run_build_datasets.sh
run_with_torchrun experiments/vlm_finetuning/scripts/train_vlm_sft.py \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --train-file data/derived/training/vlm_sft_smoke.jsonl \
  --eval-file data/derived/training/vlm_sft_eval.jsonl \
  --output-dir outputs/sft_smoke_qwen25vl_3b_qlora \
  --train-mode text \
  --qlora \
  --bf16 \
  --gradient-checkpointing \
  --learning-rate 2e-4 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --save-steps 50 \
  --eval-steps 50 \
  --logging-steps 10 \
  --save-adapter-only

tar -czf outputs/sft_smoke_qwen25vl_3b_qlora.tar.gz outputs/sft_smoke_qwen25vl_3b_qlora
