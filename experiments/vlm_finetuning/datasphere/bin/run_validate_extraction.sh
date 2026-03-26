#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

mkdir -p reports/validation
if [ ! -f data/derived/training/vlm_sft_eval.jsonl ]; then
  bash experiments/vlm_finetuning/datasphere/bin/run_build_datasets.sh
fi
python experiments/vlm_finetuning/scripts/validate_extraction_run.py \
  --predictions data/derived/training/vlm_sft_eval.jsonl \
  --gold data/derived/training/vlm_sft_eval.jsonl \
  --out-dir reports/validation

tar -czf reports/validation.tar.gz reports/validation
