#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

python experiments/vlm_finetuning/scripts/build_vlm_sft_dataset.py \
  --repo-root "$ROOT" \
  --output-train data/derived/training/vlm_sft_train.jsonl \
  --output-eval data/derived/training/vlm_sft_eval.jsonl \
  --output-all data/derived/training/vlm_sft.jsonl \
  --output-smoke data/derived/training/vlm_sft_smoke.jsonl \
  --output-summary reports/build_summary.txt

python experiments/vlm_finetuning/scripts/build_vlm_preference_dataset.py \
  --repo-root "$ROOT" \
  --output-train data/derived/training/vlm_dpo_train.jsonl \
  --output-eval data/derived/training/vlm_dpo_eval.jsonl \
  --output-all data/derived/training/vlm_dpo.jsonl \
  --output-summary reports/dpo_build_summary.txt

python experiments/vlm_finetuning/scripts/estimate_datasphere_costs.py --scenario sft_smoke --out reports/datasphere_cost_sft_smoke.txt
python experiments/vlm_finetuning/scripts/estimate_datasphere_costs.py --scenario sft_pilot --out reports/datasphere_cost_sft_pilot.txt
python experiments/vlm_finetuning/scripts/estimate_datasphere_costs.py --scenario dpo_pilot --out reports/datasphere_cost_dpo_pilot.txt
python experiments/vlm_finetuning/scripts/estimate_datasphere_costs.py --scenario teacher_30b_a3b_sft --out reports/datasphere_cost_teacher_30b_a3b_sft.txt
python experiments/vlm_finetuning/scripts/estimate_datasphere_costs.py --scenario student_distill_4b --out reports/datasphere_cost_student_distill_4b.txt
