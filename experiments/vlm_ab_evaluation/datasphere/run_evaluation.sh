#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

set -euo pipefail

CONFIG="${VLM_AB_CONFIG:-experiments/vlm_ab_evaluation/configs/qwen3vl_scireason_publication.yaml}"
RUNNER="experiments/vlm_ab_evaluation/run_pipeline.py"
EXPLORATORY_ARGS=()
if [ "${VLM_AB_EXPLORATORY:-0}" = "1" ]; then
  EXPLORATORY_ARGS+=(--exploratory)
fi

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python "$RUNNER" --config "$CONFIG" plan
python "$RUNNER" --config "$CONFIG" prepare "${EXPLORATORY_ARGS[@]}"

# Separate processes guarantee that one arm is released before the next model is loaded.
python "$RUNNER" --config "$CONFIG" infer --arm base --backend transformers "${EXPLORATORY_ARGS[@]}"
python "$RUNNER" --config "$CONFIG" infer --arm tuned --backend transformers "${EXPLORATORY_ARGS[@]}"
python "$RUNNER" --config "$CONFIG" blind "${EXPLORATORY_ARGS[@]}"
