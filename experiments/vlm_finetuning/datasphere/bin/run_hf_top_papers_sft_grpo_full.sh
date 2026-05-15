#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

DATASET_ID="${HF_DATASET_ID:-top-papers/top-papers-graph-experts-data}"
DATASET_SPLIT="${HF_DATASET_SPLIT:-validation}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
OUT_PREFIX="${OUT_PREFIX:-hf_top_papers_qwen3vl_8b}"
DATA_DIR="data/derived/hf_top_papers_graph_experts"
SFT_DIR="outputs/${OUT_PREFIX}_sft_lora"
GRPO_DIR="outputs/${OUT_PREFIX}_grpo_lora"
REPORT_DIR="reports/${OUT_PREFIX}_datasphere"
BUDGET_RUB="${BUDGET_RUB:-100000}"
G2_2_RUB_PER_HOUR="${G2_2_RUB_PER_HOUR:-1085.76}"
MAX_SFT_STEPS="${MAX_SFT_STEPS:-180}"
MAX_GRPO_STEPS="${MAX_GRPO_STEPS:-80}"
SFT_TIMEOUT_HOURS="${SFT_TIMEOUT_HOURS:-30}"
GRPO_TIMEOUT_HOURS="${GRPO_TIMEOUT_HOURS:-45}"
DATA_TIMEOUT_HOURS="${DATA_TIMEOUT_HOURS:-4}"

mkdir -p "$REPORT_DIR" outputs data/derived

write_budget_plan() {
  python - <<PY
import json
budget = float("$BUDGET_RUB")
price = float("$G2_2_RUB_PER_HOUR")
plan = {
    "budget_rub": budget,
    "datasphere_instance_type": "g2.2",
    "price_rub_per_hour_used_for_guard": price,
    "max_theoretical_hours_before_storage_and_traffic": round(budget / price, 2),
    "phase_timeouts_hours": {
        "dataset_prepare": float("$DATA_TIMEOUT_HOURS"),
        "sft": float("$SFT_TIMEOUT_HOURS"),
        "grpo": float("$GRPO_TIMEOUT_HOURS"),
    },
    "configured_training_steps": {
        "sft": int("$MAX_SFT_STEPS"),
        "grpo": int("$MAX_GRPO_STEPS"),
    },
    "note": "DataSphere bills jobs per second on the selected configuration; the phase timeouts are a local hard stop. Configure official project spending thresholds in DataSphere for an account-level hard limit.",
}
print(json.dumps(plan, ensure_ascii=False, indent=2))
PY
}

write_budget_plan | tee "$REPORT_DIR/budget_plan.json"

run_timeout() {
  local hours="$1"
  shift
  local seconds
  seconds="$(python - <<PY
print(int(float("$hours") * 3600))
PY
)"
  echo "[datasphere-pipeline] running with timeout ${hours}h: $*"
  timeout --signal=TERM --kill-after=1800s "${seconds}s" "$@"
}

run_torchrun_timeout() {
  local hours="$1"
  shift
  local nproc
  nproc="$(python_gpu_count)"
  if [ -z "$nproc" ] || [ "$nproc" -lt 1 ]; then
    nproc=1
  fi
  run_timeout "$hours" torchrun --standalone --nproc_per_node="$nproc" "$@"
}

package_artifacts() {
  set +e
  mkdir -p "$REPORT_DIR"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$REPORT_DIR/finished_at_utc.txt"
  find outputs "$DATA_DIR" "$REPORT_DIR" -maxdepth 3 -type f 2>/dev/null | sort > "$REPORT_DIR/artifact_manifest.txt"
  [ -d "$SFT_DIR" ] && tar -czf "outputs/${OUT_PREFIX}_sft_lora.tar.gz" "$SFT_DIR"
  [ -d "$GRPO_DIR" ] && tar -czf "outputs/${OUT_PREFIX}_grpo_lora.tar.gz" "$GRPO_DIR"
  tar -czf "reports/${OUT_PREFIX}_datasphere_reports.tar.gz" "$REPORT_DIR" 2>/dev/null
}
trap 'status=$?; package_artifacts; exit $status' EXIT

run_timeout "$DATA_TIMEOUT_HOURS" python experiments/vlm_finetuning/scripts/build_hf_graph_experts_dataset.py \
  --dataset-id "$DATASET_ID" \
  --split "$DATASET_SPLIT" \
  --out-dir "$DATA_DIR" \
  --eval-ratio 0.15 \
  --seed 42

run_torchrun_timeout "$SFT_TIMEOUT_HOURS" experiments/vlm_finetuning/scripts/train_vlm_sft.py \
  --model-id "$BASE_MODEL" \
  --train-file "$DATA_DIR/sft_train.jsonl" \
  --eval-file "$DATA_DIR/sft_eval.jsonl" \
  --output-dir "$SFT_DIR" \
  --train-mode vlm \
  --image-column image \
  --use-lora \
  --bf16 \
  --gradient-checkpointing \
  --learning-rate "${SFT_LR:-1e-4}" \
  --max-steps "$MAX_SFT_STEPS" \
  --num-train-epochs "${SFT_EPOCHS:-3}" \
  --per-device-train-batch-size "${SFT_PER_DEVICE_BATCH:-1}" \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps "${SFT_GRAD_ACCUM:-8}" \
  --save-steps "${SFT_SAVE_STEPS:-50}" \
  --eval-steps "${SFT_EVAL_STEPS:-50}" \
  --logging-steps 5 \
  --save-adapter-only

tar -czf "outputs/${OUT_PREFIX}_sft_lora.tar.gz" "$SFT_DIR"

run_torchrun_timeout "$GRPO_TIMEOUT_HOURS" experiments/vlm_finetuning/scripts/train_vlm_grpo.py \
  --model-id "$BASE_MODEL" \
  --sft-adapter-path "$SFT_DIR" \
  --train-file "$DATA_DIR/grpo_train.jsonl" \
  --eval-file "$DATA_DIR/grpo_eval.jsonl" \
  --output-dir "$GRPO_DIR" \
  --train-mode vlm \
  --image-column images \
  --bf16 \
  --gradient-checkpointing \
  --learning-rate "${GRPO_LR:-2e-5}" \
  --max-steps "$MAX_GRPO_STEPS" \
  --num-train-epochs "${GRPO_EPOCHS:-1}" \
  --per-device-train-batch-size "${GRPO_PER_DEVICE_BATCH:-1}" \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps "${GRPO_GRAD_ACCUM:-8}" \
  --num-generations "${GRPO_NUM_GENERATIONS:-2}" \
  --max-completion-length "${GRPO_MAX_COMPLETION_LENGTH:-64}" \
  --save-steps "${GRPO_SAVE_STEPS:-40}" \
  --eval-steps "${GRPO_EVAL_STEPS:-40}" \
  --logging-steps 5

tar -czf "outputs/${OUT_PREFIX}_grpo_lora.tar.gz" "$GRPO_DIR"
python - <<PY > "$REPORT_DIR/final_summary.json"
import json
from pathlib import Path
summary = {
    "dataset_id": "$DATASET_ID",
    "base_model": "$BASE_MODEL",
    "sft_dir": "$SFT_DIR",
    "grpo_dir": "$GRPO_DIR",
    "sft_archive": "outputs/${OUT_PREFIX}_sft_lora.tar.gz",
    "grpo_archive": "outputs/${OUT_PREFIX}_grpo_lora.tar.gz",
    "report_archive": "reports/${OUT_PREFIX}_datasphere_reports.tar.gz",
}
for path in [Path("$DATA_DIR/summary.json"), Path("$SFT_DIR/run_config.json"), Path("$GRPO_DIR/planned_run_config.json")]:
    if path.exists():
        summary[path.stem] = json.loads(path.read_text(encoding="utf-8"))
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
