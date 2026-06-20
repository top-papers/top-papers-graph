#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# DataSphere secret bridge: a project secret named HFTOKEN is exposed as $HFTOKEN,
# while Hugging Face Hub libraries expect HF_TOKEN/HUGGING_FACE_HUB_TOKEN.
if [ -z "${HF_TOKEN:-}" ] && [ -n "${HFTOKEN:-}" ]; then
  export HF_TOKEN="$HFTOKEN"
fi
if [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ] && [ -n "${HF_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

# Managed DataSphere images can start jobs with an ASCII default locale. TRL's
# optional model-card generation and Hugging Face Hub templates contain UTF-8
# characters, so force a UTF-8 process locale before Python imports anything.
export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export PYTHONUTF8="${PYTHONUTF8:-1}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
# The pipeline writes explicit run_config/final_summary metadata. Disable TRL's
# optional README/model-card side effect by default; it can fail on locale quirks
# and is not needed for checkpoints or adapter archives.
export DISABLE_TRL_MODEL_CARD="${DISABLE_TRL_MODEL_CARD:-1}"

DATASET_ID="${HF_DATASET_ID:-top-papers/top-papers-graph-experts-data}"
DATASET_SPLIT="${HF_DATASET_SPLIT:-validation}"
DATASET_REVISION="${HF_DATASET_REVISION:-main}"
DATASET_SOURCE_MODE="${HF_DATASET_SOURCE_MODE:-export}"
DATASET_EXPORT_SUBDIR="${HF_DATASET_EXPORT_SUBDIR:-exports/colab-run-001}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
OUT_PREFIX="${OUT_PREFIX:-hf_top_papers_qwen3vl_8b}"
DATA_DIR="data/derived/hf_top_papers_graph_experts"
SFT_DIR="outputs/${OUT_PREFIX}_sft_lora"
GRPO_DIR="outputs/${OUT_PREFIX}_grpo_lora"
REPORT_DIR="reports/${OUT_PREFIX}_datasphere"
export REPORT_DIR
HF_REPO_ID="${HF_REPO_ID:-top-papers/Qwen3-VL-8B-Instruct-scireason}"
HF_REPO_TYPE="${HF_REPO_TYPE:-model}"
HF_REVISION="${HF_REVISION:-main}"
HF_UPLOAD_AFTER_TRAINING="${HF_UPLOAD_AFTER_TRAINING:-1}"
HF_UPLOAD_PATH_PREFIX="${HF_UPLOAD_PATH_PREFIX:-}"
HF_UPLOAD_BUNDLE_DIR="${HF_UPLOAD_BUNDLE_DIR:-$REPORT_DIR/hf_upload_bundle}"
BUDGET_RUB="${BUDGET_RUB:-100000}"
BUDGET_RESERVE_RUB="${BUDGET_RESERVE_RUB:-5000}"
G2_2_RUB_PER_HOUR="${G2_2_RUB_PER_HOUR:-1085.76}"
MAX_SFT_STEPS="${MAX_SFT_STEPS:--1}"
MAX_GRPO_STEPS="${MAX_GRPO_STEPS:-160}"
SFT_TIMEOUT_HOURS="${SFT_TIMEOUT_HOURS:-48}"
GRPO_TIMEOUT_HOURS="${GRPO_TIMEOUT_HOURS:-35}"
DATA_TIMEOUT_HOURS="${DATA_TIMEOUT_HOURS:-3}"
HF_UPLOAD_TIMEOUT_HOURS="${HF_UPLOAD_TIMEOUT_HOURS:-1}"
BUDGET_SHUTDOWN_MARGIN_SECONDS="${BUDGET_SHUTDOWN_MARGIN_SECONDS:-900}"
EVAL_RATIO="${EVAL_RATIO:-0.10}"
MAX_IMAGES_PER_EXAMPLE_SFT="${MAX_IMAGES_PER_EXAMPLE_SFT:-0}"
MAX_IMAGES_PER_EXAMPLE_GRPO="${MAX_IMAGES_PER_EXAMPLE_GRPO:-0}"
# Smoke/debug caps. If the smoke config is accidentally not propagated by an
# older DataSphere CLI/cache, still keep OUT_PREFIX=*smoke* runs small enough to
# avoid downloading the whole HF export asset tree.
if [[ "${OUT_PREFIX:-}" == *smoke* ]]; then
  MAX_SFT_SAMPLES="${MAX_SFT_SAMPLES:-96}"
  MAX_GRPO_SAMPLES="${MAX_GRPO_SAMPLES:-48}"
  MAX_DATASET_SAMPLES="${MAX_DATASET_SAMPLES:-0}"
  # GRPO computes group-relative advantages, so even smoke runs need at least
  # two completions per prompt. Stale configs with value 1 fail before training.
  GRPO_NUM_GENERATIONS="${GRPO_NUM_GENERATIONS:-2}"
  GRPO_NUM_GENERATIONS_EVAL="${GRPO_NUM_GENERATIONS_EVAL:-2}"
else
  MAX_SFT_SAMPLES="${MAX_SFT_SAMPLES:-0}"
  MAX_GRPO_SAMPLES="${MAX_GRPO_SAMPLES:-0}"
  MAX_DATASET_SAMPLES="${MAX_DATASET_SAMPLES:-0}"
  GRPO_NUM_GENERATIONS="${GRPO_NUM_GENERATIONS:-2}"
  GRPO_NUM_GENERATIONS_EVAL="${GRPO_NUM_GENERATIONS_EVAL:-2}"
fi
if [ "${GRPO_NUM_GENERATIONS:-0}" -lt 2 ]; then
  echo "[datasphere-pipeline] GRPO_NUM_GENERATIONS=${GRPO_NUM_GENERATIONS} is invalid for GRPO; forcing 2." >&2
  GRPO_NUM_GENERATIONS=2
fi
if [ "${GRPO_NUM_GENERATIONS_EVAL:-0}" -lt 2 ]; then
  echo "[datasphere-pipeline] GRPO_NUM_GENERATIONS_EVAL=${GRPO_NUM_GENERATIONS_EVAL} is invalid for GRPO; forcing 2." >&2
  GRPO_NUM_GENERATIONS_EVAL=2
fi
HF_DOWNLOAD_MAX_WORKERS="${HF_DOWNLOAD_MAX_WORKERS:-2}"
VLM_MIN_PIXELS="${VLM_MIN_PIXELS:-}"
VLM_MAX_PIXELS="${VLM_MAX_PIXELS:-1003520}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-auto}"

prefetch_base_model_and_enable_offline_hub() {
  if [ "${PREFETCH_BASE_MODEL:-1}" = "1" ]; then
    echo "[datasphere-pipeline] prefetching base model into HF cache before offline training: $BASE_MODEL"
    BASE_MODEL="$BASE_MODEL" BASE_MODEL_REVISION="${BASE_MODEL_REVISION:-main}" python - <<'MODEL_PREFETCH_PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=os.environ["BASE_MODEL"],
    repo_type="model",
    revision=os.environ.get("BASE_MODEL_REVISION") or None,
)
MODEL_PREFETCH_PY
  fi
  if [ "${ENABLE_TRAINING_HF_OFFLINE:-1}" = "1" ]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    echo "[datasphere-pipeline] enabled HF_HUB_OFFLINE=1 for training stages after prefetch."
  fi
}

append_optional_bool_flag() {
  local var_name="$1"
  local flag="$2"
  local no_flag="$3"
  local value="${!var_name:-}"
  case "$value" in
    1|true|TRUE|yes|YES|on|ON)
      printf '%s\n' "$flag"
      ;;
    0|false|FALSE|no|NO|off|OFF|'')
      if [ -n "$no_flag" ] && [ -n "$value" ]; then
        printf '%s\n' "$no_flag"
      fi
      ;;
    *)
      echo "[datasphere-pipeline] Unsupported boolean $var_name=$value; expected 0/1/true/false." >&2
      exit 2
      ;;
  esac
}

SFT_DDP_ARGS=()
while IFS= read -r flag; do
  [ -n "$flag" ] && SFT_DDP_ARGS+=("$flag")
done < <(append_optional_bool_flag SFT_DDP_FIND_UNUSED_PARAMETERS --ddp-find-unused-parameters --no-ddp-find-unused-parameters)
GRPO_DDP_ARGS=()
while IFS= read -r flag; do
  [ -n "$flag" ] && GRPO_DDP_ARGS+=("$flag")
done < <(append_optional_bool_flag GRPO_DDP_FIND_UNUSED_PARAMETERS --ddp-find-unused-parameters --no-ddp-find-unused-parameters)

mkdir -p "$REPORT_DIR" outputs data/derived
PIPELINE_STARTED_EPOCH="$(date +%s)"

calc_max_compute_seconds() {
  python - <<PY
budget = float("$BUDGET_RUB")
reserve = float("$BUDGET_RESERVE_RUB")
price = float("$G2_2_RUB_PER_HOUR")
margin = int(float("$BUDGET_SHUTDOWN_MARGIN_SECONDS"))
usable = max(0.0, budget - reserve)
seconds = int((usable / price) * 3600) - margin
print(max(1, seconds))
PY
}

MAX_COMPUTE_SECONDS="$(calc_max_compute_seconds)"
PIPELINE_DEADLINE_EPOCH="$((PIPELINE_STARTED_EPOCH + MAX_COMPUTE_SECONDS))"

remaining_budget_seconds() {
  local now remaining
  now="$(date +%s)"
  remaining="$((PIPELINE_DEADLINE_EPOCH - now))"
  if [ "$remaining" -lt 0 ]; then
    remaining=0
  fi
  printf '%s\n' "$remaining"
}

write_budget_plan() {
  python - <<PY
import json
budget = float("$BUDGET_RUB")
reserve = float("$BUDGET_RESERVE_RUB")
price = float("$G2_2_RUB_PER_HOUR")
phase_timeouts = {
    "dataset_prepare": float("$DATA_TIMEOUT_HOURS"),
    "sft": float("$SFT_TIMEOUT_HOURS"),
    "grpo": float("$GRPO_TIMEOUT_HOURS"),
    "hf_upload": float("$HF_UPLOAD_TIMEOUT_HOURS"),
}
configured_compute_hours = sum(phase_timeouts.values())
safe_compute_hours = max(0.0, (budget - reserve) / price) - float("$BUDGET_SHUTDOWN_MARGIN_SECONDS") / 3600.0
plan = {
    "budget_rub": budget,
    "reserve_for_storage_traffic_and_rounding_rub": reserve,
    "datasphere_instance_type": "g2.2",
    "price_rub_per_hour_used_for_guard": price,
    "max_theoretical_hours_before_storage_and_traffic": round(budget / price, 2),
    "safe_compute_hours_after_reserve_and_shutdown_margin": round(safe_compute_hours, 2),
    "configured_phase_timeouts_hours": phase_timeouts,
    "configured_phase_timeout_sum_hours": configured_compute_hours,
    "worst_case_compute_cost_rub_before_guard": round(configured_compute_hours * price, 2),
    "guarded_compute_cost_ceiling_rub": round(max(0.0, safe_compute_hours) * price, 2),
    "budget_guard": {
        "started_epoch": int("$PIPELINE_STARTED_EPOCH"),
        "max_compute_seconds": int("$MAX_COMPUTE_SECONDS"),
        "deadline_epoch": int("$PIPELINE_DEADLINE_EPOCH"),
        "shutdown_margin_seconds": int("$BUDGET_SHUTDOWN_MARGIN_SECONDS"),
    },
    "configured_training_steps": {
        "sft": int("$MAX_SFT_STEPS"),
        "grpo": int("$MAX_GRPO_STEPS"),
    },
    "dataset": {
        "id": "$DATASET_ID",
        "revision": "$DATASET_REVISION",
        "source_mode": "$DATASET_SOURCE_MODE",
        "export_subdir": "$DATASET_EXPORT_SUBDIR",
        "eval_ratio": float("$EVAL_RATIO"),
        "max_images_per_example_sft": int("$MAX_IMAGES_PER_EXAMPLE_SFT"),
        "max_images_per_example_grpo": int("$MAX_IMAGES_PER_EXAMPLE_GRPO"),
        "max_sft_samples": int("$MAX_SFT_SAMPLES"),
        "max_grpo_samples": int("$MAX_GRPO_SAMPLES"),
        "max_dataset_samples": int("$MAX_DATASET_SAMPLES"),
        "hf_download_max_workers": int("$HF_DOWNLOAD_MAX_WORKERS"),
    },
    "vlm_preprocessing": {
        "min_pixels": "$VLM_MIN_PIXELS" or None,
        "max_pixels": "$VLM_MAX_PIXELS" or None,
        "attn_implementation": "$ATTN_IMPLEMENTATION",
        "sft_max_text_chars": int("${SFT_MAX_TEXT_CHARS:-0}"),
        "sft_ddp_timeout_seconds": int("${SFT_DDP_TIMEOUT_SECONDS:-7200}"),
    },
    "huggingface_upload": {
        "enabled": "$HF_UPLOAD_AFTER_TRAINING" not in {"0", "false", "False", "no", "NO"},
        "repo_id": "$HF_REPO_ID",
        "repo_type": "$HF_REPO_TYPE",
        "revision": "$HF_REVISION",
        "path_in_repo": "$HF_UPLOAD_PATH_PREFIX" or ".",
    },
    "note": "DataSphere bills jobs per second on the selected configuration and additionally stores job data. This wrapper enforces a conservative wall-clock guard below the 100000 RUB project budget; still configure official cloud budget alerts/quotas for account-level protection.",
}
print(json.dumps(plan, ensure_ascii=False, indent=2))
PY
}

write_budget_plan | tee "$REPORT_DIR/budget_plan.json"

run_timeout_budgeted() {
  local hours="$1"
  shift
  local configured_seconds remaining seconds
  configured_seconds="$(python - <<PY
print(int(float("$hours") * 3600))
PY
)"
  remaining="$(remaining_budget_seconds)"
  if [ "$remaining" -le 0 ]; then
    echo "[datasphere-pipeline] budget guard reached before command: $*" >&2
    exit 90
  fi
  seconds="$configured_seconds"
  if [ "$seconds" -gt "$remaining" ]; then
    seconds="$remaining"
  fi
  echo "[datasphere-pipeline] running with timeout ${seconds}s (configured ${hours}h, remaining budget guard ${remaining}s): $*"
  timeout --signal=TERM --kill-after=1800s "${seconds}s" "$@"
}

run_torchrun_timeout_budgeted() {
  local hours="$1"
  shift
  local nproc
  nproc="$(python_gpu_count)"
  if [ -z "$nproc" ] || [ "$nproc" -lt 1 ]; then
    nproc=1
  fi
  run_timeout_budgeted "$hours" torchrun --standalone --nproc_per_node="$nproc" "$@"
}

prune_adapter_artifact_dir() {
  local dir="$1"
  [ -d "$dir" ] || return 0
  # GRPO only needs the adapter/config/tokenizer files at the directory root.
  # Trainer checkpoint folders often contain optimizer states and can make a
  # smoke-test archive exceed DataSphere's download size limit.
  find "$dir" -maxdepth 1 -type d -name 'checkpoint-*' -exec rm -rf {} + 2>/dev/null || true
  find "$dir" -type f \( -name 'optimizer.pt' -o -name 'scheduler.pt' -o -name 'rng_state*.pth' -o -name 'scaler.pt' \) -delete 2>/dev/null || true
}

tar_adapter_artifact_dir() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "$dst")"
  if [ ! -d "$src" ]; then
    local tmpdir
    tmpdir="$(mktemp -d)"
    mkdir -p "$tmpdir/placeholder"
    cat > "$tmpdir/placeholder/README.txt" <<PLACEHOLDER_EOF
This placeholder archive was created because $src did not exist when the DataSphere job packaged artifacts.
The training phase likely failed before this artifact was produced; see reports/${OUT_PREFIX}_datasphere/final_summary.json and job logs.
PLACEHOLDER_EOF
    tar -czf "$dst" -C "$tmpdir" placeholder
    rm -rf "$tmpdir"
    return 0
  fi
  prune_adapter_artifact_dir "$src"
  tar     --exclude='*/checkpoint-*'     --exclude='*/optimizer.pt'     --exclude='*/scheduler.pt'     --exclude='*/rng_state*.pth'     --exclude='*/scaler.pt'     -czf "$dst" "$src"
}

write_fallback_final_summary() {
  local exit_status="${1:-0}"
  [ -s "$REPORT_DIR/final_summary.json" ] && return 0
  python - <<PY > "$REPORT_DIR/final_summary.json"
import json
from pathlib import Path
summary = {
    "status": "incomplete_or_failed",
    "exit_status": int("$exit_status"),
    "dataset_id": "$DATASET_ID",
    "dataset_revision": "$DATASET_REVISION",
    "dataset_source_mode": "$DATASET_SOURCE_MODE",
    "dataset_export_subdir": "$DATASET_EXPORT_SUBDIR",
    "base_model": "$BASE_MODEL",
    "sft_dir": "$SFT_DIR",
    "grpo_dir": "$GRPO_DIR",
    "sft_archive": "outputs/${OUT_PREFIX}_sft_lora.tar.gz",
    "grpo_archive": "outputs/${OUT_PREFIX}_grpo_lora.tar.gz",
    "report_archive": "reports/${OUT_PREFIX}_datasphere_reports.tar.gz",
}
for key, path in {
    "budget_plan": Path("$REPORT_DIR/budget_plan.json"),
    "dataset_summary": Path("$DATA_DIR/summary.json"),
    "sft_run_config": Path("$SFT_DIR/run_config.json"),
    "grpo_planned_run_config": Path("$GRPO_DIR/planned_run_config.json"),
}.items():
    if path.exists():
        try:
            summary[key] = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            summary[key] = {"unreadable": str(exc), "path": str(path)}
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
}

package_artifacts() {
  local exit_status="${1:-0}"
  local had_errexit=0
  case "$-" in
    *e*) had_errexit=1 ;;
  esac
  set +e
  mkdir -p "$REPORT_DIR"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$REPORT_DIR/finished_at_utc.txt"
  remaining_budget_seconds > "$REPORT_DIR/remaining_budget_guard_seconds.txt"
  write_fallback_final_summary "$exit_status"
  if [ ! -s "$REPORT_DIR/hf_upload_summary.json" ]; then
    python - <<HF_UPLOAD_SUMMARY_PY > "$REPORT_DIR/hf_upload_summary.json"
import json
print(json.dumps({
    "status": "not_run_or_incomplete",
    "exit_status": int("$exit_status"),
    "reason": "pipeline packaged artifacts before Hugging Face upload completed",
}, ensure_ascii=False, indent=2))
HF_UPLOAD_SUMMARY_PY
  fi
  mkdir -p "$HF_UPLOAD_BUNDLE_DIR/artifacts/reports"
  if [ ! -s "$HF_UPLOAD_BUNDLE_DIR/artifacts/reports/hf_upload_manifest.json" ]; then
    python - <<HF_UPLOAD_MANIFEST_PY > "$HF_UPLOAD_BUNDLE_DIR/artifacts/reports/hf_upload_manifest.json"
import json
print(json.dumps({
    "status": "not_run_or_incomplete",
    "exit_status": int("$exit_status"),
    "reason": "pipeline packaged artifacts before Hugging Face upload completed",
}, ensure_ascii=False, indent=2))
HF_UPLOAD_MANIFEST_PY
  fi
  find outputs "$DATA_DIR" "$REPORT_DIR" -maxdepth 3 -type f 2>/dev/null | sort > "$REPORT_DIR/artifact_manifest.txt"
  tar_adapter_artifact_dir "$SFT_DIR" "outputs/${OUT_PREFIX}_sft_lora.tar.gz"
  tar_adapter_artifact_dir "$GRPO_DIR" "outputs/${OUT_PREFIX}_grpo_lora.tar.gz"
  tar --exclude="$REPORT_DIR/hf_upload_bundle" -czf "reports/${OUT_PREFIX}_datasphere_reports.tar.gz" "$REPORT_DIR" 2>/dev/null
  if [ "$had_errexit" -eq 1 ]; then
    set -e
  fi
}

hf_upload_enabled() {
  case "$(printf '%s' "$HF_UPLOAD_AFTER_TRAINING" | tr '[:upper:]' '[:lower:]')" in
    0|false|no|off) return 1 ;;
    *) return 0 ;;
  esac
}

upload_to_huggingface() {
  if ! hf_upload_enabled; then
    echo "[datasphere-pipeline] Hugging Face upload disabled: HF_UPLOAD_AFTER_TRAINING=$HF_UPLOAD_AFTER_TRAINING"
    return 0
  fi
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
  echo "[datasphere-pipeline] uploading fine-tuned model and artifacts to Hugging Face: $HF_REPO_ID"
  run_timeout_budgeted "$HF_UPLOAD_TIMEOUT_HOURS" python experiments/vlm_finetuning/scripts/upload_hf_finetuned_artifacts.py \
    --repo-id "$HF_REPO_ID" \
    --repo-type "$HF_REPO_TYPE" \
    --revision "$HF_REVISION" \
    --path-in-repo "$HF_UPLOAD_PATH_PREFIX" \
    --base-model "$BASE_MODEL" \
    --dataset-id "$DATASET_ID" \
    --out-prefix "$OUT_PREFIX" \
    --data-dir "$DATA_DIR" \
    --sft-dir "$SFT_DIR" \
    --grpo-dir "$GRPO_DIR" \
    --report-dir "$REPORT_DIR" \
    --bundle-dir "$HF_UPLOAD_BUNDLE_DIR"
}

trap 'status=$?; package_artifacts "$status"; exit $status' EXIT

if [ "${ENABLE_GPU_PREFLIGHT:-1}" != "0" ]; then
  python experiments/vlm_finetuning/datasphere/bin/check_gpu_before_pipeline.py --report-dir "$REPORT_DIR" --require-bf16
else
  echo "[gpu-check] ENABLE_GPU_PREFLIGHT=0; skipping early CUDA/BF16 preflight"
fi

PIXEL_ARGS=()
if [ -n "$VLM_MIN_PIXELS" ]; then
  PIXEL_ARGS+=(--min-pixels "$VLM_MIN_PIXELS")
fi
if [ -n "$VLM_MAX_PIXELS" ]; then
  PIXEL_ARGS+=(--max-pixels "$VLM_MAX_PIXELS")
fi

DATASET_SAMPLE_ARGS=()
if [ "${MAX_DATASET_SAMPLES:-0}" -gt 0 ]; then
  DATASET_SAMPLE_ARGS+=(--max-samples "$MAX_DATASET_SAMPLES")
fi
if [ "${MAX_SFT_SAMPLES:-0}" -gt 0 ]; then
  DATASET_SAMPLE_ARGS+=(--max-sft-samples "$MAX_SFT_SAMPLES")
fi
if [ "${MAX_GRPO_SAMPLES:-0}" -gt 0 ]; then
  DATASET_SAMPLE_ARGS+=(--max-grpo-samples "$MAX_GRPO_SAMPLES")
fi

run_timeout_budgeted "$DATA_TIMEOUT_HOURS" python experiments/vlm_finetuning/scripts/build_hf_graph_experts_dataset.py \
  --dataset-id "$DATASET_ID" \
  --revision "$DATASET_REVISION" \
  --split "$DATASET_SPLIT" \
  --source-mode "$DATASET_SOURCE_MODE" \
  --export-subdir "$DATASET_EXPORT_SUBDIR" \
  --out-dir "$DATA_DIR" \
  --eval-ratio "$EVAL_RATIO" \
  --max-images-per-example-sft "$MAX_IMAGES_PER_EXAMPLE_SFT" \
  --max-images-per-example-grpo "$MAX_IMAGES_PER_EXAMPLE_GRPO" \
  "${DATASET_SAMPLE_ARGS[@]}" \
  --hf-download-max-workers "$HF_DOWNLOAD_MAX_WORKERS" \
  --seed 42

prefetch_base_model_and_enable_offline_hub

run_torchrun_timeout_budgeted "$SFT_TIMEOUT_HOURS" experiments/vlm_finetuning/scripts/train_vlm_sft.py \
  --model-id "$BASE_MODEL" \
  --train-file "$DATA_DIR/sft_train.jsonl" \
  --eval-file "$DATA_DIR/sft_eval.jsonl" \
  --output-dir "$SFT_DIR" \
  --train-mode vlm \
  --image-column images \
  --use-lora \
  --bf16 \
  --tf32 \
  --gradient-checkpointing \
  "${SFT_DDP_ARGS[@]}" \
  --attn-implementation "$ATTN_IMPLEMENTATION" \
  "${PIXEL_ARGS[@]}" \
  --learning-rate "${SFT_LR:-7e-5}" \
  --warmup-ratio "${SFT_WARMUP_RATIO:-0.05}" \
  --lr-scheduler-type "${SFT_LR_SCHEDULER:-cosine}" \
  --weight-decay "${SFT_WEIGHT_DECAY:-0.01}" \
  --max-grad-norm "${SFT_MAX_GRAD_NORM:-0.3}" \
  --max-text-chars "${SFT_MAX_TEXT_CHARS:-0}" \
  --ddp-timeout-seconds "${SFT_DDP_TIMEOUT_SECONDS:-7200}" \
  --lora-r "${SFT_LORA_R:-32}" \
  --lora-alpha "${SFT_LORA_ALPHA:-64}" \
  --lora-dropout "${SFT_LORA_DROPOUT:-0.05}" \
  --max-steps "$MAX_SFT_STEPS" \
  --num-train-epochs "${SFT_EPOCHS:-4}" \
  --per-device-train-batch-size "${SFT_PER_DEVICE_BATCH:-1}" \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps "${SFT_GRAD_ACCUM:-8}" \
  --save-steps "${SFT_SAVE_STEPS:-60}" \
  --eval-steps "${SFT_EVAL_STEPS:-60}" \
  --logging-steps 5 \
  --dataloader-num-workers "${SFT_DATALOADER_NUM_WORKERS:-4}" \
  --save-adapter-only

tar_adapter_artifact_dir "$SFT_DIR" "outputs/${OUT_PREFIX}_sft_lora.tar.gz"

GRPO_MASK_ARGS=()
case "${GRPO_MASK_TRUNCATED_COMPLETIONS:-0}" in
  1|true|TRUE|yes|YES|on|ON)
    GRPO_MASK_ARGS+=(--mask-truncated-completions)
    ;;
  0|false|FALSE|no|NO|off|OFF)
    echo "[datasphere-pipeline] GRPO_MASK_TRUNCATED_COMPLETIONS=${GRPO_MASK_TRUNCATED_COMPLETIONS}; keeping truncated completions in GRPO loss."
    ;;
  *)
    echo "[datasphere-pipeline] Unsupported GRPO_MASK_TRUNCATED_COMPLETIONS=${GRPO_MASK_TRUNCATED_COMPLETIONS}; expected 0/1/true/false." >&2
    exit 2
    ;;
esac

run_torchrun_timeout_budgeted "$GRPO_TIMEOUT_HOURS" experiments/vlm_finetuning/scripts/train_vlm_grpo.py \
  --model-id "$BASE_MODEL" \
  --sft-adapter-path "$SFT_DIR" \
  --train-file "$DATA_DIR/grpo_train.jsonl" \
  --eval-file "$DATA_DIR/grpo_eval.jsonl" \
  --output-dir "$GRPO_DIR" \
  --train-mode vlm \
  --image-column images \
  --bf16 \
  --tf32 \
  --gradient-checkpointing \
  "${GRPO_DDP_ARGS[@]}" \
  --attn-implementation "$ATTN_IMPLEMENTATION" \
  "${PIXEL_ARGS[@]}" \
  --learning-rate "${GRPO_LR:-1e-5}" \
  --warmup-ratio "${GRPO_WARMUP_RATIO:-0.08}" \
  --lr-scheduler-type "${GRPO_LR_SCHEDULER:-cosine}" \
  --max-grad-norm "${GRPO_MAX_GRAD_NORM:-0.3}" \
  --max-steps "$MAX_GRPO_STEPS" \
  --num-train-epochs "${GRPO_EPOCHS:-1}" \
  --per-device-train-batch-size "${GRPO_PER_DEVICE_BATCH:-1}" \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps "${GRPO_GRAD_ACCUM:-8}" \
  --num-generations "${GRPO_NUM_GENERATIONS:-2}" \
  --num-generations-eval "${GRPO_NUM_GENERATIONS_EVAL:-2}" \
  --max-completion-length "${GRPO_MAX_COMPLETION_LENGTH:-512}" \
  --temperature "${GRPO_TEMPERATURE:-0.8}" \
  --top-p "${GRPO_TOP_P:-0.95}" \
  --top-k "${GRPO_TOP_K:-0}" \
  "${GRPO_MASK_ARGS[@]}" \
  --importance-sampling-level "${GRPO_IMPORTANCE_SAMPLING_LEVEL:-sequence}" \
  --multi-objective-aggregation "${GRPO_MULTI_OBJECTIVE_AGGREGATION:-normalize_then_sum}" \
  --reward-weights ${GRPO_REWARD_WEIGHTS:-0.0 1.0 0.8 1.2 0.5 1.5} \
  --save-steps "${GRPO_SAVE_STEPS:-40}" \
  --eval-steps "${GRPO_EVAL_STEPS:-40}" \
  --logging-steps 5 \
  --dataloader-num-workers "${GRPO_DATALOADER_NUM_WORKERS:-2}" \
  --log-completions

tar_adapter_artifact_dir "$GRPO_DIR" "outputs/${OUT_PREFIX}_grpo_lora.tar.gz"
python - <<PY > "$REPORT_DIR/final_summary.json"
import json
from pathlib import Path
summary = {
    "dataset_id": "$DATASET_ID",
    "dataset_revision": "$DATASET_REVISION",
    "dataset_source_mode": "$DATASET_SOURCE_MODE",
    "dataset_export_subdir": "$DATASET_EXPORT_SUBDIR",
    "base_model": "$BASE_MODEL",
    "sft_dir": "$SFT_DIR",
    "grpo_dir": "$GRPO_DIR",
    "sft_archive": "outputs/${OUT_PREFIX}_sft_lora.tar.gz",
    "grpo_archive": "outputs/${OUT_PREFIX}_grpo_lora.tar.gz",
    "report_archive": "reports/${OUT_PREFIX}_datasphere_reports.tar.gz",
    "budget_plan": json.loads(Path("$REPORT_DIR/budget_plan.json").read_text(encoding="utf-8")),
}
for path in [Path("$DATA_DIR/summary.json"), Path("$SFT_DIR/run_config.json"), Path("$GRPO_DIR/planned_run_config.json")]:
    if path.exists():
        summary[path.stem] = json.loads(path.read_text(encoding="utf-8"))
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

package_artifacts
upload_to_huggingface
package_artifacts
