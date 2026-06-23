#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# DataSphere secret bridge: project secret HFTOKEN -> HF_TOKEN.
if [ -z "${HF_TOKEN:-}" ] && [ -n "${HFTOKEN:-}" ]; then export HF_TOKEN="$HFTOKEN"; fi
if [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ] && [ -n "${HF_TOKEN:-}" ]; then export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"; fi
export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export PYTHONUTF8="${PYTHONUTF8:-1}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export DISABLE_TRL_MODEL_CARD="${DISABLE_TRL_MODEL_CARD:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-$PYTORCH_CUDA_ALLOC_CONF}"

DATASET_ID="${HF_DATASET_ID:-top-papers/top-papers-graph-experts-data}"
DATASET_REVISION="${HF_DATASET_REVISION:-main}"
DATASET_EXPORT_SUBDIR="${HF_DATASET_EXPORT_SUBDIR:-exports/colab-run-001}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
OUT_PREFIX="${OUT_PREFIX:-hf_top_papers_qwen3vl_8b_v2}"
DATA_DIR="${DATA_DIR:-data/derived/hf_top_papers_scireason_v2}"
TEXT_SFT_DIR="outputs/${OUT_PREFIX}_text_sft_lora"
VLM_SFT_DIR="outputs/${OUT_PREFIX}_vlm_sft_lora"
DPO_DIR="outputs/${OUT_PREFIX}_dpo_lora"
GRPO_DIR="outputs/${OUT_PREFIX}_grpo_lora"
REPORT_DIR="reports/${OUT_PREFIX}_datasphere"
HF_REPO_ID="${HF_REPO_ID:-top-papers/Qwen3-VL-8B-Instruct-scireason}"
HF_REPO_TYPE="${HF_REPO_TYPE:-model}"
HF_REVISION="${HF_REVISION:-main}"
HF_UPLOAD_AFTER_TRAINING="${HF_UPLOAD_AFTER_TRAINING:-0}"
HF_UPLOAD_PATH_PREFIX="${HF_UPLOAD_PATH_PREFIX:-}"
HF_UPLOAD_BUNDLE_DIR="${HF_UPLOAD_BUNDLE_DIR:-$REPORT_DIR/hf_upload_bundle}"
mkdir -p "$REPORT_DIR" outputs data/derived
# G2.2 memory-safe training projection. Raw data/audit files still preserve all
# rows and all image refs when MAX_IMAGES_PER_EXAMPLE_*=0.
SFT_TRAIN_MAX_IMAGES_PER_EXAMPLE="${SFT_TRAIN_MAX_IMAGES_PER_EXAMPLE:-3}"
DPO_TRAIN_MAX_IMAGES_PER_EXAMPLE="${DPO_TRAIN_MAX_IMAGES_PER_EXAMPLE:-3}"
GRPO_TRAIN_MAX_IMAGES_PER_EXAMPLE="${GRPO_TRAIN_MAX_IMAGES_PER_EXAMPLE:-2}"
DDP_UNUSED_ARGS=()
if [ "${DDP_FIND_UNUSED_PARAMETERS:-1}" = "1" ]; then
  DDP_UNUSED_ARGS+=(--ddp-find-unused-parameters)
else
  DDP_UNUSED_ARGS+=(--no-ddp-find-unused-parameters)
fi

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

NPROC="$(python_gpu_count)"
if [ -z "$NPROC" ] || [ "$NPROC" -lt 1 ]; then NPROC=1; fi

torchrun_stage() {
  torchrun --standalone --nproc_per_node="$NPROC" "$@"
}

tar_adapter_dir() {
  local src="$1" dst="$2"
  mkdir -p "$(dirname "$dst")"
  find "$src" -maxdepth 1 -type d -name 'checkpoint-*' -exec rm -rf {} + 2>/dev/null || true
  tar --exclude='*/checkpoint-*' --exclude='*/optimizer.pt' --exclude='*/scheduler.pt' --exclude='*/rng_state*.pth' --exclude='*/scaler.pt' -czf "$dst" "$src"
}

write_stage_summary() {
  python - <<PY > "$REPORT_DIR/final_summary.json"
import json
from pathlib import Path
paths = {
    "dataset_summary": Path("$DATA_DIR/summary.json"),
    "leakage_report": Path("$DATA_DIR/leakage_report.json"),
    "image_resolution_report": Path("$DATA_DIR/image_resolution_report.json"),
    "reward_audit": Path("$DATA_DIR/reward_audit_by_task_family.json"),
    "full_data_usage_report": Path("$DATA_DIR/full_data_usage_report.json"),
    "text_sft_config": Path("$TEXT_SFT_DIR/run_config.json"),
    "vlm_sft_config": Path("$VLM_SFT_DIR/run_config.json"),
    "dpo_config": Path("$DPO_DIR/run_config.json"),
    "grpo_config": Path("$GRPO_DIR/run_config.json") if Path("$GRPO_DIR/run_config.json").exists() else Path("$GRPO_DIR/planned_run_config.json"),
}
summary = {
    "pipeline": "scireason_v2_text_sft_vlm_sft_dpo_grpo",
    "dataset_id": "$DATASET_ID",
    "dataset_revision": "$DATASET_REVISION",
    "dataset_export_subdir": "$DATASET_EXPORT_SUBDIR",
    "base_model": "$BASE_MODEL",
    "outputs": {
        "text_sft_dir": "$TEXT_SFT_DIR",
        "vlm_sft_dir": "$VLM_SFT_DIR",
        "dpo_dir": "$DPO_DIR",
        "grpo_dir": "$GRPO_DIR",
        "report_dir": "$REPORT_DIR",
        "dpo_archive": "outputs/${OUT_PREFIX}_dpo_lora.tar.gz",
        "grpo_archive": "outputs/${OUT_PREFIX}_grpo_lora.tar.gz",
        "hf_upload_bundle_dir": "$HF_UPLOAD_BUNDLE_DIR",
    },
}
for key, path in paths.items():
    if path.exists():
        try:
            summary[key] = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            summary[key] = {"path": str(path), "error": str(exc)}
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
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
  echo "[datasphere-pipeline] uploading DPO-first artifacts to Hugging Face: $HF_REPO_ID"
  python experiments/vlm_finetuning/scripts/upload_hf_finetuned_artifacts.py \
    --repo-id "$HF_REPO_ID" \
    --repo-type "$HF_REPO_TYPE" \
    --revision "$HF_REVISION" \
    --path-in-repo "$HF_UPLOAD_PATH_PREFIX" \
    --base-model "$BASE_MODEL" \
    --dataset-id "$DATASET_ID" \
    --out-prefix "$OUT_PREFIX" \
    --data-dir "$DATA_DIR" \
    --sft-dir "$VLM_SFT_DIR" \
    --dpo-dir "$DPO_DIR" \
    --grpo-dir "$GRPO_DIR" \
    --final-stage "${HF_FINAL_STAGE:-auto}" \
    --report-dir "$REPORT_DIR" \
    --bundle-dir "$HF_UPLOAD_BUNDLE_DIR"
}

trap 'status=$?; write_stage_summary || true; exit $status' EXIT

python experiments/vlm_finetuning/scripts/build_scireason_alignment_datasets.py \
  --dataset-id "$DATASET_ID" \
  --revision "$DATASET_REVISION" \
  --export-subdir "$DATASET_EXPORT_SUBDIR" \
  --out-dir "$DATA_DIR" \
  --eval-ratio "${EVAL_RATIO:-0.10}" \
  --max-images-per-example-sft "${MAX_IMAGES_PER_EXAMPLE_SFT:-0}" \
  --max-images-per-example-grpo "${MAX_IMAGES_PER_EXAMPLE_GRPO:-0}" \
  --max-sft-samples "${MAX_SFT_SAMPLES:-0}" \
  --max-grpo-samples "${MAX_GRPO_SAMPLES:-0}" \
  --max-dpo-pairs-per-row "${MAX_DPO_PAIRS_PER_ROW:-3}" \
  --hf-download-max-workers "${HF_DOWNLOAD_MAX_WORKERS:-2}" \
  --seed "${SEED:-42}"

python experiments/vlm_finetuning/scripts/audit_alignment_readiness.py \
  --data-dir "$DATA_DIR" \
  --out "$DATA_DIR/alignment_readiness_report.json" \
  --min-dpo-train-rows "${MIN_DPO_TRAIN_ROWS:-100}" \
  --min-sft-vlm-train-rows "${MIN_SFT_VLM_TRAIN_ROWS:-100}" \
  --min-grpo-verified-rows "${MIN_GRPO_VERIFIED_ROWS:-50}" \
  ${QUALITY_GATE_STRICT:+--strict}

FULL_DATA_AUDIT_ARGS=()
if [ "${FULL_DATA_STRICT:-1}" = "1" ]; then FULL_DATA_AUDIT_ARGS+=(--strict); fi
if [ "${REQUIRE_ALL_IMAGES:-1}" = "1" ]; then FULL_DATA_AUDIT_ARGS+=(--require-all-images); fi
python experiments/vlm_finetuning/scripts/audit_full_data_usage.py \
  --data-dir "$DATA_DIR" \
  --out "$DATA_DIR/full_data_usage_report.json" \
  "${FULL_DATA_AUDIT_ARGS[@]}"

prefetch_base_model_and_enable_offline_hub

# 1) Text-only SFT: this is the stage where assistant_only_loss is effective.
torchrun_stage experiments/vlm_finetuning/scripts/train_vlm_sft.py \
  --model-id "$BASE_MODEL" \
  --train-file "$DATA_DIR/sft_text_train.jsonl" \
  --eval-file "$DATA_DIR/sft_text_eval.jsonl" \
  --output-dir "$TEXT_SFT_DIR" \
  --train-mode text \
  --use-lora \
  --assistant-only-loss \
  --bf16 --tf32 --gradient-checkpointing \
  "${DDP_UNUSED_ARGS[@]}" \
  --attn-implementation "${ATTN_IMPLEMENTATION:-auto}" \
  --learning-rate "${TEXT_SFT_LR:-5e-5}" \
  --warmup-ratio "${TEXT_SFT_WARMUP_RATIO:-0.05}" \
  --lr-scheduler-type "${TEXT_SFT_LR_SCHEDULER:-cosine}" \
  --weight-decay "${TEXT_SFT_WEIGHT_DECAY:-0.01}" \
  --max-grad-norm "${TEXT_SFT_MAX_GRAD_NORM:-0.3}" \
  --lora-r "${LORA_R:-32}" \
  --lora-alpha "${LORA_ALPHA:-64}" \
  --lora-dropout "${LORA_DROPOUT:-0.05}" \
  --max-steps "${TEXT_SFT_MAX_STEPS:--1}" \
  --num-train-epochs "${TEXT_SFT_EPOCHS:-3}" \
  --per-device-train-batch-size "${SFT_PER_DEVICE_BATCH:-1}" \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps "${SFT_GRAD_ACCUM:-8}" \
  --save-steps "${SFT_SAVE_STEPS:-60}" \
  --eval-steps "${SFT_EVAL_STEPS:-60}" \
  --logging-steps 5 \
  --dataloader-num-workers "${SFT_DATALOADER_NUM_WORKERS:-0}" \
  --save-adapter-only

tar_adapter_dir "$TEXT_SFT_DIR" "outputs/${OUT_PREFIX}_text_sft_lora.tar.gz"

# 2) Multimodal SFT continues from the text adapter and focuses on evidence images.
torchrun_stage experiments/vlm_finetuning/scripts/train_vlm_sft.py \
  --model-id "$BASE_MODEL" \
  --init-adapter-path "$TEXT_SFT_DIR" \
  --train-file "$DATA_DIR/sft_vlm_train.jsonl" \
  --eval-file "$DATA_DIR/sft_vlm_eval.jsonl" \
  --output-dir "$VLM_SFT_DIR" \
  --train-mode vlm \
  --image-column images \
  --bf16 --tf32 --gradient-checkpointing \
  "${DDP_UNUSED_ARGS[@]}" \
  --attn-implementation "${ATTN_IMPLEMENTATION:-auto}" \
  --max-pixels "${VLM_MAX_PIXELS:-1003520}" \
  --learning-rate "${VLM_SFT_LR:-3e-5}" \
  --warmup-ratio "${VLM_SFT_WARMUP_RATIO:-0.05}" \
  --lr-scheduler-type "${VLM_SFT_LR_SCHEDULER:-cosine}" \
  --weight-decay "${VLM_SFT_WEIGHT_DECAY:-0.01}" \
  --max-grad-norm "${VLM_SFT_MAX_GRAD_NORM:-0.3}" \
  --max-text-chars "${SFT_MAX_TEXT_CHARS:-12000}" \
  --max-images-per-example "$SFT_TRAIN_MAX_IMAGES_PER_EXAMPLE" \
  --max-steps "${VLM_SFT_MAX_STEPS:--1}" \
  --num-train-epochs "${VLM_SFT_EPOCHS:-2}" \
  --per-device-train-batch-size "${SFT_PER_DEVICE_BATCH:-1}" \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps "${SFT_GRAD_ACCUM:-8}" \
  --save-steps "${SFT_SAVE_STEPS:-60}" \
  --eval-steps "${SFT_EVAL_STEPS:-60}" \
  --logging-steps 5 \
  --dataloader-num-workers "${SFT_DATALOADER_NUM_WORKERS:-0}" \
  --save-adapter-only

tar_adapter_dir "$VLM_SFT_DIR" "outputs/${OUT_PREFIX}_vlm_sft_lora.tar.gz"

# 3) DPO is the default alignment stage. It trains on chosen/rejected pairs.
DPO_LOSS_ARGS=()
read -r -a DPO_LOSS_TYPES_ARR <<< "${DPO_LOSS_TYPES:-${DPO_LOSS_TYPE:-robust sft}}"
DPO_LOSS_ARGS+=(--loss-type "${DPO_LOSS_TYPES_ARR[@]}")
if [ -n "${DPO_LOSS_WEIGHTS:-1.0 0.15}" ]; then
  read -r -a DPO_LOSS_WEIGHTS_ARR <<< "${DPO_LOSS_WEIGHTS:-1.0 0.15}"
  DPO_LOSS_ARGS+=(--loss-weights "${DPO_LOSS_WEIGHTS_ARR[@]}")
fi
if [ "${DPO_USE_WEIGHTING:-1}" = "1" ]; then DPO_LOSS_ARGS+=(--use-weighting); else DPO_LOSS_ARGS+=(--no-use-weighting); fi
if [ "${DPO_PRECOMPUTE_REF_LOG_PROBS:-0}" = "1" ]; then
  DPO_LOSS_ARGS+=(--precompute-ref-log-probs)
  if [ -n "${DPO_PRECOMPUTE_REF_BATCH_SIZE:-2}" ]; then DPO_LOSS_ARGS+=(--precompute-ref-batch-size "${DPO_PRECOMPUTE_REF_BATCH_SIZE:-2}"); fi
fi
torchrun_stage experiments/vlm_finetuning/scripts/train_vlm_dpo.py \
  --model-id "$BASE_MODEL" \
  --sft-adapter-path "$VLM_SFT_DIR" \
  --train-file "$DATA_DIR/dpo_train.jsonl" \
  --eval-file "$DATA_DIR/dpo_eval.jsonl" \
  --output-dir "$DPO_DIR" \
  --train-mode vlm \
  --image-column images \
  --max-pixels "${VLM_MAX_PIXELS:-1003520}" \
  --max-images-per-example "$DPO_TRAIN_MAX_IMAGES_PER_EXAMPLE" \
  --bf16 --gradient-checkpointing \
  "${DDP_UNUSED_ARGS[@]}" \
  --attn-implementation "${ATTN_IMPLEMENTATION:-auto}" \
  --learning-rate "${DPO_LR:-7e-6}" \
  --beta "${DPO_BETA:-0.06}" \
  "${DPO_LOSS_ARGS[@]}" \
  --label-smoothing "${DPO_LABEL_SMOOTHING:-0.05}" \
  --max-steps "${DPO_MAX_STEPS:--1}" \
  --num-train-epochs "${DPO_EPOCHS:-1}" \
  --per-device-train-batch-size "${DPO_PER_DEVICE_BATCH:-1}" \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps "${DPO_GRAD_ACCUM:-8}" \
  --save-steps "${DPO_SAVE_STEPS:-60}" \
  --eval-steps "${DPO_EVAL_STEPS:-60}" \
  --logging-steps 5

tar_adapter_dir "$DPO_DIR" "outputs/${OUT_PREFIX}_dpo_lora.tar.gz"

# 4) GRPO polish. Enabled by default for the full SFT->DPO->GRPO scheme; set ENABLE_GRPO_POLISH=0 only for DPO-only ablations.
if [ "${ENABLE_GRPO_POLISH:-1}" = "1" ]; then
  torchrun_stage experiments/vlm_finetuning/scripts/train_vlm_grpo.py \
    --model-id "$BASE_MODEL" \
    --sft-adapter-path "$DPO_DIR" \
    --train-file "$DATA_DIR/grpo_train_verified.jsonl" \
    --eval-file "$DATA_DIR/grpo_eval_verified.jsonl" \
    --output-dir "$GRPO_DIR" \
    --train-mode vlm \
    --image-column images \
    --bf16 --tf32 --gradient-checkpointing \
    "${DDP_UNUSED_ARGS[@]}" \
    --attn-implementation "${ATTN_IMPLEMENTATION:-auto}" \
    --max-pixels "${VLM_MAX_PIXELS:-1003520}" \
    --max-images-per-example "$GRPO_TRAIN_MAX_IMAGES_PER_EXAMPLE" \
    --learning-rate "${GRPO_LR:-2e-6}" \
    --warmup-ratio "${GRPO_WARMUP_RATIO:-0.05}" \
    --beta "${GRPO_BETA:-0.005}" \
    --lr-scheduler-type "${GRPO_LR_SCHEDULER:-cosine}" \
    --max-grad-norm "${GRPO_MAX_GRAD_NORM:-0.3}" \
    --max-steps "${GRPO_MAX_STEPS:--1}" \
    --num-train-epochs "${GRPO_EPOCHS:-1}" \
    --per-device-train-batch-size "${GRPO_PER_DEVICE_BATCH:-1}" \
    --per-device-eval-batch-size 1 \
    --gradient-accumulation-steps "${GRPO_GRAD_ACCUM:-8}" \
    --num-generations "${GRPO_NUM_GENERATIONS:-4}" \
    --num-generations-eval "${GRPO_NUM_GENERATIONS_EVAL:-4}" \
    --num-iterations "${GRPO_NUM_ITERATIONS:-2}" \
    --max-completion-length "${GRPO_MAX_COMPLETION_LENGTH:-640}" \
    --temperature "${GRPO_TEMPERATURE:-0.85}" \
    --top-p "${GRPO_TOP_P:-0.95}" \
    --top-k "${GRPO_TOP_K:-0}" \
    --epsilon "${GRPO_EPSILON:-0.2}" \
    --epsilon-high "${GRPO_EPSILON_HIGH:-0.28}" \
    --top-entropy-quantile "${GRPO_TOP_ENTROPY_QUANTILE:-0.2}" \
    --mask-truncated-completions \
    --importance-sampling-level "${GRPO_IMPORTANCE_SAMPLING_LEVEL:-sequence}" \
    --multi-objective-aggregation "${GRPO_MULTI_OBJECTIVE_AGGREGATION:-normalize_then_sum}" \
    --reward-weights ${GRPO_REWARD_WEIGHTS:-0.0 0.15 0.35 0.0 0.55 1.0} \
    --save-steps "${GRPO_SAVE_STEPS:-40}" \
    --eval-steps "${GRPO_EVAL_STEPS:-40}" \
    --logging-steps 5 \
    --min-reward-std "${GRPO_MIN_REWARD_STD:-0.08}" \
    --max-zero-std-frac "${GRPO_MAX_ZERO_STD_FRAC:-0.60}" \
    --log-completions \
    --fail-on-weak-reward
  tar_adapter_dir "$GRPO_DIR" "outputs/${OUT_PREFIX}_grpo_lora.tar.gz"
else
  mkdir -p "$GRPO_DIR"
  printf '%s\n' '{"status":"skipped","reason":"ENABLE_GRPO_POLISH is not 1; use DPO checkpoint as the main candidate."}' > "$GRPO_DIR/planned_run_config.json"
fi

write_stage_summary
tar --exclude="$REPORT_DIR/hf_upload_bundle" -czf "reports/${OUT_PREFIX}_datasphere_reports.tar.gz" "$REPORT_DIR" "$DATA_DIR" 2>/dev/null || true
upload_to_huggingface
write_stage_summary
tar --exclude="$REPORT_DIR/hf_upload_bundle" -czf "reports/${OUT_PREFIX}_datasphere_reports.tar.gz" "$REPORT_DIR" "$DATA_DIR" 2>/dev/null || true
