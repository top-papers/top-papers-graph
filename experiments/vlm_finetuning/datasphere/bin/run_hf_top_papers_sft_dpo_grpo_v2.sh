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
mkdir -p "$REPORT_DIR" outputs data/derived

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
    "grpo_config": Path("$GRPO_DIR/planned_run_config.json"),
}
summary = {
    "pipeline": "scireason_v2_text_sft_vlm_sft_dpo_optional_grpo",
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
  --attn-implementation "${ATTN_IMPLEMENTATION:-auto}" \
  --max-pixels "${VLM_MAX_PIXELS:-1003520}" \
  --learning-rate "${VLM_SFT_LR:-3e-5}" \
  --warmup-ratio "${VLM_SFT_WARMUP_RATIO:-0.05}" \
  --lr-scheduler-type "${VLM_SFT_LR_SCHEDULER:-cosine}" \
  --weight-decay "${VLM_SFT_WEIGHT_DECAY:-0.01}" \
  --max-grad-norm "${VLM_SFT_MAX_GRAD_NORM:-0.3}" \
  --max-text-chars "${SFT_MAX_TEXT_CHARS:-12000}" \
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
torchrun_stage experiments/vlm_finetuning/scripts/train_vlm_dpo.py \
  --model-id "$BASE_MODEL" \
  --sft-adapter-path "$VLM_SFT_DIR" \
  --train-file "$DATA_DIR/dpo_train.jsonl" \
  --eval-file "$DATA_DIR/dpo_eval.jsonl" \
  --output-dir "$DPO_DIR" \
  --train-mode vlm \
  --image-column images \
  --bf16 --gradient-checkpointing \
  --attn-implementation "${ATTN_IMPLEMENTATION:-auto}" \
  --learning-rate "${DPO_LR:-7e-6}" \
  --beta "${DPO_BETA:-0.08}" \
  --max-steps "${DPO_MAX_STEPS:--1}" \
  --num-train-epochs "${DPO_EPOCHS:-1}" \
  --per-device-train-batch-size "${DPO_PER_DEVICE_BATCH:-1}" \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps "${DPO_GRAD_ACCUM:-8}" \
  --save-steps "${DPO_SAVE_STEPS:-60}" \
  --eval-steps "${DPO_EVAL_STEPS:-60}" \
  --logging-steps 5

tar_adapter_dir "$DPO_DIR" "outputs/${OUT_PREFIX}_dpo_lora.tar.gz"

# 4) Optional short GRPO polish. Disabled unless ENABLE_GRPO_POLISH=1.
if [ "${ENABLE_GRPO_POLISH:-0}" = "1" ]; then
  torchrun_stage experiments/vlm_finetuning/scripts/train_vlm_grpo.py \
    --model-id "$BASE_MODEL" \
    --sft-adapter-path "$DPO_DIR" \
    --train-file "$DATA_DIR/grpo_train_verified.jsonl" \
    --eval-file "$DATA_DIR/grpo_eval_verified.jsonl" \
    --output-dir "$GRPO_DIR" \
    --train-mode vlm \
    --image-column images \
    --bf16 --tf32 --gradient-checkpointing \
    --attn-implementation "${ATTN_IMPLEMENTATION:-auto}" \
    --max-pixels "${VLM_MAX_PIXELS:-1003520}" \
    --learning-rate "${GRPO_LR:-5e-6}" \
    --warmup-ratio "${GRPO_WARMUP_RATIO:-0.05}" \
    --beta "${GRPO_BETA:-0.02}" \
    --lr-scheduler-type "${GRPO_LR_SCHEDULER:-cosine}" \
    --max-grad-norm "${GRPO_MAX_GRAD_NORM:-0.3}" \
    --max-steps "${GRPO_MAX_STEPS:--1}" \
    --num-train-epochs "${GRPO_EPOCHS:-1}" \
    --per-device-train-batch-size "${GRPO_PER_DEVICE_BATCH:-1}" \
    --per-device-eval-batch-size 1 \
    --gradient-accumulation-steps "${GRPO_GRAD_ACCUM:-8}" \
    --num-generations "${GRPO_NUM_GENERATIONS:-4}" \
    --num-generations-eval "${GRPO_NUM_GENERATIONS_EVAL:-2}" \
    --max-completion-length "${GRPO_MAX_COMPLETION_LENGTH:-768}" \
    --temperature "${GRPO_TEMPERATURE:-0.8}" \
    --top-p "${GRPO_TOP_P:-0.95}" \
    --top-k "${GRPO_TOP_K:-0}" \
    --mask-truncated-completions \
    --importance-sampling-level "${GRPO_IMPORTANCE_SAMPLING_LEVEL:-sequence}" \
    --multi-objective-aggregation "${GRPO_MULTI_OBJECTIVE_AGGREGATION:-normalize_then_sum}" \
    --reward-weights ${GRPO_REWARD_WEIGHTS:-0.0 0.5 0.8 1.6 1.2 1.5} \
    --save-steps "${GRPO_SAVE_STEPS:-40}" \
    --eval-steps "${GRPO_EVAL_STEPS:-40}" \
    --logging-steps 5 \
    --log-completions \
    --fail-on-weak-reward
  tar_adapter_dir "$GRPO_DIR" "outputs/${OUT_PREFIX}_grpo_lora.tar.gz"
else
  mkdir -p "$GRPO_DIR"
  printf '%s\n' '{"status":"skipped","reason":"ENABLE_GRPO_POLISH is not 1; use DPO checkpoint as the main candidate."}' > "$GRPO_DIR/planned_run_config.json"
fi

write_stage_summary
tar --exclude="$REPORT_DIR/hf_upload_bundle" -czf "reports/${OUT_PREFIX}_datasphere_reports.tar.gz" "$REPORT_DIR" "$DATA_DIR" 2>/dev/null || true
