#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 {hf-full|hf-full-managed|build-datasets|sft-smoke|sft-pilot|dpo-pilot|teacher-sft-30b|student-distill-4b|validate|list|get|attach|cancel|ttl|download} [args...]"
  exit 1
fi

ACTION="$1"
shift || true
PROJECT_ID="${DATASPHERE_PROJECT_ID:-}"
need_project() {
  if [ -z "$PROJECT_ID" ]; then
    echo "Set DATASPHERE_PROJECT_ID before running this command" >&2
    exit 2
  fi
}

need_args() {
  local expected="$1"
  shift
  if [ "$#" -lt "$expected" ]; then
    echo "Action '$ACTION' expects at least $expected argument(s)" >&2
    exit 2
  fi
}

case "$ACTION" in
  hf-full)
    need_project
    datasphere project job execute -p "$PROJECT_ID" -c experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_full_g2_2.yaml
    ;;
  hf-full-managed)
    need_project
    python experiments/vlm_finetuning/datasphere/run_full_pipeline.py --project-id "$PROJECT_ID" "$@"
    ;;
  build-datasets)
    need_project
    datasphere project job execute -p "$PROJECT_ID" -c experiments/vlm_finetuning/datasphere/job_configs/build_datasets.yaml
    ;;
  sft-smoke)
    need_project
    datasphere project job execute -p "$PROJECT_ID" -c experiments/vlm_finetuning/datasphere/job_configs/sft_smoke.yaml
    ;;
  sft-pilot)
    need_project
    datasphere project job execute -p "$PROJECT_ID" -c experiments/vlm_finetuning/datasphere/job_configs/sft_pilot.yaml
    ;;
  dpo-pilot)
    need_project
    datasphere project job execute -p "$PROJECT_ID" -c experiments/vlm_finetuning/datasphere/job_configs/dpo_pilot.yaml
    ;;
  teacher-sft-30b)
    need_project
    datasphere project job execute -p "$PROJECT_ID" -c experiments/vlm_finetuning/datasphere/job_configs/teacher_sft_qwen3vl_30b_a3b.yaml
    ;;
  student-distill-4b)
    need_project
    datasphere project job execute -p "$PROJECT_ID" -c experiments/vlm_finetuning/datasphere/job_configs/student_distill_qwen3vl_4b.yaml
    ;;
  validate)
    need_project
    datasphere project job execute -p "$PROJECT_ID" -c experiments/vlm_finetuning/datasphere/job_configs/validate_extraction.yaml
    ;;
  list)
    need_project
    datasphere project job list -p "$PROJECT_ID"
    ;;
  get)
    need_args 1 "$@"
    datasphere project job get --id "$1"
    ;;
  attach)
    need_args 1 "$@"
    datasphere project job attach --id "$1"
    ;;
  cancel)
    need_args 1 "$@"
    datasphere project job cancel --id "$1"
    ;;
  ttl)
    need_args 2 "$@"
    datasphere project job set-data-ttl --id "$1" --days "$2"
    ;;
  download)
    need_args 1 "$@"
    datasphere project job download-files --id "$1"
    ;;
  *)
    echo "Unknown action: $ACTION"
    exit 1
    ;;
esac
