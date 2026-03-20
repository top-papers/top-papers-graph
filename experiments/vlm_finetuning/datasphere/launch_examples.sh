#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 {build-datasets|sft-smoke|sft-pilot|dpo-pilot|teacher-sft-30b|student-distill-4b|validate|list|get|attach|cancel|ttl|download} [args...]"
  exit 1
fi

ACTION="$1"
shift || true
PROJECT_ID="${DATASPHERE_PROJECT_ID:-}"
need_project() {
  if [ -z "$PROJECT_ID" ]; then
    echo "Set DATASPHERE_PROJECT_ID before running this command"
    exit 2
  fi
}

case "$ACTION" in
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
    datasphere project job get --id "$1"
    ;;
  attach)
    datasphere project job attach --id "$1"
    ;;
  cancel)
    datasphere project job cancel --id "$1"
    ;;
  ttl)
    datasphere project job set-data-ttl --id "$1" --days "$2"
    ;;
  download)
    datasphere project job download-files --id "$1"
    ;;
  *)
    echo "Unknown action: $ACTION"
    exit 1
    ;;
esac
