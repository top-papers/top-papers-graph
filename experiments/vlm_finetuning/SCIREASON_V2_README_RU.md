# Рекомендуемый v2-пайплайн дообучения SciReason

Этот пайплайн заменяет прямой `SFT -> GRPO` запуск на более устойчивую схему:

```text
text-only SFT -> multimodal SFT -> DPO -> optional short GRPO
```

## Почему так

- `assistant_only_loss` корректно используется в text-only SFT.
- Multimodal SFT получает только evidence-bearing изображения, выбранные по релевантности.
- DPO лучше соответствует экспертным правкам и chosen/rejected парам, чем преждевременный GRPO.
- GRPO включается только после reward audit.

## Быстрая сборка датасетов

```bash
python experiments/vlm_finetuning/scripts/build_scireason_alignment_datasets.py \
  --dataset-id top-papers/top-papers-graph-experts-data \
  --export-subdir exports/colab-run-001 \
  --out-dir data/derived/hf_top_papers_scireason_v2
```

## DataSphere

```bash
python experiments/vlm_finetuning/datasphere/run_full_pipeline.py \
  --project-id "$DATASPHERE_PROJECT_ID" \
  --config experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_dpo_grpo_v2_g2_2.yaml
```

По умолчанию `ENABLE_GRPO_POLISH=0`. Включайте GRPO только после просмотра:

```text
data/derived/hf_top_papers_scireason_v2/reward_audit_by_task_family.json
```

## Главные outputs

```text
data/derived/hf_top_papers_scireason_v2/summary.json
data/derived/hf_top_papers_scireason_v2/leakage_report.json
data/derived/hf_top_papers_scireason_v2/image_resolution_report.json
data/derived/hf_top_papers_scireason_v2/reward_audit_by_task_family.json
outputs/hf_top_papers_qwen3vl_8b_v2_text_sft_lora.tar.gz
outputs/hf_top_papers_qwen3vl_8b_v2_vlm_sft_lora.tar.gz
outputs/hf_top_papers_qwen3vl_8b_v2_dpo_lora.tar.gz
```
