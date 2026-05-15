# Адаптация VLM fine-tuning pipeline под Qwen/Qwen3-VL-8B-Instruct

Дата ревизии: 2026-05-15.

## Цель

Перевести основной DataSphere full pipeline для VLM дообучения с `Qwen/Qwen2.5-VL-7B-Instruct` на `Qwen/Qwen3-VL-8B-Instruct`, сохранив существующий процесс:

1. сборка HF dataset `top-papers/top-papers-graph-experts-data` в локальные multimodal JSONL;
2. LoRA SFT на SFT JSONL;
3. GRPO/RL поверх SFT LoRA adapter;
4. упаковка adapters, logs, reports и manifest.

## Что изменено

### Основной full pipeline

- `experiments/vlm_finetuning/datasphere/bin/run_hf_top_papers_sft_grpo_full.sh`
  - default `BASE_MODEL` заменён на `Qwen/Qwen3-VL-8B-Instruct`;
  - default `OUT_PREFIX` заменён на `hf_top_papers_qwen3vl_8b`, чтобы outputs не смешивались со старыми Qwen2.5-VL артефактами.

- `experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_full_g2_2.yaml`
  - job name/description уточнены под Qwen3-VL-8B;
  - env vars и declared outputs переведены на `Qwen/Qwen3-VL-8B-Instruct` и `hf_top_papers_qwen3vl_8b`.

### Конфиги и experiment matrix

- Добавлен `experiments/vlm_finetuning/configs/sft_grpo_full_qwen3vl_8b_lora.yaml` — явный сценарий полного SFT→GRPO запуска для Qwen3-VL-8B.
- `experiments/vlm_finetuning/results/experiment_matrix.csv` дополнен строкой `sft_grpo_full_qwen3vl_8b_lora`.

### Документация

Обновлены инструкции и ожидаемые имена output-файлов:

- `experiments/vlm_finetuning/datasphere/HF_TOP_PAPERS_FULL_PIPELINE_RU.md`
- `experiments/vlm_finetuning/datasphere/TUTORIAL_FULL_EXPERIMENT_RU.md`
- `experiments/vlm_finetuning/README.md`
- `DATASPHERE_VLM_FULL_PIPELINE_PATCH_REPORT_RU.md`

## Совместимость training entrypoints

Скрипты `train_vlm_sft.py`, `train_vlm_dpo.py` и `train_vlm_grpo.py` уже содержали ветку загрузки `Qwen3VLForConditionalGeneration` для model id с `Qwen3-VL`. Поэтому для перехода на 8B не потребовалось переписывать нормализацию данных: текущий формат `messages` + top-level `image`/`images` сохранён.

Критичные guardrails оставлены без изменений:

- `remove_unused_columns=False`;
- `max_length=None` для VLM stages;
- `bf16`;
- `gradient_checkpointing`;
- batch size 1;
- LoRA/PEFT вместо full fine-tuning.

## Основной запуск

```bash
export DATASPHERE_PROJECT_ID=<project_id>
bash experiments/vlm_finetuning/datasphere/launch_examples.sh hf-full-managed
```

Ожидаемые главные artifacts:

```text
outputs/hf_top_papers_qwen3vl_8b_sft_lora.tar.gz
outputs/hf_top_papers_qwen3vl_8b_grpo_lora.tar.gz
reports/hf_top_papers_qwen3vl_8b_datasphere_reports.tar.gz
reports/hf_top_papers_qwen3vl_8b_datasphere/final_summary.json
```

## Проверки, выполненные в локальной среде

- `python -m py_compile` для VLM training scripts и DataSphere launcher;
- `bash -n` для DataSphere shell wrappers;
- YAML parse для DataSphere job configs и VLM configs;
- regression tests для VLM training format normalization и DataSphere job configs.

Полный GPU/DataSphere training run в этой среде не выполнялся: здесь нет доступа к DataSphere и GPU.
