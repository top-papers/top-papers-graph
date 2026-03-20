<<<<<<< HEAD
# VLM fine-tuning bundle for temporal scientific KG extraction

This bundle extends the repository with a practical plan for fine-tuning a vision-language model
on expert artifacts so that the model extracts higher quality structured evidence from scientific
papers for temporal knowledge graph (TKG) construction and downstream hypothesis generation.

## What is included

- `DESIGN_RU.md` — full training-system design in Russian.
- `EXPERIMENT_PLAN_RU.md` — validation plan, ablations, and go/no-go criteria.
- `configs/` — starter configs for SFT, DPO, and optional GRPO.
- `schemas/` — JSON schemas for multimodal SFT and preference data.
- `scripts/build_vlm_sft_dataset.py` — compile expert trajectories/MM reviews into VLM SFT JSONL.
- `scripts/build_vlm_preference_dataset.py` — compile expert reviews/corrections into VLM preference JSONL.
- `scripts/estimate_datasphere_costs.py` — scenario cost calculator for Yandex DataSphere.
- `scripts/validate_extraction_run.py` — offline evaluator for extraction/TKG runs.
- `datasphere/requirements.txt` — practical package set for first experiments.
- `datasphere/launch_examples.sh` — example launch commands.

## Recommended training order

1. Baseline inference on the current extractor + current VLM backend.
2. SFT on expert trajectories + multimodal review data.
3. DPO on expert corrections and review-derived preferences.
4. Optional narrow GRPO only for reward-verifiable subproblems.

## Fast start
=======
# Пакет дообучения VLM для извлечения темпоральных научных графов знаний

Этот пакет расширяет репозиторий практическим планом дообучения (fine-tuning) визуально-языковой модели (VLM) на экспертных артефактах. Это позволит модели извлекать более качественные структурированные данные из научных статей для построения темпоральных графов знаний (TKG) и последующей генерации гипотез.

## Что включено

- `DESIGN_RU.md` — полный дизайн системы обучения на русском языке.
- `EXPERIMENT_PLAN_RU.md` — план валидации, абляционные исследования и критерии принятия решений (go/no-go).
- `configs/` — стартовые конфигурации для SFT, DPO и опционального GRPO.
- `schemas/` — JSON-схемы для мультимодального SFT и данных предпочтений.
- `scripts/build_vlm_sft_dataset.py` — компиляция экспертных траекторий/мультимодальных (MM) ревью в VLM SFT JSONL.
- `scripts/build_vlm_preference_dataset.py` — компиляция экспертных ревью/исправлений в VLM preference JSONL.
- `scripts/estimate_datasphere_costs.py` — калькулятор стоимости сценариев для Yandex DataSphere.
- `scripts/validate_extraction_run.py` — офлайн-оценщик для запусков извлечения/TKG.
- `datasphere/requirements.txt` — практический набор пакетов для первых экспериментов.
- `datasphere/launch_examples.sh` — примеры команд запуска.

## Рекомендуемый порядок обучения

1. Базовый инференс (baseline) на текущем экстракторе + текущем VLM бэкенде.
2. SFT на экспертных траекториях + мультимодальных данных ревью.
3. DPO на экспертных исправлениях и предпочтениях, полученных из ревью.
4. Опциональный узкий GRPO только для подзадач, проверяемых через функцию вознаграждения (reward-verifiable).

## Быстрый старт
>>>>>>> 2cb7cf18c35667ddd62ccd23515c33934455aaf4

```bash
python experiments/vlm_finetuning/scripts/build_vlm_sft_dataset.py \
  --repo-root . \
  --out data/derived/training/vlm_sft.jsonl

python experiments/vlm_finetuning/scripts/build_vlm_preference_dataset.py \
  --repo-root . \
  --out data/derived/training/vlm_dpo.jsonl

python experiments/vlm_finetuning/scripts/estimate_datasphere_costs.py \
  --scenario experiments/vlm_finetuning/configs/sft_pilot_qwen3vl_4b_lora.yaml
```

<<<<<<< HEAD

## DataSphere CLI adaptation

The bundle now includes a DataSphere Jobs / CLI layer:

- `datasphere/CLI_ADAPTATION_RU.md` — practical Russian guide for DataSphere CLI lifecycle.
- `datasphere/job_configs/*.yaml` — ready job configs for dataset build, SFT smoke/pilot, DPO pilot, and validation.
- `datasphere/bin/*.sh` — job runtime wrappers.
- `datasphere/launch_examples.sh` — helper launcher around `datasphere project job execute/list/get/attach/cancel/set-data-ttl/download-files`.

Example:
=======
## Адаптация DataSphere CLI

Теперь пакет включает слой DataSphere Jobs / CLI:

- `datasphere/CLI_ADAPTATION_RU.md` — практическое руководство на русском языке по жизненному циклу DataSphere CLI.
- `datasphere/job_configs/*.yaml` — готовые конфигурации заданий (jobs) для сборки датасета, smoke/pilot запусков SFT, пилотного DPO и валидации.
- `datasphere/bin/*.sh` — обертки среды выполнения заданий (runtime wrappers).
- `datasphere/launch_examples.sh` — вспомогательный лаунчер для команд `datasphere project job execute/list/get/attach/cancel/set-data-ttl/download-files`.

Пример:
>>>>>>> 2cb7cf18c35667ddd62ccd23515c33934455aaf4

```bash
export DATASPHERE_PROJECT_ID=<project_id>
bash experiments/vlm_finetuning/datasphere/launch_examples.sh build-datasets
bash experiments/vlm_finetuning/datasphere/launch_examples.sh sft-smoke
bash experiments/vlm_finetuning/datasphere/launch_examples.sh list
```
