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

## Адаптация DataSphere CLI

Теперь пакет включает слой DataSphere Jobs / CLI:

- `datasphere/CLI_ADAPTATION_RU.md` — практическое руководство на русском языке по жизненному циклу DataSphere CLI.
- `datasphere/job_configs/*.yaml` — готовые конфигурации заданий (jobs) для сборки датасета, smoke/pilot запусков SFT, пилотного DPO и валидации.
- `datasphere/bin/*.sh` — обертки среды выполнения заданий (runtime wrappers).
- `datasphere/launch_examples.sh` — вспомогательный лаунчер для команд `datasphere project job execute/list/get/attach/cancel/set-data-ttl/download-files`.

Пример:

```bash
export DATASPHERE_PROJECT_ID=<project_id>
bash experiments/vlm_finetuning/datasphere/launch_examples.sh build-datasets
bash experiments/vlm_finetuning/datasphere/launch_examples.sh sft-smoke
bash experiments/vlm_finetuning/datasphere/launch_examples.sh list
```
