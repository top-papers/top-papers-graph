# Исправление smoke-прогона DataSphere: выравнивание image placeholders и images

Дата: 2026-06-15

## Симптом

Последний smoke-прогон дошёл дальше предыдущих исправлений:

- DataSphere job создалась и выполнилась на `g2.2`.
- GPU preflight прошёл успешно: `torch==2.5.1+cu121`, CUDA доступна, 2 GPU A100, BF16 поддерживается.
- Dataset smoke caps применились: `MAX_SFT_SAMPLES=96`, `MAX_GRPO_SAMPLES=48`, `HF_DOWNLOAD_MAX_WORKERS=1`.
- SFT rows были успешно нормализованы и прошли проверку Qwen3-VL chat-template safety.
- `SFTTrainer` создался и обучение дошло до первого batch.

Новая ошибка появилась уже внутри TRL VLM collator:

```text
ValueError: Number of images provided (3) does not match number of image placeholders (5).
ValueError: Number of images provided (3) does not match number of image placeholders (8).
```

## Причина

TRL `SFTTrainer` для VLM вызывает `prepare_multimodal_messages(example["messages"], images=example["images"])`. Эта функция требует, чтобы количество top-level изображений в `images` строго совпадало с количеством блоков `{"type": "image"}` в `messages`.

В smoke-режиме export builder ограничивает число реально переданных изображений, например до 3 на SFT example, но исходные `messages` некоторых строк могут содержать 5 или 8 image placeholders. До этого исправления sanitizer проверял безопасность типов блоков для Qwen3-VL chat template, но не проверял равенство:

```text
len(images) == count({"type": "image"} in messages)
```

## Исправления

### `experiments/vlm_finetuning/scripts/train_vlm_sft.py`

Добавлено:

- `_count_image_placeholders(messages)`
- `_align_image_placeholders_to_images(messages, image_count)`
- `_row_images_for_placeholder_alignment(example)`

Теперь SFT sanitizer перед `SFTTrainer` делает следующее:

1. нормализует `messages` в Qwen/TRL-safe blocks;
2. считает фактические top-level images;
3. удаляет лишние `{"type":"image"}` placeholders сверх `len(images)`;
4. добавляет недостающие placeholders, если images больше, чем placeholders;
5. validation guard падает до DataLoader, если равенство снова нарушится.

### `experiments/vlm_finetuning/scripts/train_vlm_grpo.py`

Аналогичная alignment-функция добавлена для GRPO prompt formatter, чтобы та же ошибка не появилась на следующем этапе после успешного SFT smoke.

### Tests

В `tests/test_vlm_training_format_normalization.py` добавлены regression tests:

- `test_sft_aligns_image_placeholders_to_capped_images`
- `test_sft_adds_missing_placeholders_for_images`
- `test_grpo_aligns_image_placeholders_to_capped_images`

## Проверки

Локально выполнено:

```text
[OK] py_compile:
     train_vlm_sft.py
     train_vlm_grpo.py
     build_hf_graph_experts_dataset.py
     run_full_pipeline.py
     check_gpu_before_pipeline.py

[OK] bash -n:
     launch_examples.sh
     datasphere/bin/*.sh

[OK] requirements.txt:
     all non-empty lines pass packaging.Requirement

[OK] YAML:
     all job_configs/*.yaml load via pyyaml
     cu121 extra-index-url is present for requirements-file configs

[OK] pytest:
     tests/test_vlm_training_format_normalization.py
     tests/test_hf_export_smoke_sampling.py
     tests/test_datasphere_job_configs.py

[OK] smoke dry-run:
     run_full_pipeline.py builds the expected datasphere project job execute command
```

## Следующая проверка

```bash
cd top-papers-graph-main
source .venv/bin/activate

export DATASPHERE_PROJECT_ID='bt18pnosk97i8n24ddnv'

datasphere project get --id "$DATASPHERE_PROJECT_ID"

bash experiments/vlm_finetuning/datasphere/launch_examples.sh hf-smoke-managed
```

Ожидаемый прогресс после этого исправления:

```text
[gpu-check] OK
sample_limited_asset_download: true
[train_vlm_sft] validated ... rows for Qwen3-VL chat template safety
SFTTrainer created
first train batch starts without "Number of images provided ... does not match number of image placeholders"
```
