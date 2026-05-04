# Исправление потери входных файлов в датасете

## Что было найдено

Приложенный файл `Avshalumov_trajectory_submission_Александр_Авшалумов.yaml` корректно парсится как Task 1 submission и при прямом `export-scidatapipe` даёт 10 строк `trajectory_reasoning`. Значит ошибка была не в содержимом этого YAML, а в пакетной сборке датасета и контроле сохранности входов.

Основная причина: несколько файлов из форм/офлайн-загрузок могли иметь одинаковый внутренний `submission_id`, например стандартный `trajectory_submission`. Скрипты для VLM SFT/DPO строили `id` записей только из этого внутреннего идентификатора и номера шага. В результате разные входные файлы создавали одинаковые record id вида `trajectory:trajectory_submission:1`, `trajectory:trajectory_submission:2` и т.д. Такие записи могли схлопываться или перезаписываться downstream-инструментами, из-за чего казалось, что часть входных файлов не попала в датасет.

Дополнительная причина в Colab notebook: режим выбора строк из CSV по умолчанию оставлял последнюю отправку на участника. Для сценария, где каждая строка формы — отдельный входной файл, это выглядело как потеря части файлов.

## Что исправлено

1. `experiments/vlm_finetuning/scripts/build_vlm_sft_dataset.py`
   - Добавлен стабильный `input_fingerprint` по содержимому файла и пути.
   - `submission_id` внутри датасета теперь получает суффикс `__input_<hash>`.
   - Оригинальный `submission_id` сохраняется в `metadata.original_submission_id`.
   - В `metadata` добавлены `input_fingerprint` и `input_file`.
   - Добавлена жёсткая проверка уникальности всех record id.
   - В summary добавлены `input_files_with_records`, `input_file_count_with_records`, `unique_record_ids`.

2. `experiments/vlm_finetuning/scripts/build_vlm_preference_dataset.py`
   - Аналогично добавлен collision-safe source id для pairwise/DPO/GRPO записей.
   - Добавлена проверка уникальности pair id.
   - В summary добавлены счётчики файлов-источников и уникальных id.

3. `src/scireason/scidatapipe_bridge/builder.py`
   - Добавлен контроль сохранности нормализованных Task 1 источников.
   - `export_summary.json` теперь содержит:
     - `normalized_task1_source_files`
     - `normalized_task1_source_file_count`
     - `task1_sources_with_sft_rows`
     - `task1_sources_without_sft_rows`
   - Это позволяет сразу увидеть, какой нормализованный входной файл не дал SFT-строк.

4. `top_papers_graph_scidatapipe_hf_colab_from_csv_only_fixed_gdown_scope_assets_fixed.ipynb`
   - Добавлен параметр `RETAIN_ALL_FORM_ROWS = True`.
   - По умолчанию notebook теперь сохраняет все строки формы как отдельные входы.
   - Старое поведение «оставить только последнюю строку на участника» доступно через `RETAIN_ALL_FORM_ROWS = False`.
   - Добавлена post-export проверка: если какой-либо нормализованный Task 1 источник не дал SFT-строк, notebook останавливается с ошибкой.
   - Убран захардкоженный Hugging Face token из параметров.

5. `tests/test_vlm_dataset_build_input_retention.py`
   - Добавлены регрессионные тесты на ситуацию с двумя YAML-файлами, у которых одинаковый `submission_id`.
   - Проверяется, что оба файла дают отдельные строки, уникальные id и сохраняют исходный файл в metadata.

## Проверки

Запуск статической проверки Python-файлов:

```bash
python -m py_compile \
  src/scireason/scidatapipe_bridge/builder.py \
  experiments/vlm_finetuning/scripts/build_vlm_sft_dataset.py \
  experiments/vlm_finetuning/scripts/build_vlm_preference_dataset.py \
  experiments/vlm_finetuning/scripts/train_vlm_sft.py \
  experiments/vlm_finetuning/scripts/train_vlm_grpo.py \
  experiments/vlm_finetuning/scripts/train_vlm_dpo.py
```

Запуск тестов:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q \
  tests/test_vlm_dataset_build_input_retention.py \
  tests/test_scidatapipe_input_retention.py \
  tests/test_scidatapipe_bridge.py \
  tests/test_vlm_training_format_normalization.py
```

Результат:

```text
12 passed
```

Проверка приложенного файла:

```json
{
  "discovered_task1_files": 1,
  "normalized_task1_submissions": 1,
  "normalized_task1_source_file_count": 1,
  "task1_sources_with_sft_rows": 1,
  "task1_sources_without_sft_rows": [],
  "sft_rows": 10
}
```

Проверка двух файлов с одинаковым `submission_id`:

```json
{
  "duplicate_input_rows": 20,
  "unique_ids": 20,
  "input_file_count_with_records": 2,
  "unique_record_ids": 20
}
```
