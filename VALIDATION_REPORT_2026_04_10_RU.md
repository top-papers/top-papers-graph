# Итоговый отчёт по прогону и исправлениям (2026-04-10)

## Что было дополнительно исправлено

### 1. Исправлен офлайн-баг в Task 3 pipeline

`prepare_task3_hypothesis_bundle(...)` больше не уходит в сетевой поиск статей, если уже передан `processed_dir`.
Это важно для детерминированных офлайн-прогонов notebook и CI smoke-validation.

Файл:
- `src/scireason/pipeline/task3_hypothesis_generation.py`

### 2. Исправлен приоритет корня репозитория в notebook

Task 2 / Task 3 notebook теперь сначала смотрят на `TPG_REPO_DIR`, а уже потом пытаются автоматически подхватывать старые распакованные архивы из `/mnt/data`.
Это устраняет ситуацию, когда notebook запускался не на текущем исправленном репозитории, а на более старой копии.

Исправлено в:
- `task2_temporal_graph_validation.ipynb`
- `notebooks/task2_temporal_graph_validation_colab.ipynb`
- `task3_multimodal_temporal_hypothesis_generation.ipynb`
- `notebooks/task3_multimodal_temporal_hypothesis_generation_colab.ipynb`
- `task3_dual_local_models_blind_ab.ipynb`
- `notebooks/task3_dual_local_models_blind_ab_colab.ipynb`

### 3. Добавлен headless smoke mode для Task 3 notebook

Обе версии Task 3 notebook (обычная и dual-local blind A/B, включая Colab-копии) теперь умеют:
- автоматически заполнять форму из переменных окружения;
- запускаться без ручного клика по widget UI;
- работать в офлайн-safe режиме с mock/text route, `hash` embeddings и отключённым реальным VLM.

Это сделано для надёжной автоматической проверки notebook через `nbclient`.

### 4. Сделан более безопасный editable install для smoke-run

В smoke-режиме notebook не пытаются тянуть полный `.[task3]` dependency stack, а используют editable install без зависимостей.
Это снижает риск зависания/падения на optional-зависимостях вроде `annoy`/build backend в средах Python 3.13.

### 5. Улучшена совместимость `pyproject.toml` с Python 3.13

- ослаблено ограничение на `numpy` до `<3`;
- `annoy` в extra `task3` вынесен под маркер `python_version < "3.13"`;
- при отсутствии `annoy` pipeline штатно использует `numpy_fallback`.

### 6. Сохранена совместимость проверок Task 2 notebook

Порядок ячеек в Task 2 notebook и Colab-копии приведён к ожидаемому тестами контракту:
- install/import cell;
- helper cell;
- form cell;
- отдельная run cell.

## Что именно прогнано

### Полный pytest по репозиторию

Лог:
- `reports/final_pytest_run_2026_04_10.log`

Результат:
- **85 passed, 3 skipped, 14 warnings in 28.97s**

### Полный smoke-run всех Task 3 notebook

Task 3 notebook были выполнены **по одной**, в изолированных `nbclient`-прогонах, чтобы избежать конфликтов между kernel lifecycle и widget state.

Проверены все четыре варианта:
- `task3_multimodal_temporal_hypothesis_generation.ipynb`
- `notebooks/task3_multimodal_temporal_hypothesis_generation_colab.ipynb`
- `task3_dual_local_models_blind_ab.ipynb`
- `notebooks/task3_dual_local_models_blind_ab_colab.ipynb`

Для smoke-run использовались:
- synthetic `processed_papers/`;
- `LLM_PROVIDER=mock`, `LLM_MODEL=mock`;
- `EMBED_PROVIDER=hash`;
- `MM_EMBED_BACKEND=none`;
- `VLM_BACKEND=none`;
- `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`.

Проверено, что создаются артефакты:
- `task3_manifest.json`
- `hypotheses_ranked.json`
- offline A/B HTML
- expert ZIP
- для dual-local blind A/B также `owner key` и два отдельных bundle (`variant_alpha`, `variant_beta`)

Детальный summary:
- `reports/validation_2026_04_10/repo_validation_summary.md`
- `reports/validation_2026_04_10/repo_validation_summary.json`

## Ограничения проверки

Smoke-validation подтверждает:
- корректность control flow notebook;
- сохранение артефактов;
- работу blind A/B packaging;
- работу офлайн-контура на synthetic input.

Smoke-validation **не** подтверждает качество реальных локальных VL/VLM-весов и не является бенчмарком научного качества гипотез для конкретной модели.
