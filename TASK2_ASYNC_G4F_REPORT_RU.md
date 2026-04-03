# Task 2: async g4f triplet extraction update

## Что изменено

1. Для g4f text-LLM добавлен асинхронный путь через `g4f.client.AsyncClient`.
2. Для g4f JSON extraction добавлены:
   - retry-попытки,
   - экспоненциальный backoff,
   - ограничение числа моделей-кандидатов на один запрос,
   - bounded concurrency через `asyncio.Semaphore`.
3. В `temporal_kg_builder` LLM extraction для g4f теперь запускается batch-режимом по evidence units с ограниченной конкуренцией.
4. Агрессивное отключение text-LLM после первого timeout больше не используется в g4f async-пути.
5. Fallback в co-occurrence теперь срабатывает локально для конкретного evidence unit после исчерпания async retries, а не после первого сбоя всего пайплайна.
6. В notebook добавлены env-defaults:
   - `G4F_ASYNC_ENABLED=1`
   - `G4F_ASYNC_MAX_CONCURRENCY=3`
   - `G4F_ASYNC_RETRIES=3`
   - `G4F_ASYNC_MAX_MODELS_PER_REQUEST=3`
   - `LLM_REQUEST_TIMEOUT_SECONDS=25`

## Изменённые файлы

- `src/scireason/config.py`
- `src/scireason/llm.py`
- `src/scireason/temporal/temporal_triplet_extractor.py`
- `src/scireason/temporal/temporal_kg_builder.py`
- `notebooks/task2_temporal_graph_validation_colab.ipynb`
- `task2_temporal_graph_validation.ipynb`
- `tests/test_task2_pipeline_regressions.py`
- `tests/test_task2_progress_and_timeout.py`
- `tests/test_task2_notebook_full_pipeline_guard.py`
- `tests/test_task2_out_of_box_defaults.py`

## Поведение после фикса

- Triplet extraction сначала пытается пройти через async g4f.
- Одновременно выполняется только ограниченное число запросов, чтобы не перегружать провайдеры.
- На transient failure/timeout запрос повторяется.
- Только если конкретный unit не удалось обработать после retry-цикла, он уходит в локальный co-occurrence fallback.

## Проверка

Запущены релевантные regression / notebook / timeout тесты:

`pytest -q tests/test_task2_pipeline_regressions.py tests/test_task2_progress_and_timeout.py tests/test_task2_notebook_compat.py tests/test_task2_notebook_full_pipeline_guard.py tests/test_task2_out_of_box_defaults.py`

Результат: `27 passed`.
