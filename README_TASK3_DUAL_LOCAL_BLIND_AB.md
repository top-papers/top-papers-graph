# Task 3 — Dual Local Models Blind A/B

Этот сценарий предназначен для сравнения двух локальных модельных конфигураций на одном и том же Task 3 pipeline:

- **Model α** — обычно baseline / модель из коробки;
- **Model β** — обычно дообученная локальная модель.

## Что делает notebook

`task3_dual_local_models_blind_ab.ipynb`

1. Принимает вход из `query`, `Task 1 YAML`, `commands` или `processed_papers.zip`.
2. Запускает `prepare_task3_hypothesis_bundle(...)` **два раза**.
3. Для второго прогона использует те же `processed_papers`, чтобы сравнение было максимально честным.
4. Генерирует blind offline HTML для эксперта.
5. Сохраняет owner-only key file с соответствием анонимных систем и реальных моделей.
6. Собирает expert ZIP **без owner key**.


## Headless smoke mode for blind A/B validation

Both dual-model notebooks now support a non-interactive smoke mode for deterministic validation runs.

Environment variables:
- `TPG_REPO_DIR` — explicit repository root; preferred over older unpacked archives
- `TASK3_DUAL_NOTEBOOK_SMOKE=1`
- `TASK3_DUAL_NOTEBOOK_SMOKE_PROCESSED_DIR=/abs/path/to/processed_papers`
- `TASK3_DUAL_NOTEBOOK_SMOKE_OUT_DIR=/abs/path/to/output`
- `TASK3_DUAL_NOTEBOOK_SMOKE_MODEL_A=/path/or/id/of/base-model`
- `TASK3_DUAL_NOTEBOOK_SMOKE_MODEL_B=/path/or/id/of/finetuned-model`

In smoke mode the notebook anonymizes the two local configurations, runs both variants on the same processed input, and still builds the blind offline HTML, owner key and expert ZIP.

## Основные артефакты

- `task3_dual_local_model_review_offline_ab.html`
- `expert_dual_model_blind_review_bundle.zip`
- `task3_dual_local_model_blind_key.json` (**не передавать эксперту**)
- два отдельных Task 3 bundle (`variant_alpha/...` и `variant_beta/...`)

## Python helper

Для blind review добавлен модуль:

- `src/scireason/task3_dual_model_review.py`

В нём есть две основные функции:

- `build_task3_dual_model_offline_review_package(...)`
- `build_task3_dual_model_expert_bundle(...)`

## Тест

```bash
pytest -q tests/test_task3_dual_model_review.py
```
