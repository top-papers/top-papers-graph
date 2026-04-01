# Task 2 — Валидация темпорального графа знаний

## Что делает эксперт
Эксперт загружает YAML из **Task 1** и получает **два графа**:

1. **Reference graph** — точное восстановление вручную размеченной reasoning-схемы из YAML.
2. **Automatic temporal KG** — автоматически построенный темпоральный граф по статьям из YAML через CLI `top-papers-graph prepare-task2-validation`.

Дальше эксперт сравнивает оба графа, просматривает триплеты, оценивает временные поля и при необходимости вносит правки в `graph_review_prefill.json` и `temporal_corrections_template.json`.

## Temporal-схема
В Task 2 используется **двойная темпоральность**:

- `start_date` — начало интервала, к которому относится утверждение / наблюдение.
- `end_date` — конец интервала, обычно дата публикации или последний год, подтверждённый источником.
- `valid_from` — момент, с которого утверждение считаем валидным в графе.
- `valid_to` — момент, до которого утверждение считается валидным; если опровержения нет, допустимо `+inf`.
- `time_source` — откуда взято время: `explicit_text`, `figure_or_table`, `metadata`, `expert_inference`, `unknown`.
- `time_source_note` — свободный комментарий эксперта.

Поддерживаются специальные значения: `unknown`, `-inf`, `+inf`.
Поле `time_interval` сохраняется только ради обратной совместимости с существующим пайплайном и формируется автоматически.

Для `time_granularity` поддерживаются значения `year`, `month`, `day`, `interval`; старый вариант `range` автоматически нормализуется в `interval`.

## Как интерпретировать время
Рекомендуемый приоритет:

1. **explicit_text** — время явно указано в тексте статьи.
2. **figure_or_table** — время извлечено из подписи, оси, таблицы.
3. **metadata** — время взято из года публикации / внешних метаданных.
4. **expert_inference** — время реконструировано экспертом, если другого источника нет.
5. **unknown** — если время определить нельзя.

## Вердикты эксперта
- `accepted`
- `rejected`
- `needs_time_fix`
- `needs_evidence_fix`
- `added`

## Дополнительный scouting
CLI также создаёт `scout/suggested_links.json` — список дополнительных статей-кандидатов, найденных по `topic` и `next_question` из trajectory. Это отдельный слой поддержки эксперта: ссылки предлагает система, а решение об их использовании принимает эксперт.

## Выбор LLM для автоматического графа
Для Task 2 теперь можно явно выбрать LLM-контур при запуске bundle:

- `--g4f-model <model>` — использовать конкретную модель из установленного реестра g4f.
- `--local-model <name>` — использовать локальную модель Ollama по имени.
- `--llm-provider ... --llm-model ...` — универсальный ручной override.

Примеры:

```bash
top-papers-graph prepare-task2-validation \
  --trajectory runs/task1/example.yaml \
  --out-dir runs/task2_validation \
  --g4f-model deepseek-r1

top-papers-graph prepare-task2-validation \
  --trajectory runs/task1/example.yaml \
  --out-dir runs/task2_validation \
  --local-model llama3.2
```

В notebook `task2_temporal_graph_validation_colab.ipynb` эти же настройки доступны через отдельные поля `LLM`, `g4f model` и `Local model`.

## Где смотреть результаты
После запуска `prepare-task2-validation` в bundle появляются:

- `reference_graph.json`
- `reference_triplets.json`
- `automatic_graph/temporal_kg.json`
- `automatic_triplets.json`
- `comparison_summary.json`
- `review_templates/graph_review_prefill.json`
- `review_templates/temporal_corrections_template.json`
- `scout/suggested_links.json`


## Обновление Task 2 v2

Теперь эксперт во втором задании валидирует не только верность ребра, но и его пригодность для downstream-конвейера генерации проверяемых гипотез. Рекомендуется заполнять дополнительные поля: `semantic_correctness`, `evidence_sufficiency`, `scope_match`, `hypothesis_role`, `hypothesis_relevance`, `testability_signal`, `causal_status`, `severity`, `evidence_before_cutoff`, `leakage_risk`, `time_type`, `time_granularity`, `time_confidence`, а для визуальных evidence — `mm_verdict` и `mm_rationale`.
