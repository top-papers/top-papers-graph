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
