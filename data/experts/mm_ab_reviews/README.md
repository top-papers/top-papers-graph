# Task 3 multimodal hard-subset A/B review templates

Эта папка содержит шаблоны для слепого A/B теста baseline VLM vs SFT/DPO VLM в Task 3.

Что здесь лежит:
- `task3_mm_ab_hard_subset_manifest.template.json` — шаблон manifest для фиксированного hard subset.
- `task3_mm_ab_review_schema.json` — схема результата экспертного ревью.
- `task3_mm_ab_review_template.json` — пустой шаблон JSON с ответами эксперта.
- `task3_mm_ab_pair_notes.csv` — CSV-шаблон для быстрых заметок по парам.
- `task3_mm_ab_expert_checklist.md` — чеклист подготовки эксперта.

Рекомендуемый порядок:
1. Владелец прогона собирает fixed hard subset manifest.
2. Генерирует blind bundle.
3. Передаёт эксперту HTML + manifest + эти шаблоны.
4. Эксперт ведёт заметки в CSV и сдаёт итоговый JSON.
