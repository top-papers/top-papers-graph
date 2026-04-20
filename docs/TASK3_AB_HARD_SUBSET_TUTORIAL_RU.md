# Task 3: как провести A/B тест так, чтобы разница baseline VLM vs SFT/DPO VLM проявилась

## Коротко
Нельзя делать primary metric по случайным финальным hypotheses. Для Task 3 primary metric должна мерить **multimodal evidence extraction + temporal grounding** на **fixed hard subset**.

## Пошаговый план

### Шаг 1. Подготовить вход эксперта
Эксперт должен подготовить ZIP/YAML с темой, cutoff year и идентификаторами статей. Если hard subset уже собран заранее, лучше передать `processed_dir` с этими статьями, а не полагаться на live search.

### Шаг 2. Собрать fixed hard subset manifest
Создайте JSON по шаблону `data/experts/mm_ab_reviews/task3_mm_ab_hard_subset_manifest.template.json`.
Включайте только такие единицы:
- figure/table/formula-heavy страницы;
- temporal-hard случаи;
- небольшую долю easy controls.

### Шаг 3. Зафиксировать всё, кроме VLM
Должны совпадать:
- список статей;
- processed papers / OCR / chunks;
- text model;
- retrieval;
- candidate generation;
- rendering.
Меняться должны только VLM weights / adapters.

### Шаг 4. Запустить dual blind A/B
Для Kaggle notebook используйте:
- `processed_dir` с curated hard subset;
- `top_pairs >= 16`;
- `top_hypotheses >= 16`;
- blind bundle для эксперта.

### Шаг 5. Передать эксперту blind bundle
Передавайте только:
- offline HTML;
- public manifest;
- шаблоны из `data/experts/mm_ab_reviews/`.
Не передавайте owner key.

### Шаг 6. Что делает эксперт
Эксперт сравнивает варианты A/B в таком порядке:
1. evidence correctness;
2. temporal correctness;
3. hallucination / overclaiming;
4. testability;
5. только потом общая предпочтительность.

### Шаг 7. Аггрегация
Primary endpoint:
- win-rate tuned VLM на `multimodal_hard + temporal_hard`.

Secondary endpoints:
- `better_evidence`;
- `better_temporal`;
- error tags: `missed_visual_fact`, `wrong_evidence_linkage`, `hallucinated_visual_inference`, `needs_time_fix`.

## Практический минимум по размерам
- 20–30 paired examples на пилот.
- 60–120 paired examples на полноценный прогон.
- 2–3 эксперта, 20–30% overlap.

## Что нельзя делать
- сравнивать случайные лёгкие статьи как основной тест;
- допускать rank fallback как основной pairing rule;
- давать эксперту только prose без акцента на evidence/time.
