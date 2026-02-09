# Артефакты и форматы данных

Идея простая: **всё, что создают люди**, должно быть:
1) машинно-проверяемо (валидатор),
2) версионируемо (git),
3) пригодно для обучения/оценки.

## 1) Reasoning Trajectories (“золотые стандарты рассуждений”)
Папка: `data/experts/trajectories/`  
Шаблон: `data/experts/trajectories/_template.yaml`

**Что это:** цепочка “факт → вывод → следующий вопрос” по известному открытию *на материалах статей ДО открытия*.

**Минимальные требования:**
- 5–10 ключевых статей
- 6–15 шагов reasoning
- в каждом шаге: цитата (или точный фрагмент) + интерпретация + логика вывода

## 2) Graph Reviews (верификация графов)
Папка: `data/experts/graph_reviews/`  
Шаблон: `data/experts/graph_reviews/_template.json`

**Что это:** эксперт подтверждает/отвергает связи и добавляет “имплицитные” связи, которые очевидны специалисту.

**Минимальные требования:**
- минимум 10 ребер просмотрено
- минимум 3 отклонённых ребра с причинами
- минимум 1 добавленное ребро (если релевантно)

## 3) Hypothesis Reviews (Red Teaming)
Папка: `data/experts/hypothesis_reviews/`  
Шаблон: `data/experts/hypothesis_reviews/_template.json`

**Что это:** “рецензия Reviewer #2” на гипотезу системы по трём осям: новизна, обоснованность, проверяемость.

**Минимальные требования:**
- выставлены оценки (0–5)
- минимум 1 major issue или явное “accept”
- чёткая рекомендация: reject / revise / accept

## 4) Автоматическая валидация артефактов
PR с данными должен проходить:

- `pytest -q`
- `python scripts/ci/validate_expert_artifacts.py` (проверяет YAML/JSON структуру)

Валидация охватывает: trajectories, graph_reviews, hypothesis_reviews, **mm_reviews**, **temporal_corrections**.

## 5) MM Reviews (исправление мультимодальности)
Папка: `data/experts/mm_reviews/`  
Шаблон: `data/experts/mm_reviews/_template.json`

**Что это:** эксперт подтверждает/исправляет подпись страницы (и/или таблицы/формулы), извлечённые VL-моделью.

Используется для:
- улучшения VLM prompting/post-processing
- обучения лёгкой модели коррекции (caption/table clean-up)

## 6) Temporal corrections (исправление времени)
Папка: `data/experts/temporal_corrections/`  
Шаблон: `data/experts/temporal_corrections/_template.json`

**Что это:** эксперт исправляет неверную временную привязку утверждения в Temporal KG.

## 7) Экспорт в training datasets
Скрипт:

```bash
python scripts/data/export_training_datasets.py
```

Результаты пишутся в `data/derived/training/` (JSONL с полем `messages`).

---


## Новые артефакты (Temporal + Multimodal)

### Мультимодальность (MM)
- `data/processed/papers/<paper_id>/mm/pages.jsonl` — записи по страницам (текст + путь к PNG + vlm_caption + tables_md).
- `data/processed/papers/<paper_id>/mm/images/page_XXX.png` — рендер страниц.

### Темпоральные утверждения
- Neo4j: ноды `:Assertion` и `:Time` (интервалы) + связи `SUBJECT/OBJECT/ASSERTED_IN/AT_TIME`.
- Экспорт (опционально): `exports/assertions_<date>.jsonl` — можно добавить как следующий шаг.
