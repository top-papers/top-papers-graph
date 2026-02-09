# Task 2 — Graph Verification (проверка графа знаний) — temporal + multimodal

## Цель
Убрать ложные связи и сделать связи применимыми (с указанием условий/времени).

## Что вы ревьюите
Список assertions (утверждений), каждое с evidence и временем/условиями.

## Вердикты
- accepted
- rejected
- needs_time_fix
- needs_evidence_fix
- added

## Особенности
- Если нет явной причинности — используйте корреляционные предикаты.
- Без условий связь часто бесполезна: фиксируйте T/SOC/chemistry.
- Evidence может быть в фигуре/таблице: укажите страницу и номер.

## Шаблон
См. `templates/graph_review_template.json`.
