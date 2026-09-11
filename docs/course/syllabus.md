<!-- SPDX-FileCopyrightText: 2026 top-papers-graph contributors -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# Программа курса «Анализ данных в научной литературе»

**Формат:** 12 недель, проектный. Каждый участник вносит вклад в общий
открытый инструмент `top-papers-graph`, а не выполняет изолированные
домашние работы.

**Объём:** 1 общий синк в неделю (60 минут) + асинхронная работа через
issue/pull request.

**Предварительные требования:** базовый Python; понимание, что такое
векторное представление и метрика качества. Опыт ML не обязателен —
роли «Исследователь» и «Координатор» не требуют программирования.

**Результат курса:** merged pull request, принятый экспертный артефакт
или воспроизводимый notebook, оставшийся полезным после окончания потока.

---

## Как пользоваться программой

Эта страница — точка входа. Она связывает недельные модули с заданиями,
экспертными материалами и ноутбуками, которые раньше были не связаны
между собой.

- **Недельный модуль** — тема, результат недели, практика, вклад по ролям.
- **Задание** — подробная инструкция к конкретному артефакту.
- **Экспертный трек** — углублённая работа для тех, кто берёт роль
  «Эксперт»: разметка, ревью, контроль качества.
- **Ноутбук** — готовый запускаемый пример.

---

## Календарь на 12 недель

| № | Тема | Результат недели | Материалы |
|---:|---|---|---|
| 1 | Вход в проект и воспроизводимое демо | Запущенное офлайн-демо, первый цикл issue → PR | [модуль](../../course/weeks/week01.md) · [путь контрибьютора](../../course/CONTRIBUTOR_GUIDE.md) · [повестка](experts/agendas/week01_agenda.md) |
| 2 | PDF → текст → чанки | Проверяемый пример качества ingestion | [модуль](../../course/weeks/week02.md) · [повестка](experts/agendas/week02_agenda.md) |
| 3 | Термины и доменные словари | Словарь и сравнение извлекателей | [модуль](../../course/weeks/week03.md) · [повестка](experts/agendas/week03_agenda.md) |
| 4 | Связи: co-occurrence и LLM-triplets | Проверенные связи, фильтр шума | [модуль](../../course/weeks/week04.md) · [повестка](experts/agendas/week04_agenda.md) |
| 5 | Время в графе знаний | Временная коррекция и regression case | [модуль](../../course/weeks/week05.md) · [повестка](experts/agendas/week05_agenda.md) |
| 6 | Графовые алгоритмы и объяснение | Анализ сообществ, объяснимый пример | [модуль](../../course/weeks/week06.md) · [повестка](experts/agendas/week06_agenda.md) |
| 7 | Link prediction как источник гипотез | 3–5 кандидатов с falsification plan | [модуль](../../course/weeks/week07.md) · [повестка](experts/agendas/week07_agenda.md) |
| 8 | Векторные baseline и абляции | Baseline, абляция, сравнение ранжирования | [модуль](../../course/weeks/week08.md) · [повестка](experts/agendas/week08_agenda.md) |
| 9 | Code agents и безопасные инструменты | Новый tool, sandbox rule | [модуль](../../course/weeks/week09.md) · [повестка](experts/agendas/week09_agenda.md) |
| 10 | Human-in-the-loop и data flywheel | Gold-набор или data pipeline | [модуль](../../course/weeks/week10.md) · [повестка](experts/agendas/week10_agenda.md) |
| 11 | GNN/TGNN и честная оценка | Holdout, метрики, честная проверка | [модуль](../../course/weeks/week11.md) · [повестка](experts/agendas/week11_agenda.md) |
| 12 | Интеграция и публичный результат | Итоговый PR, demo, dataset или release note | [модуль](../../course/weeks/week12.md) · [повестка](experts/agendas/week12_agenda.md) |

Подробный план-график с ролями и Definition of Done:
[00_master_plan_12weeks.md](experts/00_master_plan_12weeks.md).

---

## Задания

Задания не привязаны жёстко к одной неделе: их берут тогда, когда неделя
подводит к нужному навыку. Полный список —
[индекс заданий](tasks/README.md).

| Задание | Когда брать | Инструкция |
|---|---|---|
| Task 2: проверка темпорального графа | недели 5, 10 | [task2_validation.md](tasks/task2_validation.md) |
| Task 3: генерация гипотез | недели 7, 11 | [task3_hypotheses.md](tasks/task3_hypotheses.md) |
| Task 3: dual-local blind A/B | недели 8, 11 | [task3_dual_local_blind_ab.md](tasks/task3_dual_local_blind_ab.md) |
| Task 3: публикация benchmark на HF | неделя 12 | [task3_hf_benchmark.md](tasks/task3_hf_benchmark.md) |
| Case-based A/B: обзор сценария | недели 10–12 | [task3_ab_case_based_overview.md](tasks/task3_ab_case_based_overview.md) |
| Case-based A/B: руководство участника | недели 10–12 | [task3_ab_participant_guide.md](tasks/task3_ab_participant_guide.md) |
| Case-based A/B: руководство автора | по желанию | [task3_ab_creator_guide.md](tasks/task3_ab_creator_guide.md) |

---

## Экспертный трек

Экспертный трек — углублённая работа для роли «Эксперт»: проверка
научного смысла утверждений, разметка, контроль качества. Начинается
с [индекса экспертной программы](experts/index.md).

Ключевые документы:

- [Программа экспертов](experts/01_expert_program.md)
- [Рубрики и лидерборды контроля качества](experts/docs_experts/qc_rubrics_leaderboards.md)
- [Task 1: траектории рассуждений](experts/docs_experts/task1_reasoning_trajectories.md)
- [Task 2: верификация темпорального мультимодального графа](experts/docs_experts/task2_graph_verification_temporal_mm.md)
- [Task 3: red teaming гипотез](experts/docs_experts/task3_hypothesis_redteaming.md)

---

## Ноутбуки

Канонические запускаемые примеры — [индекс ноутбуков](../../notebooks/README.md).

- [Task 1: траектории рассуждений (Colab)](../../notebooks/task1_reasoning_trajectories_form_colab.ipynb)
- [Task 2: валидация темпорального графа (Colab)](../../notebooks/task2_temporal_graph_validation_colab.ipynb)
- [Task 3: генерация гипотез (Colab)](../../notebooks/task3_multimodal_temporal_hypothesis_generation_colab.ipynb)
- [Task 3: blind A/B, локальные модели (Colab)](../../notebooks/task3_dual_local_models_blind_ab_colab.ipynb)

---

## Роли и оценивание

Роли помогают разделить сложность, а не ранжировать участников:
**Исследователь**, **Эксперт**, **Разработчик**, **Координатор**.
Подробная карта ответственности: [docs/roles.md](../roles.md).

Критерии оценивания вклада:

1. **Полезность** — изменение решает понятную задачу проекта.
2. **Проверяемость** — есть источник, тест, пример или процедура проверки.
3. **Воспроизводимость** — другой участник может повторить результат.
4. **Качество коммуникации** — контекст и ограничения описаны ясно.
5. **Совместность** — вклад облегчает следующий шаг другим людям.

Объём кода сам по себе не является критерием. Критерии готовности
отдельных артефактов заданы в [плане-графике](experts/00_master_plan_12weeks.md)
и в [рубриках контроля качества](experts/docs_experts/qc_rubrics_leaderboards.md).

---

## Для преподавателя

Как запустить поток, назначить роли, организовать ревью и завершить
курс публичным результатом — [руководство преподавателя](../../course/INSTRUCTOR_GUIDE.md).

## Машиночитаемая версия

Та же программа в машиночитаемом виде — [`syllabus.json`](syllabus.json).
Он предназначен для агентов и инструментов планирования.
