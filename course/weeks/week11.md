<!-- SPDX-FileCopyrightText: 2026 top-papers-graph contributors -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# Week 11 — GNN/TGNN и честная оценка

## Результат недели

Участник сравнивает модель с baseline на фиксированном split и добавляет метрику или benchmark case.

## Зачем это проекту

Без holdout и слепой проверки легко измерить запоминание, утечку или красивый пример вместо обобщения.

## Практика

- включите опциональный `.[gnn]`/TGNN-трек;
- подготовьте temporal или edge holdout;
- сравните heuristics, spectral baseline и модель;
- измерьте hits@k/precision@k и экспертные novelty/soundness/testability.

## Вклад по ролям

- **Исследователь:** определяет валидный split.
- **Эксперт:** проводит blind review.
- **Разработчик:** реализует eval и защиту от leakage.
- **Координатор:** фиксирует протокол и результаты.

## Готово, когда

Метрика рассчитана на неизменяемом наборе, негативные результаты сохранены, а вывод не превышает доказательства.

## Материалы недели

- [Программа курса](../../docs/course/syllabus.md)
- [Повестка недели 11](../../docs/course/experts/agendas/week11_agenda.md)
- [Task 2: проверка графа](../../docs/course/tasks/task2_validation.md)
- [Task 3: гипотезы](../../docs/course/tasks/task3_hypotheses.md)

## Проверьте себя

1. Проверена ли модель на holdout, которого не было при разработке?
2. Какие ограничения результата вы описали явно?

## Источники

- [Task 2: проверка графа](../../docs/course/tasks/task2_validation.md)
