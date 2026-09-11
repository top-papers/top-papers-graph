<!-- SPDX-FileCopyrightText: 2026 top-papers-graph contributors -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# Week 08 — Векторные baseline и абляции

## Результат недели

Участник реализует простой embedding baseline и честно сравнивает его с графовой эвристикой.

## Зачем это проекту

Сложная модель не должна приниматься без простого baseline. Абляция показывает, что именно даёт улучшение.

## Практика

- постройте spectral embedding или другой лёгкий baseline;
- вычислите cosine similarity для кандидатов;
- сравните top-K с Adamic–Adar;
- зафиксируйте одинаковый датасет, split и критерии.

## Вклад по ролям

- **Исследователь:** выбирает релевантный срез.
- **Эксперт:** слепо оценивает несколько кандидатов.
- **Разработчик:** реализует baseline и метрики.
- **Координатор:** собирает таблицу сравнения.

## Готово, когда

Сравнение воспроизводимо и включает случаи, где новый метод проигрывает.

## Материалы недели

- [Программа курса](../../docs/course/syllabus.md)
- [Повестка недели 8](../../docs/course/experts/agendas/week08_agenda.md)
- [Task 3: dual-local blind A/B](../../docs/course/tasks/task3_dual_local_blind_ab.md)
- [Ноутбук: blind A/B](../../notebooks/task3_dual_local_models_blind_ab_colab.ipynb)

## Проверьте себя

1. Какой baseline вы взяли и почему он честный?
2. Что показала абляция — какая часть вносит вклад?

## Источники

- [Task 3: dual-local blind A/B](../../docs/course/tasks/task3_dual_local_blind_ab.md)
