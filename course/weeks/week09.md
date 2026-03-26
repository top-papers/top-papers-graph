# Week 09 — ИИ-агент, который пишет код: internal vs smolagents

## Цели занятия

- Понять ключевые понятия недели и как они нужны проекту.
- Получить артефакт/изменение в репозитории по итогам практики.

## Теория (простыми словами)

- Что такое **code-writing agent** и чем он отличается от “tool-calling”.
- Почему нужен sandbox и какие ограничения разумны в учебном проекте.
- Как агент встраивается в пайплайн: анализ графа → выбор кандидатов → формулировка гипотез.

Два подхода, которые поддерживает проект:

1) **internal (по умолчанию)** — `src/scireason/agentic/code_agent.py`
   - максимально простая реализация, легко читать и модифицировать
   - полезно, чтобы понять “скелет” агента: prompt → code → exec → final_answer
   - работает полностью оффлайн с `LLM_PROVIDER=mock`

2) **smolagents (Hugging Face)** — `HYP_AGENT_BACKEND=smolagents`
   - использует `smolagents.CodeAgent`
   - удобен как “взрослая” рамка: декларативные tools, больше готовых паттернов, варианты sandbox
   - поддерживает разные бэкенды моделей:
     - `SMOL_MODEL_BACKEND=scireason` (использует общий роутер `LLM_PROVIDER=...`)
     - `SMOL_MODEL_BACKEND=g4f` (прямой g4f)
     - `SMOL_MODEL_BACKEND=transformers` (локальная HF модель)
   - подробности: `docs/smolagents.md`

## Практика в контексте проекта

- Изучить:
  - internal агента: `src/scireason/agentic/code_agent.py`
  - smolagents интеграцию: `src/scireason/smolagents_integration/*` + `docs/smolagents.md`
  - инструменты графа: `src/scireason/agentic/graph_tools.py`

- Прогнать оффлайн пайплайн (важно для воспроизводимости):
  ```bash
  top-papers-graph demo-run --llm-provider mock
  ```

- (если установлен `.[agents]`) попробовать переключить backend агента:
  ```bash
  export HYP_AGENT_BACKEND=smolagents
  export SMOL_MODEL_BACKEND=scireason
  top-papers-graph demo-run --llm-provider mock
  ```

- Добавить новый tool или улучшить существующий:
  - shortest paths (`shortest_path_terms`)
  - centrality ranking (pagerank / betweenness)
  - edge prediction (classic heuristics / spectral / GNN)

- Сравнить качество кандидатов:
  - internal vs smolagents
  - link_prediction vs spectral_link_prediction vs (если включили) gnn_link_prediction

## Что сдаём (deliverables)

- Короткий отчёт (5–10 предложений) + ссылка на артефакты/PR.
- 1–2 скриншота/фрагмента `temporal_kg.json` или `hypotheses.json` с комментариями.
- Список следующих вопросов/рисков (что непонятно, что хочется улучшить).
