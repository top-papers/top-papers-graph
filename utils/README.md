# Task 3 A/B — утилита-сборщик диагностического набора (top-papers-graph)

Инструмент для подготовки диагностического набора кейсов A/B-теста Task 3: где
дообученная VLM (SFT/DPO) сравнивается с baseline на графе знаний и **мультимодальных
evidence** (рисунки, таблицы, формулы) из научных статей.

Проект **пустой по умолчанию**: тему, статьи и кейсы задаёте вы. Статьи и кейсы
описываются в одном YAML-конфиге, метаданные эксперимента — в GUI. Никакого
предзаполненного содержимого и обращений к сети.

## Как это работает

```
config.yaml ──► scripts/build.py ──► task3_ab_case_manifest.filled.json
                                  └─► web_form/ (GUI-форма, предзагружена манифестом)
                                  └─► validation/VALIDATION_LOG.txt
```

1. Опишите статьи и кейсы в [`config.yaml`](config.yaml).
2. Соберите набор: `python3 scripts/build.py`.
3. Откройте GUI (веб-форма или telegram-бот), заполните метаданные эксперимента,
   при необходимости отредактируйте кейсы и выгрузите итоговый
   `task3_ab_case_manifest.filled.json`.

## Конфиг (`config.yaml`)

Два списка — `papers` (статьи) и `cases` (кейсы). По умолчанию пусты.

```yaml
papers:
  - id: arxiv:2305.02402
    title: Normalizing flows for lattice gauge theory
    year: 2023
cases:
  - paper_id: arxiv:2305.02402      # ссылка на papers[].id (title/year подтянутся)
    stratum: multimodal_hard        # multimodal_hard | temporal_hard | easy_control
    evidence_kind: figure           # figure | table | figure_or_table | formula | page | mixed
    page_hint: "Fig. 1"
    creator_prompt: "Один короткий конкретный вопрос по объекту статьи."
    creator_rationale: "Почему кейс диагностичен и что теряет baseline."
    review_focus: [evidence, visual_fact]       # evidence | visual_fact | temporal | overall
    expected_error_modes: [missed_visual_fact]  # missed_visual_fact | wrong_evidence_linkage | needs_time_fix | hallucinated_visual_inference
    match:
      hypothesis_title_contains: "..."
```

`case_id` проставляется автоматически (`case_01`, `case_02`, …), если не задан.
Метаданные эксперимента (`topic`, `creator_id`, `cutoff_year`, `review_goal`) в конфиге
**не указываются** — их заполняют в GUI.

## Заполнение через GUI

### Вариант A — веб-форма (сервер на :9000)
```bash
python3 web_form/serve.py            # http://localhost:9000
```
Все поля (метаданные и кейсы) редактируются в браузере. **«Сохранить на сервер»**
пишет манифест прямо в корневой `task3_ab_case_manifest.filled.json`. Подробности —
[`web_form/README.md`](web_form/README.md).

### Вариант B — telegram-бот
Впишите токен бота в `telegram_bot/.env`, запустите `python3 telegram_bot/bot.py`,
заполняйте набор командами в чате, `/export` отдаёт манифест. Подробности —
[`telegram_bot/README.md`](telegram_bot/README.md).

> Отправка в Google Форму — **ручной шаг человека**; ни форма, ни бот её не делают.

## Схема и правило готовности

Источник истины по схеме — ячейка-сборщик формы (cell 8) блокнота
[`notebook/`](notebook/). Манифест:

```jsonc
{
  "schema_version": "task3-ab-creator-v1",
  "experiment_meta": { "topic", "submission_id", "creator_id", "cutoff_year", "review_goal" },
  "cases": [ /* ... */ ]
}
```

Кейс получает статус **«Готов к экспорту»**, когда непусты `paper_title`,
`creator_prompt`, `creator_rationale`, `review_focus` (≥1) и **хотя бы одно поле `match.*`**.

## Структура репозитория

```
.
├── config.yaml                          # статьи + кейсы (заполняете вы)
├── task3_ab_case_manifest.filled.json   # итоговый манифест (генерируется build.py)
├── notebook/      # исходный блокнот Task 3 (источник схемы и шаблона формы)
├── web_form/      # GUI-форма + веб-сервер на :9000
├── telegram_bot/  # альтернатива форме: бот сбора кейсов (токен в .env)
├── scripts/       # build.py — сборщик из config.yaml (зависимость: PyYAML)
└── validation/    # VALIDATION_LOG.txt — отчёт валидации последней сборки
```

В каждой папке есть свой `README.md`.

## Зависимости
- `scripts/build.py` — PyYAML (`pip install -r scripts/requirements.txt`).
- `web_form/serve.py` — только стандартная библиотека Python.
- `telegram_bot/bot.py` — python-telegram-bot (`pip install -r telegram_bot/requirements.txt`).
