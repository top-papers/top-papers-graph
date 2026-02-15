# top-papers-graph — генерация научных гипотез из графа знаний, который создается на основе научных публикаций и подходит для *любой* тематики

Проект собирает публикации из нескольких крупных источников, нормализует метаданные в единый формат и помогает строить **проверяемый** граф знаний (KG) и evidence‑based синтез по вашей теме.

> CLI: `top-papers-graph ...` (алиас `scireason ...` сохранён для обратной совместимости)

## Возможности
- **Источники публикаций**: arXiv, PubMed (NCBI), Europe PMC, bioRxiv/medRxiv, Crossref, OpenAlex, Semantic Scholar.
- **Единая схема метаданных**: `PaperMetadata` (Pydantic) + нормализация ответов всех источников.
- **Resolver идентификаторов**: DOI ⇄ PMID ⇄ arXivID ⇄ OpenAlexID.
- **Кеширование и rate-limit** на уровне HTTP‑клиента (особенно полезно для NCBI/Crossref).
- **CLI**, **FastAPI** (наружный API) и **MCP‑сервер** (интеграции с AI‑сервисами).

## Быстрый старт
### 1) Установка
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 2) Настройка
Скопируйте `.env.example` → `.env`.  
По умолчанию используется домен `science` (`configs/domains/science.yaml`).

### 3) Полностью автоматический пайплайн (рекомендуется)
Одна команда:
```bash
top-papers-graph run --query "graph neural network survey" --sources all --top-papers 20
```

Артефакты появятся в `runs/<timestamp>_<slug>/`:
- `temporal_kg.json` — темпоральный граф знаний (термы/связи/временные счётчики)
- `hypotheses.json` + `hypotheses.md` — ранжированный набор проверяемых гипотез
- `review_queue/` — шаблоны для экспертной разметки (hypothesis_reviews)

> Пайплайн старается скачать PDF (если доступен OA) и распарсить его через GROBID.
> Если PDF недоступен, он продолжит работу по абстрактам.

### 4) Поиск статей по вашей теме (отдельный шаг)
```bash
top-papers-graph fetch "graph neural network survey" --source arxiv --limit 10 --out data/papers/arxiv.json
top-papers-graph fetch "graph neural network survey" --source pubmed --limit 10 --out data/papers/pubmed.json
```

### 5) Наружный API (FastAPI)
```bash
pip install -e ".[api]"
top-papers-graph-api
```

### 6) MCP‑сервер
```bash
pip install -e ".[mcp]"
top-papers-graph-mcp
```

## Конфиг домена (topic‑agnostic)
- Домен настраивается YAML‑файлом в `configs/domains/<DOMAIN_ID>.yaml`.
- “Скептик” (критический чек‑лист) задаётся файлом в `configs/checklists/`.

Смотрите: `docs/quickstart.md`, `docs/architecture.md`, `docs/sources.md`.

## Примеры
- `examples/battery_fastcharge/` — пример домена “быстрая зарядка батарей” с PyBaMM и профилями зарядки.
  - Чтобы включить пример, укажите `DOMAIN_ID=ied_fastcharge` и пути на конфиги из `examples/...`.

## Лицензия
См. `LICENSE`.
