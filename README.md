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

### Student quickstart (3 команды)

**Linux/macOS:**
```bash
./scripts/bootstrap.sh
top-papers-graph demo-run --agent-backend smolagents --llm-provider mock --smol-model-backend scireason
top-papers-graph smoke-all --agent-backend smolagents --llm-provider mock --smol-model-backend scireason
```

**Windows (PowerShell):**
```powershell
.\scripts\bootstrap.ps1
top-papers-graph demo-run --agent-backend smolagents --llm-provider mock --smol-model-backend scireason
top-papers-graph smoke-all --agent-backend smolagents --llm-provider mock --smol-model-backend scireason
```

> Здесь используется **smolagents CodeAgent** (агент пишет и исполняет Python‑код) — это основной режим курса.

### 1) Установка
```bash
# Вариант 1 (рекомендуется для курса): одна команда
./scripts/bootstrap.sh  # Windows: .\scripts\bootstrap.ps1

# Вариант 2 (ручной):
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev,agents]"
```

> `g4f` теперь ставится автоматически вместе с `.[task2_notebook]`, `.[mm]`, `.[multimodal]` и `.[fullstack]`.
> Отдельный extra `.[g4f]` сохранён для явной установки только g4f.

### 2) Настройка
Скопируйте `.env.example` → `.env`.  
По умолчанию используется домен `science` (`configs/domains/science.yaml`).

### 3) Полностью автоматический пайплайн (рекомендуется)
Одна команда:
```bash
top-papers-graph run --query "graph neural network survey" --sources all --top-papers 20
```

#### Оффлайн демонстрация (без интернета и сервисов)
```bash
top-papers-graph demo-run --edge-mode cooccurrence
top-papers-graph smoke-all
```

#### Запуск в Docker (из коробки)

В репозитории есть Dockerfile для CLI/API и `docker-compose.yml`, который поднимает **всю инфраструктуру**, используемую проектом:
**Neo4j** (графовая БД), **Qdrant** (векторное хранилище для demo‑few‑shot) и **GROBID** (парсинг PDF).

```bash
# 1) Собрать образ и поднять стек
docker compose up -d --build

# 2) Запустить пайплайн внутри контейнера
docker compose exec app top-papers-graph run \
  --query "graph neural network survey" \
  --sources all \
  --top-papers 20

# 3) Остановить и удалить тома (опционально)
docker compose down -v
```

Подсказки:
- GROBID доступен на `http://localhost:8070` (проверка: `/api/isalive`). Если он недоступен, пайплайн автоматически
  переключится на локальный PDF‑парсер.
- Neo4j Browser: `http://localhost:7474`.
- Qdrant: `http://localhost:6333`.
- Для локального Ollama на хосте используйте `OLLAMA_BASE_URL` (по умолчанию `http://host.docker.internal:11434`).

По умолчанию LLM = **auto/auto**: проект пробует локальный Ollama (если доступен), иначе g4f (если установлен), иначе включает оффлайн `mock`.

Вы можете явно переопределить модель в команде:
```bash
# g4f (ставится по умолчанию в task2/mm/multimodal/fullstack; отдельно: pip install -e '.[g4f]')
top-papers-graph run --query "..." --g4f-model deepseek-r1

# (опционально) попробовать в первую очередь конкретные модели (если они есть в g4f/models.py)
G4F_MODEL_PREFER="gpt-4o-mini,deepseek-r1" top-papers-graph run --query "..."

# (опционально) ограничить число попыток автоподбора (по умолчанию 25)
G4F_AUTO_MAX_MODELS=10 top-papers-graph run --query "..."

# (опционально) форсировать список провайдеров g4f (RetryProvider)
G4F_PROVIDERS="Phind,FreeChatgpt,Liaobots" top-papers-graph run --query "..."

# локальная модель через Ollama
top-papers-graph run --query "..." --local-model llama3.2

# универсальный формат (provider:model)
top-papers-graph run --query "..." --llm g4f:gpt-4o-mini
top-papers-graph run --query "..." --llm ollama:llama3.2
```

Артефакты появятся в `runs/<timestamp>_<slug>/`:
- `temporal_kg.json` — темпоральный граф знаний (термы/связи/временные счётчики)
- `hypotheses.json` + `hypotheses.md` — ранжированный набор проверяемых гипотез
- `review_queue/` — шаблоны для экспертной разметки (hypothesis_reviews)

> Пайплайн старается скачать PDF (если доступен OA) и распарсить его через GROBID.
> Если GROBID не запущен, будет fallback‑парсинг PDF через `pypdf` (по умолчанию).
> Для более качественного парсинга установите опциональные зависимости: `pip install -e ".[mm]"` или `pip install -e ".[multimodal]"` (алиас для того же стека).
> Если PDF недоступен, пайплайн продолжит работу по абстрактам.

> **Опционально (GNN mode):** для “более взрослого” режима генерации гипотез через
> PyTorch Geometric (GNN link prediction) установите `pip install -e ".[gnn]"` и включите
> `HYP_GNN_ENABLED=1`. Подробности: `docs/gnn.md`.

> **Опционально (smolagents):** чтобы использовать Hugging Face **smolagents CodeAgent** вместо
> встроенного агента, установите `pip install -e ".[agents]"` и включите
> `HYP_AGENT_BACKEND=smolagents`.
> Для **локальных HF моделей**: `pip install -e ".[agents_hf]"`.
> Подробности: `docs/smolagents.md`.

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


## Full-stack OCR -> Temporal KG -> Multimodal verification

This repository now supports a staged full-stack pipeline aligned with the uploaded architecture notes:

- **Step 1 / OCR + document structure**: PaddleOCR + PP-Structure/PP-StructureV3 ingestion with a richer `ChunkRecord` contract and automatic fallback to GROBID / PyMuPDF.
- **Step 2 / unified PyTorch contour**: optional PyTorch Geometric TGN memory backend for temporal link prediction, Neo4j temporal KG, Qdrant dense+sparse hybrid retrieval, and Qwen2.5-VL-ready multimodal verification hooks.
- **Step 3 / Memgraph-centric analytics**: optional Memgraph + MAGE dual-write for temporal events/assertions/chunks, plus best-effort MAGE analytics snapshots.

### Docker stack

By default, `docker-compose.yml` builds the app with the `fullstack` extra and starts:

- Qdrant
- Neo4j
- Memgraph (MAGE image)
- GROBID
- the application container

Example:

```bash
docker compose up --build
```

### Key environment switches

- `OCR_BACKEND=auto|paddleocr|grobid|pymupdf`
- `PADDLEOCR_WORKER_TIMEOUT_SECONDS=90`, `PADDLEOCR_WORKER_TIMEOUT_PER_PAGE_SECONDS=8`, `PADDLEOCR_WORKER_TIMEOUT_MAX_SECONDS=900`
- `GRAPH_BACKEND=dual|neo4j|memgraph|none`
- `QDRANT_RETRIEVAL_MODE=hybrid|dense`
- `HYP_TGNN_BACKEND=auto|pyg|heuristic`
- `VLM_BACKEND=none|qwen2_vl|g4f`

### Product contracts

The main product artifact chain is now explicitly modeled as:

`ChunkRecord -> TemporalEvent -> Hypothesis`

`ChunkRecord` is stored in `chunks.jsonl`, reused for Qdrant payloads, temporal graph provenance, and downstream multimodal verification.
