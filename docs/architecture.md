# Architecture

Эта страница описывает целевую архитектуру фреймворка SciReason:
ИИ‑агент анализирует научные статьи (text + vision), строит **темпоральный граф знаний**,
а затем генерирует и проверяет научные гипотезы.

## Пайплайн end‑to‑end

1) **Search / acquisition**
   - коннекторы: arXiv, OpenAlex, Semantic Scholar
   - `top-papers-graph fetch ...`
   - `top-papers-graph ingest-arxiv ...`

2) **Ingestion (PDF → artifacts)**
   - GROBID: PDF → structured text → `chunks.jsonl`
   - Multimodal: PDF → pages (`mm/pages.jsonl`) + images (`mm/images/...`) + (опционально) VLM подписи/таблицы/формулы

3) **Extraction**
   - LLM: извлечение (temporal) триплетов/утверждений из чанков
   - VLM: подписи страниц + таблицы/формулы (multimodal facts)

4) **Storage / indexing**
   - Qdrant: текстовые эмбеддинги чанков (`collection_text`)
   - Qdrant: мультимодальные эмбеддинги (page text + page image, `collection_mm`)
   - Neo4j: граф знаний (Paper, Entity, Assertion, Time, Chunk)

5) **Reasoning**
   - GraphRAG: retrieval → контекст → многoагентные “дебаты” → гипотеза
   - Verification: литература + temporal KG → verdict + план эксперимента

6) **Human‑in‑the‑loop / data flywheel**
   - ≈80 участников курса (экспертов по предметной области) создают маленькие артефакты (JSON/YAML):
     - trajectories (рассуждения)
     - graph_reviews (вердикты по утверждениям)
     - hypothesis_reviews (вердикты по гипотезам)
     - mm_reviews (исправление VLM)
     - temporal_corrections (исправление времени)
   - CI валидирует артефакты, а скрипт экспорта формирует датасеты для SFT/RM.

## Data layout

Рекомендованный layout (git‑friendly):

```
data/
  raw/
    papers/        # PDF (малый объём или ссылки)
    metadata/      # meta.json
  processed/
    papers/<paper_id>/
      meta.json
      chunks.jsonl
      mm/
        pages.jsonl
        images/
  experts/
    trajectories/
    graph_reviews/
    hypothesis_reviews/
    mm_reviews/
    temporal_corrections/
  derived/
    training/      # JSONL для fine‑tuning
```

## Graph schema (Neo4j)

**Nodes**
 - `(:Paper {id, title, year, source, url})`
 - `(:Entity {name})`
 - `(:Time {key, start, end, granularity})`
 - `(:Assertion {id, predicate, object, polarity, confidence, evidence_quote})`
 - `(:Chunk {id, paper_id, idx, text})` (MVP: текст усечён)

**Relationships**
 - `(Paper)-[:HAS_ASSERTION]->(Assertion)`
 - `(Entity)-[:SUBJECT_OF]->(Assertion)`
 - `(Assertion)-[:AT_TIME]->(Time)`
 - `(Assertion)-[:EVIDENCE {quote}]->(Chunk)`
 - `(Paper)-[:HAS_CHUNK]->(Chunk)`

## Where experts plug in

Эксперты влияют на систему двумя путями:

1) **instant feedback loop**
   - `graph_reviews` → `data/derived/expert_overrides.jsonl` → rule-based reward/фильтрация

2) **training data flywheel**
   - `python scripts/data/export_training_datasets.py`
   - получаются JSONL для SFT (и далее RM/DPO по мере взросления процесса)
