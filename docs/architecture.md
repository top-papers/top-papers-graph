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
   - LLM: извлечение temporal triplets / assertions из чанков
   - VLM: подписи страниц + таблицы/формулы (multimodal facts)

4) **Storage / indexing**
   - Qdrant: текстовые эмбеддинги чанков (`collection_text`) — legacy / backward compatibility
   - Qdrant: мультимодальные эмбеддинги (page text + page image, `collection_mm`)
   - Neo4j: граф знаний **и** векторные индексы для `Chunk` / `Assertion`

5) **Reasoning**
   - GraphRAG: retrieval → контекст → многoагентные “дебаты” → гипотеза
   - Verification: литература + temporal KG → verdict + план эксперимента
   - Temporal link prediction: TGNN/TGN-style event-stream predictor → кандидаты недостающих temporal links

6) **Human‑in‑the‑loop / data flywheel**
   - trajectories (рассуждения)
   - graph_reviews (вердикты по утверждениям)
   - hypothesis_reviews (вердикты по гипотезам)
   - mm_reviews (исправление VLM)
   - temporal_corrections (исправление времени)
   - CI валидирует артефакты, а скрипт экспорта формирует датасеты для SFT/RM.

## Data layout

Рекомендованный layout (git‑friendly):

```text
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

### Explainable layer

**Nodes**
- `(:Paper {id, title, year, source, url})`
- `(:Entity {name})`
- `(:Time {key, start, end, granularity})`
- `(:Assertion {id, predicate, object, polarity, confidence, evidence_quote, extraction_method, review_status, embedding})`
- `(:Chunk {id, paper_id, idx, text, embedding})`

**Relationships**
- `(Paper)-[:HAS_ASSERTION]->(Assertion)`
- `(Assertion)-[:SUBJECT]->(Entity)`
- `(Assertion)-[:OBJECT]->(Entity)`
- `(Assertion)-[:AT_TIME]->(Time)`
- `(Assertion)-[:EVIDENCE {quote}]->(Chunk)`
- `(Paper)-[:HAS_CHUNK]->(Chunk)`

### Event layer (for TGNN / temporal prediction)

**Nodes**
- `(:Event {id, ts_start, ts_end, granularity, confidence, polarity, split, event_type, extraction_method, weight})`

**Relationships**
- `(Event)-[:SOURCE_ENTITY]->(Entity)`
- `(Event)-[:TARGET_ENTITY]->(Entity)`
- `(Event)-[:ASSERTS]->(Assertion)`
- `(Event)-[:FROM_PAPER]->(Paper)`
- `(Event)-[:AT_TIME]->(Time)`

Идея простая:
- `Assertion` остаётся explainability / provenance слоем;
- `Event` образует хронологический stream, пригодный для temporal models.

## Temporal prediction design

Проект теперь предпочитает **TGNN/TGN-style** предсказание связей вместо статического GNN по умолчанию.

### Почему

Статический GNN работает на агрегированном графе и теряет порядок событий. Для temporal KG это плохо:
важно не только *что* связано, но и *когда* связь возникла и усилилась.

### Как это устроено в репозитории

1. Из temporal KG строится event stream (`TemporalEvent`).
2. Events сортируются хронологически.
3. TGNN/TGN-style predictor считает recency-aware node memory, temporal common neighbors и pair recurrence.
4. На выходе получаем top-k кандидатов `tgnn_missing_link`.
5. Опциональный статический GraphSAGE остаётся baseline-режимом для сравнения.

## Neo4j as graph + vector DB

Репозиторий поддерживает dual-write:
- текстовые и multimodal эмбеддинги по-прежнему можно писать в Qdrant;
- embeddings для `Chunk` и `Assertion` также записываются в Neo4j,
  где для них создаются vector indexes (best effort).

Это позволяет:
- использовать Neo4j как единый graph + vector backend;
- искать похожие чанки/утверждения и сразу идти по provenance traversal;
- упростить будущую миграцию к `neo4j-graphrag`.

## Where experts plug in

Эксперты влияют на систему двумя путями:

1) **instant feedback loop**
   - `graph_reviews` → `data/derived/expert_overrides.jsonl` → rule-based reward / фильтрация

2) **training data flywheel**
   - `python scripts/data/export_training_datasets.py`
   - получаются JSONL для SFT (и далее RM/DPO по мере взросления процесса)

3) **temporal corrections**
   - temporal fixes теперь можно применять как замену `old Assertion -> REPLACED_BY -> new Assertion`
   - параллельно сохраняется новый `Event`, чтобы temporal models видели корректный stream
