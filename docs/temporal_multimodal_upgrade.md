# Апгрейд пайплайна: Temporal GraphRAG + Multimodal GraphRAG (2025–2026)

Этот репозиторий обновлён так, чтобы поддерживать два ключевых направления развития GraphRAG:

1) **Мультимодальность (VL-модели + MM Knowledge Graph)** — извлекаем смысл из **текста, таблиц, фигур и страниц** и связываем его с текстовыми сущностями.
2) **Темпоральность (Temporal GraphRAG)** — различаем *один и тот же факт в разные периоды* и избегаем «темпоральных галлюцинаций».

---

## Что добавлено в код

### 1) Мультимодальная интеграция

- `src/scireason/mm/structured_pdf.py`
  - новый unified extractor поверх **Docling** + fallback на `pdf_mm_extract.py`
  - сохраняет `structured_chunks.jsonl` со стабильными объектами `text/table/figure/page`
  - экспортирует page images, figure images, table markdown/CSV/HTML, condition-friendly provenance

- `src/scireason/mm/vlm.py`
  - поддержка **открытых VLM** через `transformers`
  - добавлена поддержка **Qwen2-VL** и **g4f vision**

- `src/scireason/mm/mm_embed.py`
  - кросс‑модальные эмбеддинги через **OpenCLIP** (текст и изображения в одном пространстве)

- `src/scireason/graph/mm_neo4j_store.py`
  - сохраняет страницы в Neo4j как `(:Page)`

- `src/scireason/graph/build_tg_mmkg.py`
  - индексирует **все структурные чанки** в Qdrant
  - создаёт `Chunk`-узлы с modality/page/figure/table provenance
  - извлекает temporal assertions не только из plain-text, но и из `text/table/figure` evidence

### 2) Темпоральные графы

- `src/scireason/temporal/schemas.py`
  - `TimeInterval`, `TemporalTriplet`, `TemporalEvent`

- `src/scireason/temporal/temporal_triplet_extractor.py`
  - извлекает триплеты **с временем** (если есть в тексте)
  - если времени нет — подставляет `paper_year` как «суррогат» времени (MVP)

- `src/scireason/graph/temporal_neo4j_store.py`
  - хранит утверждения как `(:Assertion)` + связи к `(:Entity)`, `(:Paper)`, `(:Chunk)` и `(:Time)`
  - поддерживает vector indexes Neo4j для `Chunk` / `Assertion`

### 3) Экспертный слой поверх графа

- `src/scireason/report/expert_cards.py`
  - генерирует `chunk_cards.jsonl`
  - формирует auto-filled **Task 2 review cards** по шаблону курса

- `src/scireason/report/expert_report.py`
  - делает query-centric retrieval
  - запускает graph analytics (centrality, communities, bridges, link prediction)
  - строит визуализации temporal graph / timeline / community structure
  - собирает `expert_report.md/json`

- `src/scireason/pipeline/e2e.py`
  - `top-papers-graph run` теперь вызывает **полный** expert pipeline в одну команду

---

## Как включить мультимодальность

### Вариант A: локальный Qwen2-VL + OpenCLIP
```bash
pip install -e ".[mm,temporal,g4f]"

export VLM_BACKEND=qwen2_vl
export VLM_MODEL_ID=Qwen/Qwen2-VL-7B-Instruct
export MM_EMBED_BACKEND=open_clip

top-papers-graph run --query "your topic" --multimodal
```

### Вариант B: g4f route для VLM/LLM
```bash
pip install -e ".[g4f,mm,temporal]"

export VLM_BACKEND=g4f
export MM_EMBED_BACKEND=open_clip

top-papers-graph run --query "your topic" --llm g4f:deepseek-r1 --multimodal
```

---

## Как включить темпоральность

Темпоральная часть включается автоматически при `top-papers-graph run` и `build-tg-mmkg`.

Для проверки быстро:
- откройте Neo4j Browser и выполните:
```cypher
MATCH (a:Assertion)-[:AT_TIME]->(t:Time)
RETURN a.subject, a.predicate, a.object, t.start, t.end, t.granularity
LIMIT 25
```

---

## Что получают эксперты на второй задаче

1. **Карточки assertions** c полями:
   - `subject/predicate/object`
   - `time_interval`
   - `evidence.page`
   - `evidence.figure_or_table`
   - `evidence.snippet_or_summary`
   - авто-предложенный `verdict` и `rationale`
2. **Карточки чанков** с modality/page/section/condition hints.
3. **Готовый expert report** с query-centric evidence retrieval и графовыми визуализациями.

---

## Ограничения текущей версии (честно)

- Качество выделения отдельных фигур/таблиц зависит от версии Docling и структуры PDF.
- VLM JSON-parsing всё ещё мягкий: мы сохраняем текстовый multimodal summary, а не строгое schema-first представление.
- Глубокие графовые алгоритмы запускаются локально через NetworkX; для крупного production-графа стоит добавить отдельный графовый compute-layer.
