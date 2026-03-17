# E2E expert pipeline: query → multimodal corpus → temporal KG → hypotheses → report

Основная команда запуска:

```bash
top-papers-graph run \
  --query "your topic" \
  --sources all \
  --top-papers 20 \
  --multimodal \
  --vlm-backend qwen2_vl \
  --mm-embed-backend open_clip
```

## Что делает пайплайн

1. **Поиск** статей по запросу через `OpenAlex / Semantic Scholar / Crossref / arXiv / PubMed / Europe PMC / bioRxiv / medRxiv`.
2. **Дедупликация + rerank** метаданных по overlap с запросом, цитируемости, году и доступности PDF.
3. **Автоскачивание PDF** (best-effort, только OA/прямые ссылки).
4. **Структурный мультимодальный ingest PDF**:
   - сначала через **Docling**: `text/table/figure/page` chunks, table structure, page images, figure export;
   - если Docling или GROBID недоступны — fallback на существующий `PyMuPDF/pypdf` pipeline;
   - для figure/table chunks опционально вызывается **VLM** (`qwen2_vl`, `g4f`, `llava`, `phi3_vision`).
5. **Индексация в Qdrant**:
   - текстовые/структурные чанки → text collection;
   - image-bearing chunks → multimodal collection (`open_clip`).
6. **Сохранение logical + temporal graph в Neo4j**:
   - `(:Paper)`, `(:Chunk)`, `(:Assertion)`, `(:Event)`, `(:Time)`, `(:Page)`;
   - связи доказуемости и временной привязки для последующей экспертной верификации.
7. **Построение агрегированного temporal KG** по всему корпусу.
8. **TGNN/TGN-style link prediction** для поиска перспективных временных связей.
9. **Генерация проверяемых гипотез** по graph signals + TGNN candidates.
10. **Генерация артефактов для эксперта**:
    - `chunk_cards.jsonl`;
    - `graph_reviews_auto/*.json` по шаблону второй экспертной задачи;
    - `expert_report.md/json` и визуализации temporal graph.

## Где лежат результаты

`runs/<timestamp>_<slug>/`:

- `papers_selected.json`
- `raw_pdfs/` и `raw_meta/`
- `processed_papers/<paper_id>/structured_chunks.jsonl`
- `indexing_status.json`
- `temporal_kg.json`
- `hypotheses.json` и `hypotheses.md`
- `review_queue/chunk_cards.jsonl`
- `review_queue/graph_reviews_auto/*.json`
- `expert_report/expert_report.md`
- `expert_report/expert_report.json`
- `expert_report/*.png`
- `artifact_manifest.json`

## Как улучшать качество экспертной разметкой

1. Эксперты заполняют/правят файлы:
   - `data/experts/graph_reviews/*.json` — правки к assertions/рёбрам;
   - `data/experts/hypothesis_reviews/*.json` — оценка гипотез.
2. Запускать компиляцию фидбэка:

```bash
top-papers-graph refresh-feedback
```

3. Следующий запуск `top-papers-graph run ...` автоматически подхватит
   `data/derived/expert_overrides.jsonl` и скорректирует скоринг рёбер/гипотез.
