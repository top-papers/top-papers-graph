# Improvements for the expert Task 2 pipeline

Этот файл кратко фиксирует, что было усилено в архиве по отношению к исходной версии репозитория.

## Что изменено

### 1) Полный one-command expert pipeline
Команда `top-papers-graph run --query "..."` теперь выполняет весь контур:

1. поиск публикаций;
2. скачивание PDF;
3. структурный мультимодальный ingest в `text/table/figure/page` chunks;
4. индексацию в Qdrant;
5. сохранение logical/temporal graph в Neo4j;
6. построение temporal KG;
7. TGNN-style temporal link prediction;
8. генерацию гипотез;
9. генерацию chunk cards и auto-filled Task-2 review cards;
10. выпуск expert report с визуализациями.

### 2) Структурный мультимодальный ingest
Добавлен `src/scireason/mm/structured_pdf.py`:
- приоритетный backend: **Docling**;
- fallback: существующий `PyMuPDF/pypdf` pipeline;
- сохраняет `structured_chunks.jsonl`.

### 3) Поддержка открытых мультимодальных моделей
Обновлён `src/scireason/mm/vlm.py`:
- поддержка **Qwen2-VL** через `transformers`;
- поддержка **g4f vision API**;
- сохранена совместимость с другими transformers-совместимыми VLM.

### 4) Индексация и графовое хранилище
Обновлён `src/scireason/graph/build_tg_mmkg.py`:
- в Qdrant индексируются все структурные чанки;
- в Neo4j сохраняются `Chunk/Page/Assertion/Event/Time` узлы;
- temporal assertions извлекаются не только из текста, но и из figure/table evidence.

### 5) Экспертная верификация по шаблону второй задачи
Добавлены:
- `src/scireason/report/expert_cards.py`
- `src/scireason/report/expert_report.py`

Они формируют:
- `review_queue/chunk_cards.jsonl`
- `review_queue/graph_reviews_auto/*.json`
- `expert_report/expert_report.md`
- `expert_report/expert_report.json`
- `expert_report/*.png`

## Главные артефакты запуска

В `runs/<timestamp>_<slug>/` появляются:
- `processed_papers/`;
- `temporal_kg.json`;
- `hypotheses.json` / `hypotheses.md`;
- `indexing_status.json`;
- `review_queue/chunk_cards.jsonl`;
- `review_queue/graph_reviews_auto/`;
- `expert_report/`;
- `artifact_manifest.json`.

## Рекомендуемый запуск

### Локальный Qwen2-VL
```bash
top-papers-graph run \
  --query "your topic" \
  --sources all \
  --top-papers 20 \
  --multimodal \
  --vlm-backend qwen2_vl \
  --vlm-model-id Qwen/Qwen2-VL-7B-Instruct \
  --mm-embed-backend open_clip
```

### g4f route
```bash
top-papers-graph run \
  --query "your topic" \
  --sources all \
  --top-papers 20 \
  --multimodal \
  --llm g4f:deepseek-r1 \
  --vlm-backend g4f \
  --mm-embed-backend open_clip
```
