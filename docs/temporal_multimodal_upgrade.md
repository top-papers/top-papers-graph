# Апгрейд пайплайна: Temporal GraphRAG + Multimodal GraphRAG (2025–2026)

Этот репозиторий обновлён так, чтобы поддерживать два ключевых направления развития GraphRAG:

1) **Мультимодальность (VL-модели + MM Knowledge Graph)** — извлекаем смысл из **таблиц/фигур/формул** и связываем его с текстовыми сущностями.
2) **Темпоральность (Temporal GraphRAG)** — различаем *один и тот же факт в разные периоды* и избегаем «темпоральных галлюцинаций».

---

## Что добавлено в код

### 1) Мультимодальная интеграция

- `src/scireason/mm/pdf_mm_extract.py`
  - рендерит PDF по страницам в `mm/images/page_XXX.png`
  - сохраняет `mm/pages.jsonl` с текстом по странице + (опционально) VLM-описанием

- `src/scireason/mm/vlm.py`
  - адаптер к **открытым VL-моделям** через `transformers`
  - по умолчанию `VLM_BACKEND=none` (не требует GPU)

- `src/scireason/mm/mm_embed.py`
  - кросс‑модальные эмбеддинги через **OpenCLIP** (текст и изображения в одном пространстве)

- `src/scireason/graph/mm_neo4j_store.py`
  - сохраняет страницы/картинки/таблицы в Neo4j как `(:Page)` и связи `(:Paper)-[:HAS_PAGE]->(:Page)`

- `src/scireason/graph/mm_retrieval.py`
  - retrieval по мультимодальному индексу Qdrant (text→image/text)

### 2) Темпоральные графы

- `src/scireason/temporal/schemas.py`
  - `TimeInterval`, `TemporalTriplet`

- `src/scireason/temporal/temporal_triplet_extractor.py`
  - извлекает триплеты **с временем** (если есть в тексте)
  - если времени нет — подставляет `paper_year` как «суррогат» времени (MVP)

- `src/scireason/graph/temporal_neo4j_store.py`
  - хранит утверждения как `(:Assertion)` + связи к `(:Entity)`, `(:Paper)` и `(:Time)`
  - таким образом можно иметь **несколько утверждений** между теми же сущностями, но *в разные периоды*

- `src/scireason/graph/build_tg_mmkg.py`
  - единая сборка: текстовый индекс + темпоральные утверждения + мультимодальные страницы

---

## Как включить мультимодальность

### Вариант A (рекомендуемый MVP): только страницы/картинки без VLM
1) Установить зависимости:
```bash
pip install -e ".[mm]"
```
2) В `.env`:
```env
VLM_BACKEND=none
MM_EMBED_BACKEND=open_clip
```
3) Запустить:
```bash
top-papers-graph parse-mm --pdf data/raw_pdfs/PAPER.pdf --meta configs/meta/paper.json --vlm false
top-papers-graph build-tg-mmkg --paper-dir data/papers/parsed/<paper_id> --collection-text demo --collection-mm demo_mm
```

### Вариант B: с VLM (подписи, извлечение таблиц/формул)
1) В `.env`:
```env
VLM_BACKEND=qwen2_vl
VLM_MODEL_ID=Qwen/Qwen2-VL-7B-Instruct
```
2) Повторить `parse-mm` с `--vlm true`.

> На CPU будет медленно. Для курса это нормально как «инструмент на выделенных машинах».
> Для продакшн‑скорости: vLLM + quantization (следующая итерация проекта).

---

## Как включить темпоральность

Темпоральная часть включается автоматически при `build-tg-mmkg`.

Для проверки быстро:
- откройте Neo4j Browser и выполните:
```cypher
MATCH (a:Assertion)-[:AT_TIME]->(t:Time)
RETURN a.predicate, a.confidence, t.start, t.end, t.granularity
LIMIT 25
```

---

## Что должны сделать участники (эксперты) в новых потоках

1) **MM verification**
   - проверить, что к страницам с ключевыми графиками/таблицами есть адекватные подписи
   - поправить “ошибочные” извлечения таблиц/формул
   - отметить важные визуальные объекты (в будущем: как отдельные ноды Figure/Table)

2) **Temporal sanity-check**
   - проверить корректность «привязки ко времени» у ключевых утверждений
   - где важно — выставить интервал точнее (например, “2018–2022”, “Q1 2024”)

---

## Ограничения текущей версии (честно)

- Детект отдельных фигур/таблиц пока не реализован — MVP хранит *страницы целиком*.
- Временная иерархия (year→month→day) и PPR‑поиск как в академических работах — планируемая следующая итерация.
- VLM‑вывод сейчас парсится «мягко» (текстом). На следующем шаге переведём на JSON‑schema.

