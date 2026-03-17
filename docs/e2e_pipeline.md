# E2E пайплайн: query → статьи → темпоральный KG → гипотезы

Команда для запуска:

```bash
top-papers-graph run --query "your topic" --sources all --top-papers 20
```

## Что делает пайплайн

1. **Поиск** статей по запросу через `OpenAlex / Semantic Scholar / ...`.
2. **Дедупликация + merge** метаданных (DOI/abstract/citations/pdf_url дополняют друг друга).
3. **Автоскачивание PDF** (best-effort, только OA/прямые ссылки).
4. **Ингест PDF**:
   - сначала через GROBID (если запущен)
   - если GROBID недоступен — fallback на PyMuPDF (текст по страницам)
5. **Построение темпорального графа знаний**:
   - режим `auto`: пытается извлекать временные триплеты LLM-экстрактором
   - если LLM недоступна — строит связи по **co-occurrence** термов
6. **Генерация проверяемых гипотез** на основе граф-сигналов:
   - emerging relations (рост связи во времени)
   - missing links (скрытая связь через общих соседей)

## Где лежат результаты

`runs/<timestamp>_<slug>/`:

- `papers_selected.json`
- `raw_pdfs/` и `raw_meta/`
- `processed_papers/` (если были PDF)
- `temporal_kg.json`
- `hypotheses.json` и `hypotheses.md`
- `review_queue/hypothesis_reviews/*.json` — шаблоны для экспертной разметки

## Как улучшать качество экспертной разметкой

1. Эксперты заполняют файлы:
   - `data/experts/graph_reviews/*.json` — правки к триплетам/ребрам
   - `data/experts/hypothesis_reviews/*.json` — оценка гипотез
2. Запускать компиляцию фидбэка:

```bash
top-papers-graph refresh-feedback
```

3. Следующий запуск `top-papers-graph run ...` автоматически подхватит
   `data/derived/expert_overrides.jsonl` и скорректирует скоринг рёбер/гипотез.
