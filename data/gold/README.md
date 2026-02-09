# Gold sets (evaluation)

Эта папка содержит *эталонные* (gold) кейсы для измерения качества:

- `data/gold/triplets/*.json` — оценка извлечения темпоральных триплетов
- `data/gold/hypotheses/*.json` — оценка проверки гипотез (verdict + evidence)

## Формат: triplets

Файл `data/gold/triplets/<case_id>.json`:

```json
{
  "domain": "example_domain",
  "paper_year": 2022,
  "chunk_text": "Текстовый фрагмент из статьи (chunk/page caption)...",
  "expected_triplets": [
    {
      "subject": "...",
      "predicate": "...",
      "object": "...",
      "confidence": 0.9,
      "polarity": "supports|contradicts|unknown",
      "evidence_quote": "...",
      "time": {"start":"2022", "end":"2022", "granularity":"year"}   // или null
    }
  ]
}
```

## Формат: hypotheses

Файл `data/gold/hypotheses/<case_id>.json`:

```json
{
  "domain": "example_domain",
  "collection_text": "papers_text",   // Qdrant коллекция с чанками (GraphRAG). Можно опустить.
  "hypothesis": { /* HypothesisDraft JSON */ },
  "ctx_override": [ /* опционально: зафиксированный контекст (результат retrieve_context) */ ],
  "expected": {
    "verdict": "supported|contradicted|insufficient_evidence|needs_revision",
    "supporting_evidence": [{"source_id":"doi:...", "text_snippet":"..."}]
  }
}
```

## Запуск оценки

```bash
# Triplets: baseline vs fewshot
python scripts/eval/eval_triplets.py --gold-dir data/gold/triplets --variant both

# Hypotheses: baseline vs fewshot
python scripts/eval/eval_hypotheses.py --gold-dir data/gold/hypotheses --variant both --collection-text papers_text
```

> Важно: скрипты вызывают LLM (`chat_json`) и эмбеддинги. Убедитесь, что настроены `.env` и подняты сервисы (Qdrant/Neo4j), если используете retrieval без ctx_override.
