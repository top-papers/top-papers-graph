# One-click ingestion + мгновенный feedback loop (expert → agents)

## Ingestion “в один клик” (arXiv)
Скачивает PDF с arXiv, резолвит метаданные и запускает ingest:

```bash
top-papers-graph ingest-arxiv 2401.01234 --multimodal true --build-graph true
```

Выход:
- PDF: `data/raw/papers/<id>.pdf`
- metadata: `data/raw/metadata/<id>.json`
- обработанные артефакты: `data/processed/papers/<id>/...`

Метаданные резолвятся из arXiv Atom API и, по возможности, обогащаются через Crossref REST API.
