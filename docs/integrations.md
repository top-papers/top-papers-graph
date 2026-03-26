# Интеграции

## top-papers-bot (Telegram)
Код бота лежит в `third_party/top-papers-bot/` (GPL-3.0).

### Как использовать бота как “вход” в SciReason
1) В боте сделайте поиск.
2) Нажмите “📥 Скачать все результаты (JSON)”.
3) Сохраните файл, например `papers_search_results_*.json`.
4) Импортируйте мета-файлы:
```bash
top-papers-graph import-top-papers --inp papers_search_results_*.json --out-dir configs/top_papers_meta
```
5) Дальше выбирайте нужные meta-файлы и скачивайте PDF (пока вручную) в `data/raw_pdfs/`, затем:
```bash
top-papers-graph parse --pdf data/raw_pdfs/<paper>.pdf --meta configs/top_papers_meta/<id>.meta.json --out-dir data/papers/parsed
top-papers-graph build-kg --paper-dir data/papers/parsed/<id> --collection demo
```

> Следующий шаг (в бэклоге): ingestion API, куда бот сможет отправлять результаты автоматически.
