<!-- SPDX-FileCopyrightText: 2026 top-papers-graph contributors -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# Интеграции

## top-papers-bot (Telegram)
Бот развивается как отдельный внешний проект. Этот репозиторий хранит только совместимый импорт JSON и не включает копию его исходного кода.

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
