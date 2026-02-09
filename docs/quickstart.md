# Quickstart (универсальный)

## 1) Установка
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## 2) Поднять сервисы (опционально)
```bash
docker compose up -d
```

## 3) Настроить .env
Скопируйте `.env.example` в `.env` и заполните ключи (если нужны).  
Дефолтный домен — `science` (configs/domains/science.yaml).

## 4) Найти статьи по вашей теме
```bash
top-papers-graph fetch "graph neural networks survey" --source arxiv --limit 10 --out data/papers/arxiv.json
top-papers-graph fetch "graph neural networks survey" --source pubmed --limit 10 --out data/papers/pubmed.json
```

## 5) Построить KG по статье (пример)
```bash
top-papers-graph build-kg --paper-dir data/processed/papers/<paper_id> --collection science --domain "Science"
```

## 6) Запустить дебаты агентов (пример)
```bash
top-papers-graph debate "Summarize the most robust evidence about X and list falsifiable hypotheses." --out results/debate.json
```

## Примеры доменов
- Battery fast-charge (PyBaMM) — см. `examples/battery_fastcharge/`
