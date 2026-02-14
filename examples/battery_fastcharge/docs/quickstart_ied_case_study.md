# Quickstart (IED case study)

## 0) Включить пример домена

В `.env` (или через переменные окружения) установите:

```bash
DOMAIN_ID=ied_fastcharge
DOMAIN_CONFIG_PATH=examples/battery_fastcharge/configs/domains/ied_fastcharge.yaml
SKEPTIC_CHECKLIST_PATH=examples/battery_fastcharge/configs/checklists/ied_fastcharge.md
CHARGING_PROFILES_DIR=examples/battery_fastcharge/configs/charging_profiles
```

## 1) Поднять сервисы
```bash
docker compose up -d
```

## 2) Установить зависимости
```bash
pip install -e ".[dev,battery]"
```

## 3) Сделать baseline симуляцию PyBaMM
```bash
top-papers-graph pybamm-fastcharge --profile baseline_cc --out-dir results/pybamm/baseline
```

## 4) Собрать минимальный набор литературы (метаданные)
```bash
top-papers-graph fetch "lithium plating fast charging protocol" --source openalex --limit 30 --out data/papers/search_ied_fastcharge.json
```

## 5) Парсинг PDF и KG (нужно для debate с ретривером)
- скачайте PDF в `data/raw/papers/`
- для каждого PDF подготовьте meta JSON в `data/raw/metadata/` (пример: `configs/meta_example.json`)
- затем:
```bash
top-papers-graph parse --pdf data/raw/papers/<file>.pdf --meta data/raw/metadata/<meta>.json --out-dir data/processed/papers
top-papers-graph build-kg --paper-dir data/processed/papers/<paper_id> --collection ied_fastcharge --domain "Electrochemical Energy"
```

## 6) Сгенерировать гипотезу через дебаты
```bash
top-papers-graph debate "Design a two-stage charging profile to reduce lithium plating risk." \
  --collection ied_fastcharge --domain "Electrochemical Energy" --max-rounds 3 \
  > results/hypotheses/debate_run.json
```

Если хотите запустить дебаты без ретривера (например, только на общем знании модели), добавьте флаг `--allow-empty-context`.

## 7) Запустить proposed профиль в PyBaMM
```bash
top-papers-graph pybamm-fastcharge --profile proposed_two_stage --out-dir results/pybamm/run_1
```

## 8) Подготовить статью
См. `paper/agents4science/README.md`.


## Опционально: мультимодальность (страницы/таблицы/графики)
Если хотите извлекать смысл из графиков/таблиц внутри PDF:
1) Установите зависимости:
```bash
pip install -e ".[mm]"
```
2) Парсинг с мультимодальностью:
```bash
top-papers-graph parse-mm --pdf data/raw/papers/<file>.pdf --meta data/raw/metadata/<paper>.json --out-dir data/processed/papers --vlm false
```
3) Сборка Temporal+MM:
```bash
top-papers-graph build-tg-mmkg --paper-dir data/processed/papers/<paper_id> --collection-text demo --collection-mm demo_mm --domain Battery
```
Подробности: `docs/temporal_multimodal_upgrade.md`.
