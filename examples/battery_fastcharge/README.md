# Example: battery_fastcharge (PyBaMM)

Этот пример оставлен как **демо-домен**, чтобы показать, как подключать доменные конфиги, чек‑листы и (опционально) доменный валидатор/симулятор.

## Как включить пример
В `.env`:

```bash
DOMAIN_ID=ied_fastcharge
DOMAIN_CONFIG_PATH=examples/battery_fastcharge/configs/domains/ied_fastcharge.yaml
SKEPTIC_CHECKLIST_PATH=examples/battery_fastcharge/configs/checklists/ied_fastcharge.md
CHARGING_PROFILES_DIR=examples/battery_fastcharge/configs/charging_profiles
```

Установка зависимостей (с PyBaMM):
```bash
pip install -e ".[dev,battery]"
```

## Документация примера
- `docs/quickstart_ied_case_study.md`
- `docs/domain_case_study_ied_fastcharge.md`
- `docs/plan_12weeks.md`
