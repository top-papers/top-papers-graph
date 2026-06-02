# scripts/

`build.py` — единственный сборщик набора. Запускать **из корня репозитория**.

## Запуск

```bash
pip install -r scripts/requirements.txt   # один раз: PyYAML
python3 scripts/build.py
```

## Что делает `build.py`

| вход | выход |
|---|---|
| [`../config.yaml`](../config.yaml) (papers + cases) + блокнот | `task3_ab_case_manifest.filled.json`, `web_form/...html`, `validation/VALIDATION_LOG.txt` |

Шаги:
1. Читает `config.yaml`: `papers` → таблица id→(title, year), `cases` → спеки кейсов.
2. Собирает манифест: `experiment_meta` — пустые поля (заполняются в GUI); кейсы — из
   конфига (авто `case_id`, резолв `paper_id`→title/year, дефолты для опущенных полей).
3. Валидирует по логике cell 8 блокнота (`requiredMissing`, enum, дубли id, длина prompt)
   и пишет `validation/VALIDATION_LOG.txt` + распределение страт.
4. Генерирует веб-форму из подлинного шаблона блокнота (cell 8), предзагружая её
   манифестом, **убирает авто-создание стартовых кейсов** (форма открывается пустой) и
   встраивает серверный мост для [`../web_form/serve.py`](../web_form/) (кнопка
   «Сохранить на сервер» + подгрузка манифеста с диска).

## Ключевые принципы
- **Схема — из блокнота.** Правило готовности кейса в точности как в форме (cell 8):
  непусты `paper_title`, `creator_prompt`, `creator_rationale`, `review_focus`≥1 и хотя
  бы одно `match.*`.
- **Никакой сети.** Билдер не скачивает статьи и ничего не проверяет в интернете —
  источник данных только `config.yaml`.

## Зависимости
PyYAML (`scripts/requirements.txt`). Остальное — стандартная библиотека Python 3.
