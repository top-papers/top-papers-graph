# Исправление блокнота Task 3 (guarded)

Что исправлено:
- устранён блокирующий дефект в headless/smoke-режиме `TASK3_DUAL_NOTEBOOK_SMOKE=1`;
- раньше prefill из `TASK3_DUAL_NOTEBOOK_SMOKE_*` применялся слишком поздно — только в форме, тогда как ячейка установки запускалась **после** обязательной ранней валидации;
- из-за этого блокнот останавливался на ошибке вида:
  - `Запуск заблокирован: быстрый сетап не прошёл проверку`
  - `Для режима "Query + identifiers" укажите query, identifiers или оба варианта сразу.`

Что сделано:
- ранний quick-setup теперь подхватывает значения из `TASK3_DUAL_NOTEBOOK_SMOKE_*` **до** отдельной ячейки валидации;
- в smoke-режиме автоматически заполняются:
  - `query`
  - `processed_path`
  - `out_dir`
  - параметры обеих моделей
  - эксперт
  - безопасные дефолты для offline/smoke-прогона

Результат проверки:
- исправленный блокнот успешно проходит последовательность ячеек 2→7 в smoke-режиме;
- собраны артефакты blind A/B review:
  - `task3_dual_local_model_review_offline_ab.html`
  - `expert_dual_model_blind_review_bundle.zip`
  - `task3_dual_local_model_blind_key.json`
  - `variant_alpha/...`
  - `variant_beta/...`

Файл исправленного блокнота в архиве:
- `task3_dual_local_models_blind_ab_guarded_fixed.ipynb`
