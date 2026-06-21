# Исправление runtime-регрессии GRPO reward normalization

## Симптом

Последний DataSphere-прогон успешно завершил SFT и сохранил best checkpoint, но упал сразу после старта GRPO на первом вызове reward-функции:

```text
NameError: name 'math' is not defined
```

Точка падения: `apply_task_aware_reward_processing()` в `experiments/vlm_finetuning/scripts/train_vlm_grpo.py`, строка с `math.tanh(...)`.

## Причина

В предыдущем патче был добавлен task-aware reward routing и robust group normalization. Код начал использовать `math.tanh`, но в `train_vlm_grpo.py` не был добавлен `import math`. `py_compile` не ловит такие ошибки, потому что имя разрешается только во время выполнения reward-функции.

## Исправление

- Добавлен `import math` в `train_vlm_grpo.py`.
- Добавлен regression-test `test_grpo_task_aware_reward_processing_imports_math_for_group_norm`, который вызывает `apply_task_aware_reward_processing(... normalize=True ...)` с повторяющимся `sample_id`, то есть проходит именно через ветку `math.tanh`.

## Ожидаемый эффект

Следующий запуск должен пройти дальше места падения и начать реальные GRPO training steps. Исправление не меняет формулу reward и не меняет датасет; оно устраняет runtime-регрессию в уже добавленной нормализации.

## Дополнительные наблюдения по логу

- SFT завершился успешно и выбрал best checkpoint `checkpoint-540`.
- Full raw export был собран корректно: `raw_sft_rows_total=2548`, `raw_grpo_rows_total=1960`, build-time image truncation отсутствует.
- В GRPO всё ещё используется training-time projection до 2 images/example, что оставляет запуск memory-safe на g2.2, но сокращает evidence coverage.
- Прогон всё ещё запущен через legacy `hf_top_papers_sft_grpo_full_g2_2.yaml`; для качества предпочтительнее v2 `SFT -> DPO -> optional GRPO`.
