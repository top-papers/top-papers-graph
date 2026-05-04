# Исправление идентификации файлов по автору

## Что было

В формах/архивах могли встречаться несколько решений с одинаковым внутренним `submission_id`, например:

```text
reasoning_failures_in_large_language_models
```

При этом имя автора было доступно только во внешнем имени загруженного файла:

```text
reasoning_failures_in_large_language_models_Илья_Фёдоров.yaml
expert_validation_bundle - Илья Фёдоров.zip
```

После распаковки ZIP внутри оставался только `trajectory_submission_id=reasoning_failures_in_large_language_models`, поэтому нормализованный Task 2 каталог назывался без автора.

## Что изменено

1. `normalize_task1/normalizer.py` теперь добавляет авторский суффикс из имени YAML-файла в `submission_id`.
2. `normalize_task2/normalizer.py` теперь принимает `source_path` и `source_identity`, чтобы:
   - брать автора из исходного ZIP-файла;
   - не ломать обработку нескольких bundle внутри одного ZIP;
   - сохранять стабильную защиту от коллизий.
3. `builder.py` передаёт в Task 2 normalizer исходный путь ZIP и bundle-specific identity вида `<zip>!<bundle>`.
4. Standalone `build_vlm_sft_dataset.py` тоже добавляет авторский суффикс из имени YAML.
5. В notebook добавлен вывод `submission_id counts`, чтобы быстро проверить, как именно файлы попали в датасет.

## Проверенный кейс

Для входных файлов:

```text
reasoning_failures_in_large_language_models_Илья_Фёдоров.yaml
expert_validation_bundle - Илья Фёдоров.zip
```

экспорт теперь даёт единый идентификатор:

```text
reasoning_failures_in_large_language_models_Илья_Фёдоров
```

Проверочный прогон дал:

```json
{
  "trajectory_reasoning": 5,
  "assertion_reconstruction": 14,
  "assertion_review_rl": 8,
  "sft_rows": 19,
  "grpo_rows": 8
}
```

Все `sft.jsonl` и `grpo.jsonl` строки для этого кейса имеют `metadata.submission_id = reasoning_failures_in_large_language_models_Илья_Фёдоров`.
