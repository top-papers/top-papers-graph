# DataSphere DPO singleton `image` column fix

## Симптом

Последний DataSphere-прогон прошёл дальше предыдущего DPO image-placeholder fix, но снова упал в `DPOTrainer` на первом батче:

```text
ValueError: Number of images provided (1) does not match number of image placeholders (3).
```

Это означает, что prompt уже был выровнен до трёх `{"type":"image"}` blocks после `max_images_per_example=3`, но в TRL collator попал только один image object.

## Причина

В DPO dataset одновременно оставались два столбца:

- `images` — plural list, например 3 изображения после memory cap;
- `image` — legacy singleton fallback, равный первому изображению.

TRL vision DPO collator имеет специальную ветку совместимости: если в examples есть `image`, он делает `example["images"] = [example.pop("image")]`. Поэтому plural `images` незаметно схлопывался до одного изображения, а prompt сохранял 3 placeholders.

## Исправление

В `experiments/vlm_finetuning/scripts/train_vlm_dpo.py` plural-image path теперь удаляет singleton-столбец `image` перед передачей dataset в `DPOTrainer`, если фактическая VLM-колонка — `images`.

Это сохраняет multi-image DPO examples и предотвращает повторную рассинхронизацию `images`/placeholders внутри TRL.

## Регрессии

Добавлены тесты:

- `test_dpo_vlm_plural_images_path_drops_singleton_image_column`;
- `test_dpo_singleton_image_column_would_collapse_plural_images_without_guard`.

## Проверки

```bash
python -m py_compile experiments/vlm_finetuning/scripts/train_vlm_dpo.py tests/test_vlm_dpo_attention_resolution.py
pytest -q tests/test_vlm_dpo_attention_resolution.py tests/test_datasphere_job_configs.py
pytest -q tests/test_vlm_dpo_attention_resolution.py tests/test_vlm_training_format_normalization.py tests/test_scireason_alignment_dataset_v2.py tests/test_audit_full_data_usage.py tests/test_datasphere_job_configs.py
```
