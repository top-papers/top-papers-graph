# Исправление: часть входных файлов не попадала в датасет

## Что было исправлено

1. **Colab notebook / curation step**
   - Последняя отправка теперь выбирается по участнику/строке формы (`ФИО`, `Username`, `Email` и аналоги), а не по имени загруженного файла.
   - Curated-файлы получают уникальное имя с `task`, ключом участника, timestamp и hash.
   - Копирование в `curated_input` больше не перезаписывает файлы с одинаковыми исходными именами (`trajectory.yaml`, `bundle.zip` и т.п.).
   - Добавлена проверка: количество файлов в `curated_input` должно совпадать с количеством успешно скачанных файлов.

2. **Task 1 normalizer**
   - Если несколько YAML имеют одинаковый `submission_id`, второй и последующие файлы больше не пропадают.
   - На коллизии добавляется стабильный суффикс `__input_<hash>`.
   - Исходный id сохраняется в `original_submission_id`.

3. **Task 2 normalizer**
   - Если несколько bundle имеют одинаковый `submission_id`, они больше не перезаписывают друг друга.
   - На коллизии добавляется стабильный суффикс `__input_<hash>`.

4. **Task 2 ZIP handling**
   - Если один ZIP содержит несколько Task 2 bundle-папок, exporter теперь обрабатывает все bundle, а не только первую найденную.

5. **Fine-tuning scripts**
   - `train_vlm_dpo.py` приведён к той же логике работы с `images`, что `train_vlm_sft.py` и `train_vlm_grpo.py`: смешанные text-only + image rows больше не фильтруются и не переводятся молча в text-only режим при наличии изображений.

## Проверки

```bash
python -m py_compile \
  src/scireason/scidatapipe_bridge/builder.py \
  src/scireason/scidatapipe_bridge/vendor/normalize_task1/normalizer.py \
  src/scireason/scidatapipe_bridge/vendor/normalize_task2/normalizer.py \
  experiments/vlm_finetuning/scripts/train_vlm_sft.py \
  experiments/vlm_finetuning/scripts/train_vlm_grpo.py \
  experiments/vlm_finetuning/scripts/train_vlm_dpo.py

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_scidatapipe_input_retention.py
```

Результат новой регрессии: `2 passed`.
