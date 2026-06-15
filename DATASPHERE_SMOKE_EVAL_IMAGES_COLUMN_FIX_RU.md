# Исправление smoke run: KeyError `images` на SFT evaluation

Дата: 2026-06-15

## Симптом

Новый smoke-прогон дошёл до конца SFT training loop: были выполнены все `20/20` шагов, после чего запуск упал уже на evaluation:

```text
KeyError: Caught KeyError in DataLoader worker process 0
...
images = [example["images"] for example in examples]
KeyError: 'images'
```

Это означает, что training dataset уже был корректен для TRL VLM collator, но eval dataset в одном из путей подготовки потерял top-level колонку `images`.

## Причина

В `maybe_prepare_dataset(..., requested_mode="vlm")` старый код всё равно удалял image columns, если в первых eval rows не было ненулевых изображений. Для VLM режима это неверно: текущий TRL VLM collator читает `example["images"]` и во время train, и во время eval. Text-only eval rows допустимы, но у них должна быть колонка:

```python
images = []
```

## Исправления

Изменены файлы:

- `experiments/vlm_finetuning/scripts/train_vlm_sft.py`
- `experiments/vlm_finetuning/scripts/train_vlm_grpo.py`
- `tests/test_vlm_training_format_normalization.py`

Добавлена функция `_ensure_trl_images_column(...)`, которая гарантирует наличие top-level `images` column в VLM режиме:

- если `images` уже есть — сохраняет её;
- если есть только `image` — создаёт `images` как list column;
- если изображений нет — добавляет `images=[]` для каждой строки;
- для `requested_mode="vlm"` больше не удаляет `images` только потому, что eval split оказался text-only.

Добавлены regression tests:

- `test_sft_eval_vlm_keeps_empty_images_column_for_text_only_eval`
- `test_grpo_eval_vlm_keeps_empty_images_column_for_text_only_eval`

## Локальные проверки

Выполнены:

```text
[OK] py_compile для SFT/GRPO/dataset/launcher scripts
[OK] bash -n для launch_examples.sh и datasphere/bin/*.sh
[OK] requirements.txt проходит packaging.Requirement
[OK] все DataSphere YAML загружаются через pyyaml
[OK] configs с requirements-file имеют cu121 extra-index-url
[OK] pytest: 20 passed
[OK] smoke dry-run строит корректную DataSphere CLI команду
```

## Следующая проверка

```bash
cd top-papers-graph-main
source .venv/bin/activate

export DATASPHERE_PROJECT_ID='bt18pnosk97i8n24ddnv'

datasphere project get --id "$DATASPHERE_PROJECT_ID"

bash experiments/vlm_finetuning/datasphere/launch_examples.sh hf-smoke-managed
```

Ожидается, что SFT теперь не упадёт на final eval с `KeyError: 'images'`, сохранит SFT adapter и перейдёт к GRPO smoke stage.
