# Исправление формата датасета для SFT/GRPO и VLM-картинок

## Что было найдено

`export-scidatapipe` собирает JSONL в scidatapipe-style формате:

- SFT: история диалога лежит в `chat.messages`, а не в top-level `messages`.
- GRPO: prompt лежит в `prompt_chat.messages`, а не в top-level `prompt`.
- Изображения лежат внутри message content-блоков как `{"type": "image", "image": ".../path.png"}`.

Для TRL это нужно привести к формату trainer'ов:

- SFT: top-level `messages`.
- GRPO: top-level `prompt`.
- VLM: top-level `image` или `images` + image placeholder в `content`, то есть `{"type": "image"}`.

## Что исправлено

### `experiments/vlm_finetuning/scripts/train_vlm_sft.py`

Добавлена нормализация входных JSONL строк:

- `chat.messages` -> `messages`;
- поддержка уже готового `messages` сохранена;
- content-блоки чистятся до TRL-совместимого вида;
- пути из `{"type": "image", "image": ...}` выносятся в колонку `images`;
- image-блоки в сообщениях заменяются на placeholder `{"type": "image"}`;
- если в датасете есть изображения, `train-mode=auto` выбирает `vlm`, а не текстовый режим;
- добавлена поддержка `--images-column`, по умолчанию `images`.

### `experiments/vlm_finetuning/scripts/train_vlm_grpo.py`

Аналогичная нормализация для GRPO:

- `prompt_chat.messages` -> `prompt`;
- `prompt_messages` также поддерживается;
- embedded image paths выносятся в `images`;
- убран хрупкий ручной `features` mapping, который мог ломаться на вложенных структурах;
- `train-mode=auto` выбирает VLM-режим при наличии image records;
- добавлена поддержка `--images-column`, по умолчанию `images`.

### Colab notebook

В исправленном блокноте изменён дефолт:

```python
MAX_MULTIMODAL_RECORDS_PER_SAMPLE = 8
```

Важно: в текущем CLI значение `0` означает не «выключить мультимодальность», а «добавить все текстовые multimodal records». Реальные картинки управляются параметром:

```python
MAX_IMAGES_PER_SAMPLE = 8
```

То есть проблема игнорирования картинок была не в `MAX_MULTIMODAL_RECORDS_PER_SAMPLE = 0`, а в том, что train-скрипты не поднимали embedded image paths в TRL-совместимую колонку `images` и поэтому могли уходить в text mode.

## Проверка

Проведены проверки:

- `py_compile` для изменённых train-скриптов;
- smoke-тест нормализации: `chat.messages` превращается в `messages`, `prompt_chat.messages` превращается в `prompt`, а image path выносится в `images`.
