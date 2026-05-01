# Исправление VLM-полей для SFT и GRPO

## Что было не так

Экспорт создавал legacy-поля:

- `chat` для SFT;
- `prompt_chat` для GRPO.

Внутри `chat.messages` / `prompt_chat.messages` могли быть image-блоки с путями к картинкам, но в top-level JSONL не было стабильной колонки `images`. Из-за этого train-скрипты не видели датасет как VLM-датасет и могли уходить в text-only режим.

`MAX_MULTIMODAL_RECORDS_PER_SAMPLE` не является переключателем картинок. Этот параметр ограничивает текстовые multimodal-summary records, которые добавляются в prompt. Реальные image attachments ограничиваются `MAX_IMAGES_PER_SAMPLE`.

## Что исправлено

### `src/scireason/scidatapipe_bridge/builder.py`

- Для `sft.jsonl` добавляется TRL-совместимое поле `messages`.
- Для `grpo.jsonl` добавляется TRL-совместимое поле `prompt`.
- Для обоих JSONL добавляется top-level поле `images`.
- Image-блоки внутри сообщений приводятся к placeholder-формату `{"type": "image"}`.
- Пути к картинкам в `images` приводятся к переносимому относительному виду `assets/...`, чтобы они работали после upload/download export-папки.
- Legacy-поля `chat` и `prompt_chat` сохранены для обратной совместимости.
- В `export_summary.json` добавлены счетчики строк/ссылок с картинками.

### `experiments/vlm_finetuning/scripts/train_vlm_sft.py`

- Скрипт понимает и новый формат `messages`, и legacy `chat.messages`.
- По умолчанию ожидает `--image-column images`.
- Извлекает embedded image paths из legacy image-блоков.
- Поддерживает смешанный датасет: VLM-строки с картинками + text-only строки с `images=[]`.

### `experiments/vlm_finetuning/scripts/train_vlm_grpo.py`

- Скрипт понимает и новый формат `prompt`, и legacy `prompt_chat.messages` / `prompt_messages`.
- По умолчанию ожидает `--image-column images`.
- Извлекает embedded image paths из legacy image-блоков.
- Не падает обратно в text-only режим только из-за смешанных строк.

### Блокнот Colab

- Убран hardcoded `HF_TOKEN`.
- Добавлена валидация VLM-полей после экспорта: проверяется наличие `images`, image placeholders и файлов в `assets`.
- В документации блока параметров уточнено, что картинки контролирует `MAX_IMAGES_PER_SAMPLE`, а не `MAX_MULTIMODAL_RECORDS_PER_SAMPLE`.

## Проверка

В репозитории выполнена синтаксическая проверка:

```bash
/usr/bin/python3 -m py_compile \
  src/scireason/scidatapipe_bridge/builder.py \
  experiments/vlm_finetuning/scripts/train_vlm_sft.py \
  experiments/vlm_finetuning/scripts/train_vlm_grpo.py
```

Полное обучение не запускалось: для этого нужна GPU-среда и установленные HF/VLM-зависимости.
