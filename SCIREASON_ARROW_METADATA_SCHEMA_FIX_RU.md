# Исправление падения SFT на Arrow schema inference в `datasets.load_dataset('json')`

## Симптом

Managed `hf-full-managed` запуск проходил build-аудиты и падал на старте `train_vlm_sft.py` при загрузке `sft_text_train.jsonl`:

```text
DatasetGenerationError: An error occurred while generating the dataset
TypeError: Couldn't cast array of type struct<... graph_kind: string, importance_score: double> to ... metadata schema
```

## Причина

`datasets.load_dataset('json')` строит Arrow-схему по JSONL-строкам. В full-data v2 артефактах поле `metadata` intentionally богаче и неоднороднее: в разных строках появляются дополнительные ключи (`graph_kind`, `importance_score`) и разный тип `extra`. При strict cast Arrow пытается привести поздние строки к уже выведенной схеме и падает до старта обучения.

## Исправление

Добавлены loose JSON/JSONL loaders в training entrypoints:

- `train_vlm_sft.py`:
  - `_load_sft_json_dataset_loose()` читает JSONL через Python `json`, нормализует строку в TRL SFT формат и создаёт `Dataset` только из стабильных колонок `messages/images/image`.
- `train_vlm_dpo.py`:
  - `_load_dpo_json_dataset_loose()` заранее нормализует `prompt/chosen/rejected/images/image`, не пропуская raw metadata в Arrow.
- `train_vlm_grpo.py`:
  - `_load_grpo_json_dataset_loose()` сохраняет `prompt/images/image`, а вложенные scalar target поля сериализует в JSON-строки, чтобы reward functions продолжали парсить их без неоднородной Arrow-схемы.

Raw full-data artifacts и аудиты не меняются: все строки и image refs остаются сохранены в builder outputs. Исправление касается только training-time загрузки JSONL в HF `Dataset`.

## Проверки

```text
py_compile: OK
bash -n launch_examples.sh: OK
bash -n run_hf_top_papers_sft_dpo_grpo_v2.sh: OK
pytest: 193 passed, 6 skipped, 13 warnings
zip integrity: OK
```
