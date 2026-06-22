# Исправление strict full-data audit для DPO dedupe/source coverage

## Симптом

`bash experiments/vlm_finetuning/datasphere/launch_examples.sh hf-full-managed` запускал v2 full pipeline и корректно собирал данные, но завершался до старта обучения на `audit_full_data_usage.py --strict`.

В логе было:

```json
"dpo_all": 4467,
"raw_sft_rows_total": 2548,
"dpo_from_sft": 2548,
"dpo_from_grpo": 1960,
"dpo_all_covers_sft_sources": {
  "ok": false,
  "observed": {
    "dpo_all_pairs": 4467,
    "sft_sources_covered": 2507,
    "dpo_from_sft_summary": 2548
  }
}
```

## Причина

После включения robust mixed DPO builder создаёт несколько hard-negative preference pairs из SFT и дополнительные GRPO-derived target-bootstrap pairs. Затем `dedupe_dpo_rows()` удаляет полностью совпадающие preference pairs.

Удаление дублей корректно для обучения, но старая реализация теряла `metadata.source_id` для SFT rows, чьи пары были объединены с уже существующей парой. Поэтому strict audit видел только 2507 уникальных `source_id` вместо 2548, хотя summary builder'а показывал, что DPO pairs были сгенерированы для всех 2548 SFT rows.

## Исправление

1. `dedupe_dpo_rows()` теперь не просто выбрасывает duplicate pair, а объединяет source metadata:
   - `metadata.source_ids` содержит все source ids, которые породили данный deduped pair;
   - `metadata.deduped_source_count` показывает число объединённых источников;
   - `metadata.pair_types` сохраняет типы hard-negative пар.

2. `audit_full_data_usage.py` теперь читает не только `metadata.source_id`, но и `metadata.source_ids`.

3. DPO full-data gate проверяет покрытие SFT sources после dedupe, а не one-to-one равенство pair count.

## Затронутые файлы

- `experiments/vlm_finetuning/scripts/build_scireason_alignment_datasets.py`
- `experiments/vlm_finetuning/scripts/audit_full_data_usage.py`
- `tests/test_scireason_alignment_dataset_v2.py`
- `tests/test_audit_full_data_usage.py`

## Проверки

- `py_compile`: OK
- `bash -n launch_examples.sh`: OK
- `bash -n run_hf_top_papers_sft_dpo_grpo_v2.sh`: OK
- `bash -n run_hf_top_papers_sft_grpo_full.sh`: OK
- `hf-full-managed --dry-run --no-download`: OK
- pytest split run: 190 passed, 3 skipped

## Ожидаемый эффект

Следующий `hf-full-managed` запуск должен пройти strict full-data audit и перейти к обучающим стадиям:

```text
text-SFT -> VLM-SFT -> robust mixed DPO -> optional GRPO polish -> HF upload
```
