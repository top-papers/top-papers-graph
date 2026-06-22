# Исправление full-data audit для DPO-first пайплайна

## Причина падения

Последний managed запуск `hf-full-managed` корректно собрал v2 alignment dataset и включил DPO-first pipeline, но остановился на `audit_full_data_usage.py --strict --require-all-images` до старта обучения.

В логе было:

```json
"dpo_all": 4467,
"raw_sft_rows_total": 2548,
"dpo_from_sft": 2548,
"dpo_from_grpo": 1960
```

При этом старый audit всё ещё проверял:

```python
counts["dpo_all"] == raw_sft
```

Эта проверка устарела после включения hard-negative mining и GRPO-derived preference pairs. Теперь `dpo_all.jsonl` является набором preference-пар, а не one-to-one копией SFT rows. Поэтому `dpo_all` должен быть `>= raw_sft` и может включать дополнительные пары из GRPO targets.

## Исправление

В `experiments/vlm_finetuning/scripts/audit_full_data_usage.py` проверка заменена на покрытие источников:

- `dpo_all_pairs >= raw_sft_rows_total`;
- все `id` из `sft_all.jsonl` должны присутствовать как `metadata.source_id` в `dpo_all.jsonl`;
- если source-id недоступны, используется fallback на `summary.counts.dpo_from_sft >= raw_sft_rows_total`.

Новая проверка называется:

```text
dpo_all_covers_sft_sources
```

Она принимает валидные DPO-first датасеты вида:

```text
DPO pairs = SFT-derived pairs + GRPO-derived target-bootstrap pairs + hard negatives
```

и больше не блокирует запуск полного цикла `text-SFT -> VLM-SFT -> robust mixed DPO -> optional GRPO polish`.

## Проверки

Добавлен regression-test:

```text
tests/test_audit_full_data_usage.py::test_full_data_audit_accepts_extra_dpo_hard_negative_pairs
```

Он воспроизводит случай, когда `dpo_all` содержит больше строк, чем `sft_all`, но покрывает все SFT source ids.

Проверено:

```text
python -m py_compile experiments/vlm_finetuning/scripts/audit_full_data_usage.py
bash -n experiments/vlm_finetuning/datasphere/launch_examples.sh
bash -n experiments/vlm_finetuning/datasphere/bin/run_hf_top_papers_sft_dpo_grpo_v2.sh
bash -n experiments/vlm_finetuning/datasphere/bin/run_hf_top_papers_sft_grpo_full.sh
DATASPHERE_PROJECT_ID=dummy bash experiments/vlm_finetuning/datasphere/launch_examples.sh hf-full-managed --dry-run --no-download
python -m pytest -q
```

Результат:

```text
188 passed, 6 skipped, 13 warnings
```
