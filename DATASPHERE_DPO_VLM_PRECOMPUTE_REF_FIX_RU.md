# Исправление DataSphere VLM DPO `precompute_ref_log_probs`

Дата: 2026-06-23

## Симптом

Новый DataSphere smoke/full-прогон прошёл дальше загрузки Qwen3-VL модели, но упал на создании `DPOTrainer`:

```text
ValueError: `precompute_ref_log_probs=True` is not supported for vision datasets.
Set `precompute_ref_log_probs=False`.
```

Падение происходило в `experiments/vlm_finetuning/scripts/train_vlm_dpo.py` после подготовки мультимодального DPO датасета.

## Причина

В full DataSphere wrapper по умолчанию добавлялся флаг `--precompute-ref-log-probs`. Это полезная оптимизация для text-only DPO, но несовместима с VLM DPO: TRL обрабатывает изображения на лету и не поддерживает предварительный полный forward reference-модели по vision dataset.

## Что изменено

1. В `train_vlm_dpo.py` добавлен `resolve_precompute_ref_log_probs(args, actual_mode)`.
2. Если фактический режим подготовки датасета — `vlm`, скрипт принудительно выставляет `precompute_ref_log_probs=False` и печатает диагностическое сообщение.
3. `precompute_ref_batch_size` теперь передаётся в `DPOConfig` только когда precompute реально включён.
4. В `experiments/vlm_finetuning/datasphere/bin/run_hf_top_papers_sft_dpo_grpo_v2.sh` default изменён с `DPO_PRECOMPUTE_REF_LOG_PROBS=1` на `0`.
5. Добавлены regression tests:
   - `tests/test_vlm_dpo_attention_resolution.py::test_dpo_disables_precompute_ref_log_probs_for_vlm`
   - `tests/test_datasphere_job_configs.py::test_v2_wrapper_does_not_precompute_dpo_ref_log_probs_by_default_for_vlm`

## Проверки

```bash
python -m py_compile experiments/vlm_finetuning/scripts/train_vlm_dpo.py \
  tests/test_vlm_dpo_attention_resolution.py \
  tests/test_datasphere_job_configs.py

python -m pytest -q \
  tests/test_vlm_dpo_attention_resolution.py \
  tests/test_vlm_training_format_normalization.py \
  tests/test_scireason_alignment_dataset_v2.py \
  tests/test_audit_full_data_usage.py \
  tests/test_datasphere_job_configs.py
```

Результат: `69 passed`.

Полный `pytest -q` был запущен дополнительно и остановлен по таймауту окружения; до таймаута assertion-падений не было.

## Ожидаемый эффект

Следующий DataSphere прогон должен пройти DPOTrainer initialization для VLM DPO и продолжить actual DPO training вместо падения на `precompute_ref_log_probs=True`.
