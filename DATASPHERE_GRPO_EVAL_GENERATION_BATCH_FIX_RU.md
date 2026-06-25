# DataSphere GRPO eval generation batch divisibility fix

## Симптом

Новый DataSphere прогон прошёл экспорт, quality gates, SFT и DPO, после чего упал уже на старте GRPO при создании `GRPOConfig`:

```text
ValueError: The global eval batch size (1 * 2) must be divisible by the number of generations used for evaluation (4).
```

## Причина

`run_hf_top_papers_sft_dpo_grpo_v2.sh` запускал GRPO с `--per-device-eval-batch-size 1` на двух процессах `torchrun`, поэтому глобальный eval batch равен 2. При этом V2 job config задавал `GRPO_NUM_GENERATIONS_EVAL=4`. TRL валидирует, что число eval-generations делит global eval batch, поэтому конфиг был несовместим с g2.2, хотя train batch для `GRPO_NUM_GENERATIONS=4` оставался валидным за счёт gradient accumulation.

## Исправление

- В `train_vlm_grpo.py` добавлен runtime guard `resolve_grpo_generation_batch_divisibility(...)`:
  - сохраняет минимум 2 генерации для train/eval, когда это возможно;
  - проверяет train effective batch: `WORLD_SIZE * per_device_train_batch_size * gradient_accumulation_steps`;
  - проверяет eval global batch: `WORLD_SIZE * per_device_eval_batch_size`;
  - при несовместимом eval значении безопасно понижает `num_generations_eval` до крупнейшего делителя batch size;
  - если eval batch слишком мал даже для 2 generations, отключает eval split вместо падения до старта GRPO.
- В V2 wrapper default изменён с `GRPO_NUM_GENERATIONS_EVAL=4` на `2`.
- В `hf_top_papers_sft_dpo_grpo_v2_g2_2.yaml` выставлено `GRPO_NUM_GENERATIONS_EVAL: 2`.
- Добавлены regression tests на runtime guard и DataSphere defaults.

## Почему это безопасно

Это не меняет raw/export dataset и не затрагивает DPO. Изменяется только GRPO eval sampling group size, чтобы он соответствовал фактическому global eval batch на g2.2. Train group size остаётся 4, потому что effective train batch с gradient accumulation делится на 4.

## Проверка

```bash
python -m py_compile experiments/vlm_finetuning/scripts/train_vlm_grpo.py \
  tests/test_vlm_training_format_normalization.py \
  tests/test_datasphere_job_configs.py

pytest -q tests/test_vlm_training_format_normalization.py tests/test_datasphere_job_configs.py
```
