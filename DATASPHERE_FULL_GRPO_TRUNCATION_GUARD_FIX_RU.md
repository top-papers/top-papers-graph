# Проверка full-run GRPO: защита от all-masked truncated completions

## Что проверено

Проверен полный DataSphere путь запуска:

- `experiments/vlm_finetuning/datasphere/launch_examples.sh`
- `experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_full_g2_2.yaml`
- `experiments/vlm_finetuning/datasphere/bin/run_hf_top_papers_sft_grpo_full.sh`
- `experiments/vlm_finetuning/datasphere/TUTORIAL_FULL_EXPERIMENT_RU.md`
- `experiments/vlm_finetuning/scripts/train_vlm_grpo.py`

## Найденный риск

В full YAML completion не был ограничен 32 токенами: там стояло `GRPO_MAX_COMPLETION_LENGTH: 384`. Поэтому точной smoke-ошибки `max_completion_length=32 + mask_truncated_completions` в full config не было.

Но full wrapper включал `--mask-truncated-completions` по умолчанию через `GRPO_MASK_TRUNCATED_COMPLETIONS:-1`, а full YAML не задавал эту переменную явно. Кроме того, tutorial для OOM-сценариев рекомендовал `GRPO_MAX_COMPLETION_LENGTH=32`, что могло воспроизвести zero-loss/all-clipped режим в полном прогоне при ручной настройке.

## Исправления

1. `hf_top_papers_sft_grpo_full_g2_2.yaml`
   - `GRPO_MAX_COMPLETION_LENGTH: 384` заменён на `512`.
   - добавлен явный `GRPO_MASK_TRUNCATED_COMPLETIONS: 0`.

2. `run_hf_top_papers_sft_grpo_full.sh`
   - default для `GRPO_MASK_TRUNCATED_COMPLETIONS` изменён с `1` на `0`.
   - fallback для `GRPO_MAX_COMPLETION_LENGTH` изменён с `384` на `512`.

3. `train_vlm_grpo.py`
   - CLI default `--max-completion-length` изменён с `384` на `512`.

4. `TUTORIAL_FULL_EXPERIMENT_RU.md`
   - OOM-рекомендация больше не предлагает `GRPO_MAX_COMPLETION_LENGTH=32`.
   - добавлено предупреждение не совмещать короткий лимит completion с masking truncated completions.

5. Regression tests
   - добавлена проверка, что full config использует `GRPO_MAX_COMPLETION_LENGTH >= 512` и отключает `GRPO_MASK_TRUNCATED_COMPLETIONS`.
   - добавлена проверка, что tutorial больше не рекомендует `GRPO_MAX_COMPLETION_LENGTH=32`.

## Проверки

```bash
python -m compileall -q experiments/vlm_finetuning/scripts experiments/vlm_finetuning/datasphere
bash -n experiments/vlm_finetuning/datasphere/launch_examples.sh experiments/vlm_finetuning/datasphere/bin/*.sh
pytest -q tests/test_datasphere_job_configs.py tests/test_vlm_training_format_normalization.py
```

Результат: `32 passed`.
