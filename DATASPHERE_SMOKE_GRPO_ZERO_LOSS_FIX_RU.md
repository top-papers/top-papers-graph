# Исправление DataSphere smoke GRPO: zero-loss / all-clipped completions

## Симптом из `smoke_test_logs(10).txt`

Job DataSphere больше не падает: окружение создано, SFT и GRPO завершились, файлы скачались. Однако smoke GRPO фактически не обучался:

- `loss: 0.0`
- `grad_norm: 0.0`
- `completions/clipped_ratio: 1.0`
- `completions/mean_length: 32.0`
- `completions/mean_terminated_length: 0.0`
- `reward_std: 0.0`
- `frac_reward_zero_std: 1.0`

Причина: smoke-конфиг ограничивал GRPO completion до 32 токенов и одновременно включал `--mask-truncated-completions`. Все генерации обрезались до лимита, не завершались EOS/JSON, получали одинаковые strict-negative rewards, а TRL GRPO нормализовал группу в нулевое advantage. В итоге шаги выполнялись, но градиент был нулевой.

## Что изменено

1. `experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_smoke_g2_2.yaml`
   - `GRPO_MAX_COMPLETION_LENGTH: 32` заменён на `128`.
   - добавлен `GRPO_MASK_TRUNCATED_COMPLETIONS: 0` для smoke, чтобы короткий диагностический run не превращался в all-masked loss.

2. `experiments/vlm_finetuning/datasphere/bin/run_hf_top_papers_sft_grpo_full.sh`
   - `--mask-truncated-completions` теперь управляется переменной `GRPO_MASK_TRUNCATED_COMPLETIONS`.
   - full-конфиг сохраняет безопасный default `1`; smoke-конфиг явно ставит `0`.

3. `experiments/vlm_finetuning/scripts/train_vlm_grpo.py`
   - strict JSON rewards дополнены dense shaping для частично сгенерированного/обрезанного JSON.
   - malformed JSON остаётся отрицательным, но разные partial outputs получают разные отрицательные оценки, чтобы smoke/debug GRPO имел reward-дисперсию и ненулевой learning signal.

4. `experiments/vlm_finetuning/datasphere/bin/check_gpu_before_pipeline.py`
   - preflight больше не печатает промежуточный `"ok": false` перед финальной проверкой CUDA/BF16. В логе будет выводиться фактический статус.

5. Regression tests
   - добавлены проверки smoke-конфига против all-masked GRPO.
   - добавлены тесты dense reward shaping для truncated JSON.

## Проверки

Выполнены локально:

```bash
python -m compileall experiments/vlm_finetuning/scripts experiments/vlm_finetuning/datasphere
bash -n experiments/vlm_finetuning/datasphere/launch_examples.sh experiments/vlm_finetuning/datasphere/bin/*.sh
pytest -q tests/test_datasphere_job_configs.py tests/test_vlm_training_format_normalization.py
```

Результат targeted regression suite: `30 passed`.

Полный `pytest -q` в этой среде был запущен, но не завершился до таймаута контейнера; до таймаута успел пройти значительный префикс suite без новых failures. Поэтому для этого патча зафиксированы targeted regression-проверки по изменённым компонентам.
