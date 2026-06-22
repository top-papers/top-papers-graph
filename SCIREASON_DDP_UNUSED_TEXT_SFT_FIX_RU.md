# Исправление DDP unused-parameters в text-SFT

## Симптом

Новый `hf-full-managed` прогон дошёл дальше предыдущих исправлений: оба preflight-аудита прошли, full-data audit прошёл, JSONL был успешно загружен через loose loader, затем начался `text-SFT`.

Падение произошло на первом training step в `train_vlm_sft.py`:

```text
RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one.
This error indicates that your module has parameters that were not used in producing loss.
...
Parameter indices which did not receive grad for rank 0: 0 1 2 ...
```

## Причина

Для text-SFT модель всё ещё создаётся как Qwen3-VL LoRA/adapter training. При `lora_target_modules=all-linear` часть trainable веток относится к multimodal/vision пути, но text-only batch не содержит image tensors. В DDP это означает, что некоторые trainable параметры на rank не участвуют в loss на данном шаге.

Старый resolver включал `ddp_find_unused_parameters=True` только для `actual_mode == "vlm"`, поэтому text-SFT в multi-GPU режиме шёл с быстрым, но неподходящим DDP default.

## Исправления

1. `experiments/vlm_finetuning/scripts/train_vlm_sft.py`
   - `resolve_ddp_find_unused_parameters()` теперь включает DDP unused-parameter detection для всех multi-process runs, включая `train_mode=text`.
   - Это закрывает text-SFT с Qwen3-VL LoRA targets.

2. `experiments/vlm_finetuning/scripts/train_vlm_dpo.py`
   - Добавлен CLI-флаг `--ddp-find-unused-parameters / --no-ddp-find-unused-parameters`.
   - Добавлен resolver с default `True` для `WORLD_SIZE > 1`.
   - `DPOConfig` теперь получает `ddp_find_unused_parameters`.
   - Добавлен `gradient_checkpointing_kwargs={"use_reentrant": False}`.

3. `experiments/vlm_finetuning/scripts/train_vlm_grpo.py`
   - Resolver приведён к той же политике: default `True` для всех multi-process runs.

4. `experiments/vlm_finetuning/datasphere/bin/run_hf_top_papers_sft_dpo_grpo_v2.sh`
   - Добавлен общий `DDP_UNUSED_ARGS`.
   - По умолчанию все стадии получают `--ddp-find-unused-parameters`.
   - Для ablation можно выключить через `DDP_FIND_UNUSED_PARAMETERS=0`.

5. `experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_dpo_grpo_v2_g2_2.yaml`
   - Добавлен `DDP_FIND_UNUSED_PARAMETERS: '1'`.
   - Добавлен `TORCH_DISTRIBUTED_DEBUG: INFO` для более информативной диагностики при будущих DDP edge cases.

## Проверки

```text
python -m py_compile train_vlm_sft.py train_vlm_dpo.py train_vlm_grpo.py
bash -n run_hf_top_papers_sft_dpo_grpo_v2.sh
bash -n launch_examples.sh
pytest tests/test_vlm_training_format_normalization.py -q
pytest -q
```

Результат полного набора:

```text
194 passed, 6 skipped, 13 warnings
```

## Ожидаемый эффект

Следующий `hf-full-managed` запуск должен пройти место падения в начале `text-SFT`. Если дальше появится новая ошибка, она уже будет относиться к следующей стадии фактического обучения, а не к DDP reducer на text-only LoRA batches.
