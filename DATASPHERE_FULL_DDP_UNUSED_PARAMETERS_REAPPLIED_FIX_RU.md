# Исправление full-run падения Qwen3-VL SFT на DDP unused parameters

Дата: 2026-06-18.

## Симптом

Full DataSphere job `hf_top_papers_sft_grpo_full_g2_2.yaml` успешно подготовил полный HF export dataset и начал SFT под `torchrun --nproc_per_node=2`, но команда была построена с явным отключением unused-parameter detection:

```text
--gradient-checkpointing --no-ddp-find-unused-parameters ...
```

На первом training step rank 1 упал:

```text
RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one.
This error indicates that your module has parameters that were not used in producing loss.
... enable unused parameter detection by passing ... find_unused_parameters=True
```

Датасетные guard-ы при этом отработали нормально: были отброшены длинные текстовые rows, image refs не отсутствовали, модель и LoRA загрузились. Это отделяет текущую проблему от предыдущего DDP/NCCL straggler-fix.

## Причина

В текущем архиве full/smoke DataSphere configs задавали:

```yaml
SFT_DDP_FIND_UNUSED_PARAMETERS: 0
GRPO_DDP_FIND_UNUSED_PARAMETERS: 0
```

Wrapper корректно интерпретировал `0` как явный `--no-ddp-find-unused-parameters`, поэтому train script передавал в Trainer/SFTConfig `ddp_find_unused_parameters=False`. Для Qwen3-VL + LoRA + VLM batches под DDP это небезопасно: отдельные trainable ветки могут не участвовать в loss на конкретном rank/step.

## Исправление

1. В `train_vlm_sft.py` и `train_vlm_grpo.py` default resolver снова стал безопасным для distributed VLM:

```python
return actual_mode == 'vlm' and get_world_size() > 1
```

Явный CLI override сохранён: `--ddp-find-unused-parameters` включает, `--no-ddp-find-unused-parameters` выключает.

2. В full и smoke DataSphere configs включено явное безопасное поведение:

```yaml
SFT_DDP_FIND_UNUSED_PARAMETERS: 1
GRPO_DDP_FIND_UNUSED_PARAMETERS: 1
```

Это гарантирует, что managed wrapper будет строить команду с `--ddp-find-unused-parameters`, а не с `--no-ddp-find-unused-parameters`.

3. Обновлены regression tests:

- distributed VLM default теперь `True`;
- text-only и single-process VLM остаются `False`;
- explicit on/off overrides продолжают работать;
- full config проверяется на `SFT_DDP_FIND_UNUSED_PARAMETERS=1` и `GRPO_DDP_FIND_UNUSED_PARAMETERS=1`.

## Изменённые файлы

- `experiments/vlm_finetuning/scripts/train_vlm_sft.py`
- `experiments/vlm_finetuning/scripts/train_vlm_grpo.py`
- `experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_full_g2_2.yaml`
- `experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_smoke_g2_2.yaml`
- `tests/test_vlm_training_format_normalization.py`
- `tests/test_datasphere_job_configs.py`

## Проверки

Локально выполнено:

```text
python3 -m py_compile experiments/vlm_finetuning/scripts/train_vlm_sft.py experiments/vlm_finetuning/scripts/train_vlm_grpo.py
python3 -m compileall -q experiments/vlm_finetuning/scripts experiments/vlm_finetuning/datasphere
bash -n experiments/vlm_finetuning/datasphere/launch_examples.sh experiments/vlm_finetuning/datasphere/bin/*.sh
python3 yaml parse для всех experiments/vlm_finetuning/datasphere/job_configs/*.yaml
pytest -q tests/test_datasphere_job_configs.py tests/test_vlm_training_format_normalization.py tests/test_hf_export_smoke_sampling.py
```

Результат targeted suite:

```text
46 passed in 1.39s
```

Также запускался полный `pytest -q`, но в этой песочнице он не завершился за 300 секунд; поэтому валидация зафиксирована по затронутым regression suites и синтаксическим проверкам.

## Ожидаемое поведение следующего run

Следующий full/smoke launch должен печатать в SFT-команде:

```text
--ddp-find-unused-parameters
```

а `run_config.json` должен содержать:

```json
"ddp_find_unused_parameters_resolved": true
```

Если после этого появится новая ошибка, она уже будет следующим independent failure после прохождения текущего DDP unused-parameters crash.
