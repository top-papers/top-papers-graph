# Исправление smoke-падения SFT на DDP unused parameters

## Симптом

Новый smoke-run проходит подготовку датасета, GPU preflight, загрузку Qwen3-VL и стартует SFT training, но падает на первом training step в DistributedDataParallel:

```text
RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one.
This error indicates that your module has parameters that were not used in producing loss.
...
You can enable unused parameter detection by passing the keyword argument `find_unused_parameters=True`
```

## Причина

Qwen3-VL + LoRA + multimodal batches под `torchrun --nproc_per_node=2` могут активировать не все trainable ветки модели на каждом rank/step. В таком случае DDP ожидает градиенты для параметров, которые на конкретном rank не участвовали в loss, и падает перед следующей итерацией.

## Исправление

В `train_vlm_sft.py` и `train_vlm_grpo.py` добавлен auto-resolver:

```python
def resolve_ddp_find_unused_parameters(args, actual_mode):
    requested = getattr(args, "ddp_find_unused_parameters", None)
    if requested is not None:
        return bool(requested)
    return actual_mode == "vlm" and get_world_size() > 1
```

Теперь для distributed VLM запусков `SFTConfig` / `GRPOConfig` получают:

```python
ddp_find_unused_parameters=True
```

В wrapper также явно добавлен флаг:

```bash
--ddp-find-unused-parameters
```

для SFT и GRPO stages.

## Изменённые файлы

- `experiments/vlm_finetuning/scripts/train_vlm_sft.py`
- `experiments/vlm_finetuning/scripts/train_vlm_grpo.py`
- `experiments/vlm_finetuning/datasphere/bin/run_hf_top_papers_sft_grpo_full.sh`
- `tests/test_vlm_training_format_normalization.py`
- `tests/test_datasphere_job_configs.py`

## Проверки

Локально выполнено:

```text
[OK] py_compile для SFT/GRPO/dataset/launcher scripts
[OK] bash -n для launch_examples.sh и datasphere/bin/*.sh
[OK] requirements.txt проходит packaging.Requirement
[OK] все DataSphere YAML загружаются через pyyaml
[OK] configs с requirements-file имеют cu121 extra-index-url
[OK] pytest: tests/test_vlm_training_format_normalization.py + tests/test_datasphere_job_configs.py = 16 passed
[OK] dry-run managed launcher строит корректную smoke-команду
```

## Следующая проверка

```bash
cd top-papers-graph-main
source .venv/bin/activate

export DATASPHERE_PROJECT_ID='bt18pnosk97i8n24ddnv'

datasphere project get --id "$DATASPHERE_PROJECT_ID"

bash experiments/vlm_finetuning/datasphere/launch_examples.sh hf-smoke-managed
```

Ожидаемый следующий этап: SFT должен пройти дальше `1/20` step без DDP unused-parameters crash.
