# Исправление smoke: GRPO import падает на FSDPModule при torch 2.5.1+cu121

## Симптом

Новый smoke run успешно прошёл:

- DataSphere GPU preflight;
- dataset export + smoke sample caps;
- SFT training 20/20 steps;
- SFT eval;
- сохранение SFT LoRA adapter.

Падение началось на старте GRPO stage при импорте `GRPOTrainer`:

```text
ImportError: cannot import name 'FSDPModule' from 'torch.distributed.fsdp'
RuntimeError: Failed to import trl.trainer.grpo_trainer ... cannot import name 'FSDPModule'
```

## Причина

DataSphere окружение намеренно использует `torch==2.5.1+cu121`, потому что этот CUDA wheel совместим с NVIDIA driver `535.261.03` / CUDA `12.2` в текущем runtime. Но актуальный `trl` импортирует optional FSDP2 helper symbol `FSDPModule` из `torch.distributed.fsdp`. В torch 2.5.1 этот symbol отсутствует; legacy FSDP class называется `FullyShardedDataParallel`.

В нашем запуске используется DDP через `torchrun`, а не FSDP, поэтому падение происходит только на import side-effect, до реального GRPO training.

## Исправления

### 1. Compatibility shim в `train_vlm_grpo.py`

Перед импортом `GRPOConfig, GRPOTrainer` добавлена функция:

```python
install_torch_fsdp_module_import_compat()
```

Она проверяет `torch.distributed.fsdp`; если `FSDPModule` отсутствует, но есть legacy `FullyShardedDataParallel`, она создаёт alias:

```python
fsdp.FSDPModule = FullyShardedDataParallel
```

Это не включает FSDP и не меняет distributed strategy. Alias нужен только для успешного импорта optional TRL FSDP helpers.

### 2. Pin `trl` ниже будущей версии 1.7

В `experiments/vlm_finetuning/datasphere/requirements.txt` заменено:

```text
trl>=1.4.0,<2
```

на:

```text
trl>=1.4.0,<1.7
```

Причина: в smoke-логах `trl` уже предупреждает о поведенческом изменении SFT loss в 1.7. Для воспроизводимости pipeline фиксирует диапазон на версии, с которой текущий SFT path уже прошёл smoke.

### 3. Regression tests

Добавлены проверки:

- GRPO shim создаёт `FSDPModule` alias для torch 2.5-style FSDP namespace;
- DataSphere requirements закрепляет `trl>=1.4.0,<1.7`.

## Локальная проверка

Выполнено:

```text
[OK] py_compile для SFT/GRPO/dataset/launcher scripts
[OK] bash -n для launch_examples.sh и datasphere/bin/*.sh
[OK] requirements.txt проходит packaging.Requirement
[OK] все DataSphere YAML загружаются через pyyaml
[OK] configs с requirements-file имеют cu121 extra-index-url
[OK] pytest: 22 passed
[OK] smoke dry-run строит корректную DataSphere CLI команду
```

## Следующая проверка

```bash
cd top-papers-graph-main
source .venv/bin/activate

export DATASPHERE_PROJECT_ID='bt18pnosk97i8n24ddnv'

datasphere project get --id "$DATASPHERE_PROJECT_ID"

bash experiments/vlm_finetuning/datasphere/launch_examples.sh hf-smoke-managed
```

Ожидаемый следующий результат:

- SFT снова проходит 20/20 + eval;
- GRPO stage импортирует `GRPOTrainer` без `FSDPModule` crash;
- smoke доходит до первых GRPO generation/training steps.
