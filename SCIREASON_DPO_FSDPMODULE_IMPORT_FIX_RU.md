# Исправление DPO import crash: FSDPModule compatibility

Дата: 2026-06-23

## Симптом

После исправления DDP unused-parameters `hf-full-managed` дошёл дальше: оба preflight-аудита прошли, `text-SFT` стартовал и начал реально обучаться. Затем пайплайн перешёл к DPO-стадии и упал ещё до инициализации `DPOTrainer`:

```text
ImportError: cannot import name 'FSDPModule' from 'torch.distributed.fsdp'
RuntimeError: Failed to import trl.trainer.dpo_trainer ... cannot import name 'FSDPModule'
```

## Причина

В текущем DataSphere окружении установлен torch build, где `torch.distributed.fsdp` содержит legacy `FullyShardedDataParallel`, но не экспортирует новый символ `FSDPModule`. При этом установленная версия TRL импортирует optional FSDP helpers при `from trl import DPOConfig, DPOTrainer`, даже если job реально использует DDP, а не FSDP.

Аналогичный compatibility shim уже был добавлен в `train_vlm_grpo.py`, но отсутствовал в `train_vlm_dpo.py`.

## Исправление

В `experiments/vlm_finetuning/scripts/train_vlm_dpo.py` добавлена функция:

```python
install_torch_fsdp_module_import_compat()
```

Она выполняется до `from trl import DPOConfig, DPOTrainer` и, если нужно, добавляет узкий alias:

```python
from torch.distributed.fsdp import FullyShardedDataParallel
setattr(fsdp, "FSDPModule", FullyShardedDataParallel)
```

Это не включает FSDP и не меняет training strategy. Alias нужен только для импортов optional TRL FSDP utilities. В v2 job по-прежнему используется DDP/LoRA.

## Тесты

Добавлен regression-test:

```text
test_dpo_installs_fsdpmodule_alias_before_trl_import
```

Проверки:

```text
py_compile: OK
bash -n launch_examples.sh: OK
bash -n run_hf_top_papers_sft_dpo_grpo_v2.sh: OK
targeted tests: 58 passed
```

Полный `pytest -q` был запущен, но в sandbox не завершился за 600 секунд на unrelated slow tests. Таргетированные тесты на затронутые VLM/DPO/DataSphere участки прошли.
