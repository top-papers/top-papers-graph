# DataSphere DPO CUDA OOM memory guard fix

## Симптом

Новый полный прогон прошёл предыдущие DPO-блокеры: dataset build, leakage/full-data gates, text SFT, VLM SFT, DPO image placeholder alignment и singleton `image` column guard. Затем DPO стартовал и выполнил первые optimizer steps, но упал на rank0 внутри TRL `DPOTrainer`:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 12.14 GiB
...
trl/trainer/dpo_trainer.py", line 1418, in _compute_loss
    lse2 = torch.logsumexp(2.0 * shift_logits, dim=-1)
```

В этом прогоне DPO всё ещё использовал training-time cap `DPO_TRAIN_MAX_IMAGES_PER_EXAMPLE=3` и общий `VLM_MAX_PIXELS=1003520`. Для отдельных preference rows это всё ещё создаёт очень длинные multimodal sequences. TRL при DPO loss материализует logits по длине последовательности и словарю, поэтому редкий длинный VLM пример может потребовать дополнительный многогигабайтный tensor даже на g2.2 / 80GB GPU.

## Исправления

### 1. Более строгая DPO-only image projection

Raw JSONL, full-data audit и source coverage по-прежнему сохраняют все строки и все image refs. Изменена только training projection для DPO:

- `DPO_TRAIN_MAX_IMAGES_PER_EXAMPLE` теперь по умолчанию `1`, а не `3`.
- Добавлен отдельный `DPO_MAX_PIXELS`, чтобы DPO не наследовал более дорогой VLM-SFT pixel budget.
- Production configs выставляют `DPO_MAX_PIXELS=501760`.

### 2. DPO text-surface guard

Добавлен `--max-text-chars` в `train_vlm_dpo.py`. Guard отбрасывает только патологически длинные DPO preference rows по сумме текстовой поверхности `prompt + chosen + rejected` до TRL collation. Это безопаснее generic `max_length` для VLM, потому что generic truncation может удалить image tokens, не удалив соответствующие pixel inputs.

Production default:

```bash
DPO_MAX_TEXT_CHARS=12000
```

### 3. Periodic CUDA cache cleanup

Добавлен `--torch-empty-cache-steps` и прокидывание в `DPOConfig` при поддержке установленной TRL/Transformers версии. Production default:

```bash
DPO_TORCH_EMPTY_CACHE_STEPS=5
```

Это не заменяет уменьшение sequence/image budget, но снижает риск накопления allocator pressure между optimizer steps.

## Затронутые файлы

- `experiments/vlm_finetuning/scripts/train_vlm_dpo.py`
- `experiments/vlm_finetuning/datasphere/bin/run_hf_top_papers_sft_dpo_grpo_v2.sh`
- `experiments/vlm_finetuning/datasphere/bin/run_hf_top_papers_sft_grpo_full.sh`
- `experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_dpo_grpo_v2_g2_2.yaml`
- `experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_full_g2_2.yaml`
- `tests/test_vlm_dpo_attention_resolution.py`
- `tests/test_datasphere_job_configs.py`
- `CHANGELOG.md`

## Локальная проверка

```text
python -m py_compile experiments/vlm_finetuning/scripts/train_vlm_dpo.py tests/test_vlm_dpo_attention_resolution.py tests/test_datasphere_job_configs.py
python -m pytest -q tests/test_vlm_dpo_attention_resolution.py tests/test_datasphere_job_configs.py
# 37 passed

python -m pytest -q tests/test_vlm_dpo_attention_resolution.py \
  tests/test_vlm_training_format_normalization.py \
  tests/test_scireason_alignment_dataset_v2.py \
  tests/test_audit_full_data_usage.py \
  tests/test_datasphere_job_configs.py
# 78 passed
```

## Что проверить в следующем DataSphere retry

1. В DPO logs должны появиться defaults:
   - cap to at most 1 image;
   - `DPO_MAX_PIXELS=501760` through processor args;
   - text filter stats in `outputs/..._dpo_lora/run_config.json`.
2. DPO должен пройти дальше шага 6, где раньше был OOM.
3. Если OOM повторится на другом длинном example, первым дальнейшим рычагом будет снижение `DPO_MAX_TEXT_CHARS` до `8000` или `DPO_MAX_PIXELS` до `401408`, а не изменение raw dataset.
