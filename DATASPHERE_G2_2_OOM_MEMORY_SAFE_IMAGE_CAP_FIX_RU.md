# Исправление CUDA OOM на full-data Qwen3-VL SFT в DataSphere g2.2

## Симптом

Новый full-data запуск дошёл до первого шага SFT и упал с:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 11.27 GiB.
GPU 0 has a total capacity of 79.33 GiB of which 10.83 GiB is free.
```

Перед падением датасет был собран без усечения изображений:

```json
"max_images_per_example_sft": 0,
"max_images_per_example_grpo": 0,
"sft.image_refs_before_cap": 3291,
"sft.image_refs_after_cap": 3291,
"grpo.image_refs_before_cap": 12046,
"grpo.image_refs_after_cap": 12046
```

То есть предыдущая правка корректно сохранила все raw image refs, но такой full-fidelity multimodal batch оказался слишком тяжёлым для g2.2.

## Причина

Qwen3-VL получает все изображения примера как top-level `images`. При `VLM_MAX_PIXELS=1003520` и `MAX_IMAGES_PER_EXAMPLE_SFT=0` отдельные примеры с несколькими evidence images создают слишком большой activation/vision-token footprint. `per_device_train_batch_size=1` уже минимален, поэтому уменьшать нужно не batch size, а training-time visual payload.

## Исправление

Сделан раздельный режим:

1. **Raw/full-data build остаётся полным**: `MAX_IMAGES_PER_EXAMPLE_SFT=0`, `MAX_IMAGES_PER_EXAMPLE_GRPO=0` по-прежнему сохраняют все строки и все image refs в derived artifacts/audit.
2. **Training projection для g2.2 ограничивает число изображений на пример**:
   - `SFT_TRAIN_MAX_IMAGES_PER_EXAMPLE=3`
   - `DPO_TRAIN_MAX_IMAGES_PER_EXAMPLE=3`
   - `GRPO_TRAIN_MAX_IMAGES_PER_EXAMPLE=2`
3. Добавлен флаг `--max-images-per-example` в:
   - `experiments/vlm_finetuning/scripts/train_vlm_sft.py`
   - `experiments/vlm_finetuning/scripts/train_vlm_dpo.py`
   - `experiments/vlm_finetuning/scripts/train_vlm_grpo.py`
4. После training-time image cap скрипты заново выравнивают Qwen image placeholders с длиной top-level `images`, чтобы TRL collator не падал на mismatch.
5. В DataSphere wrappers добавлен allocator guard:
   - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   - `PYTORCH_ALLOC_CONF=expandable_segments:True`

## Как отключить cap на более крупной GPU

Для full-fidelity обучения на конфигурации с большим запасом GPU RAM можно явно поставить:

```bash
SFT_TRAIN_MAX_IMAGES_PER_EXAMPLE=0
DPO_TRAIN_MAX_IMAGES_PER_EXAMPLE=0
GRPO_TRAIN_MAX_IMAGES_PER_EXAMPLE=0
```

На g2.2 это не рекомендуется: последний лог уже подтвердил OOM на первом SFT-шаге.

## Проверки

```text
python -m py_compile train_vlm_sft.py train_vlm_dpo.py train_vlm_grpo.py: OK
bash -n DataSphere wrappers: OK
pytest: 180 passed, 6 skipped
```
