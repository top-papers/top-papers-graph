# Fix: GRPO падает на `images=None` в Qwen3-VL processor

## Симптом

Новый DataSphere smoke job дошёл до GRPO после успешной сборки окружения, GPU preflight и SFT, но упал на первом GRPO training step:

```text
TypeError: only a single or a list of entries is supported but got type=<class 'NoneType'>
```

Трассировка идёт через `trl.trainer.grpo_trainer._generate_and_score_completions` в `Qwen3VLProcessor(... images=images ...)`.

## Причина

SFT допускает смешанный VLM/text-only датасет. В текущей связке TRL + Transformers multimodal GRPO вызывает Qwen3-VL processor с общим `images=...` для generation batch. Если в batch попадает text-only строка с `images=[]`/`None`, processor получает `None` как один из image entries и падает.

Это проявилось на smoke-сэмпле: часть GRPO rows осталась без доступных изображений после sample-limited asset download и cap `MAX_IMAGES_PER_EXAMPLE_GRPO=2`.

## Исправление

1. В `experiments/vlm_finetuning/scripts/train_vlm_grpo.py` добавлен предфильтр для VLM GRPO:
   - train split в forced VLM режиме оставляет только rows с непустыми `images`/`image`;
   - если forced VLM train split не содержит изображений, job падает рано с понятным `ValueError`;
   - eval split без изображений отключается вместо падения в processor.

2. В `experiments/vlm_finetuning/datasphere/bin/run_hf_top_papers_sft_grpo_full.sh` добавлена более компактная упаковка adapter artifacts:
   - удаляются/исключаются `checkpoint-*`, `optimizer.pt`, `scheduler.pt`, `rng_state*.pth`, `scaler.pt`;
   - это предотвращает превышение DataSphere download limit для smoke outputs;
   - `final_summary.json` теперь создаётся fallback-ом даже при аварийном завершении, чтобы declared output не ломал скачивание диагностик.

## Проверка

Локально выполнены:

```bash
python -m compileall experiments/vlm_finetuning/scripts experiments/vlm_finetuning/datasphere
bash -n experiments/vlm_finetuning/datasphere/launch_examples.sh experiments/vlm_finetuning/datasphere/bin/*.sh
pytest tests/test_vlm_training_format_normalization.py tests/test_datasphere_job_configs.py
```
