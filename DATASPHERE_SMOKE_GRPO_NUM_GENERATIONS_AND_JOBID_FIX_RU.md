# Исправление smoke GRPO: минимальное число генераций и парсинг job id

## Что показали новые логи

Smoke job `bt184khn14ilcq93juc6` успешно прошёл GPU preflight, dataset build, SFT training 20/20 и SFT evaluation. Новая ошибка возникла на старте GRPO:

```text
ValueError: GRPO requires at least 2 generations per prompt to calculate the advantages. You provided 1.
```

Также managed launcher ошибочно распарсил auth `subject-id` как `job_id`, поэтому TTL/download были вызваны с `ajesipqgkc5efi6eo4am` вместо настоящего DataSphere job id `bt184khn14ilcq93juc6`.

## Внесённые изменения

1. В `hf_top_papers_sft_grpo_smoke_g2_2.yaml` выставлено:

```yaml
GRPO_NUM_GENERATIONS: 2
GRPO_NUM_GENERATIONS_EVAL: 2
```

2. В `run_hf_top_papers_sft_grpo_full.sh` добавлен guard, который принудительно поднимает `GRPO_NUM_GENERATIONS` и `GRPO_NUM_GENERATIONS_EVAL` до 2, если stale env/config передал 1.

3. В `train_vlm_grpo.py` добавлен `enforce_minimum_grpo_generations(args)`, чтобы прямой CLI-запуск также не падал на `GRPOConfig`.

4. В `run_full_pipeline.py` удалён общий parser `id ...`, который ловил `subject-id`. Теперь job id берётся только из `created job ...` или `/job/<id>` URL.

5. Добавлены regression tests для GRPO generation count и parsing `subject-id`.

## Ожидаемый следующий smoke результат

SFT снова должен завершиться успешно, GRPO должен пройти создание `GRPOConfig` и перейти к generation/training steps.
