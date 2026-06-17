# Исправление full-run падения SFT на DDP/NCCL watchdog

Дата: 2026-06-17.

## Симптом

Full DataSphere job `hf-full-managed` дошёл до полного SFT на Qwen3-VL-8B-Instruct и начал обучение без OOM. Датасет был подготовлен полностью:

```text
sft.train = 2293
sft.eval = 255
grpo.train = 1764
grpo.eval = 196
missing_image_refs = 0
```

SFT метрики до падения были живыми: loss снижался, `grad_norm` был ненулевым. После шага 35 один из rank'ов перестал доходить до следующего collective, и через 30 минут PyTorch/NCCL завершил job:

```text
Watchdog caught collective operation timeout: WorkNCCL(... OpType=_ALLGATHER_BASE ... Timeout(ms)=1800000)
RuntimeError: CUDA error: unspecified launch failure
```

Это не ошибка окружения, не `UnicodeDecodeError` model-card и не классический OOM. Паттерн соответствует DDP straggler: один rank ждёт tiny scalar all-gather, пока другой rank слишком долго обрабатывает патологически тяжёлый VLM пример.

## Причина

Для VLM SFT `max_length=None` оставлен намеренно, чтобы не обрезать image tokens. Но без отдельного ограничения по текстовой поверхности отдельный пример с большим conversation JSON/text может сделать один DDP step на одном rank намного тяжелее соседнего rank. На DataSphere это приводит к 30-минутному NCCL watchdog timeout.

Дополнительный риск — `dataloader_num_workers=4` для full SFT: multiprocessing image decoding/processing может усиливать rank skew на полном наборе из 15k+ asset files.

## Исправление

1. В `train_vlm_sft.py` добавлен guard:

```bash
--max-text-chars N
```

Он отбрасывает только патологически длинные нормализованные SFT rows до TRL multimodal collator. Для full DataSphere config выставлено:

```text
SFT_MAX_TEXT_CHARS=12000
```

2. В `train_vlm_sft.py` добавлен явный DDP timeout:

```bash
--ddp-timeout-seconds N
```

Для full/smoke DataSphere config выставлено:

```text
SFT_DDP_TIMEOUT_SECONDS=7200
```

Это даёт запас после удаления pathological rows, но не является основным лечением.

3. Для full-run SFT dataloader workers снижены:

```text
SFT_DATALOADER_NUM_WORKERS=0
```

Это делает image loading более детерминированным внутри rank-процессов и убирает риск multiprocessing stragglers/deadlocks на полном датасете.

4. Budget plan теперь явно логирует:

```json
"sft_max_text_chars": 12000,
"sft_ddp_timeout_seconds": 7200
```

5. `run_config.json` SFT теперь сохраняет:

```json
"dropped_long_train_rows": ...,
"dropped_long_eval_rows": ...
```

## Проверки

Локально выполнено:

```text
python -m compileall -q experiments/vlm_finetuning/scripts experiments/vlm_finetuning/datasphere
bash -n experiments/vlm_finetuning/datasphere/launch_examples.sh experiments/vlm_finetuning/datasphere/bin/*.sh
pytest -q
```

Результат:

```text
170 passed, 6 skipped, 13 warnings
```

## Что смотреть в следующем full-run

После запуска проверьте первые SFT логи:

```text
[train_vlm_sft] dropped X/Y train rows with normalized text longer than 12000 chars ...
```

Нормально, если `X` небольшой. Если будет отброшено слишком много строк, можно поднять лимит до `16000` или `20000`. Если строк почти не отброшено и timeout повторится, следующий безопасный шаг — уменьшить `VLM_MAX_PIXELS` с `1003520` до `752640`.
