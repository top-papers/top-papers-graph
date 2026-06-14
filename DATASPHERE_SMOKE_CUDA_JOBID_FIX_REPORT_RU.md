# Отчёт об исправлениях smoke-прогона DataSphere Qwen3-VL-8B

Дата: 2026-06-14.

## Причина исправлений

Smoke-прогон дошёл до SFT stage и упал из-за ошибки CUDA/PyTorch:

```text
CUDA initialization: The NVIDIA driver on your system is too old (found version 12020)
ValueError: Your setup doesn't support bf16/gpu.
```

Дополнительно managed launcher неверно распарсил локальный путь вида `/tmp/datasphere/job_2026-...` как DataSphere job id и затем попытался выполнить `cancel`/`set-data-ttl` с неправильным id.

## Изменённые файлы

```text
experiments/vlm_finetuning/datasphere/requirements.txt
experiments/vlm_finetuning/datasphere/bin/check_gpu_before_pipeline.py
experiments/vlm_finetuning/datasphere/bin/run_hf_top_papers_sft_grpo_full.sh
experiments/vlm_finetuning/datasphere/run_full_pipeline.py
experiments/vlm_finetuning/datasphere/job_configs/*.yaml
TUTORIAL_QWEN3_VL_8B_DATASPHERE_SMOKE_FIX_RU.md
```

## Исправление 1: pinned PyTorch CUDA wheels

В `requirements.txt` заменены плавающие зависимости `torch>=...`, `torchvision>=...` на:

```text
--extra-index-url https://download.pytorch.org/whl/cu121
torch==2.5.1+cu121
torchvision==0.20.1+cu121
torchaudio==2.5.1+cu121
```

## Исправление 2: early GPU/BF16 preflight

Добавлен скрипт:

```text
experiments/vlm_finetuning/datasphere/bin/check_gpu_before_pipeline.py
```

Wrapper вызывает его до dataset download. При ошибке он пишет:

```text
reports/<OUT_PREFIX>_datasphere/gpu_preflight_status.json
```

## Исправление 3: HFTOKEN bridge

В wrapper добавлено:

```bash
if [ -z "${HF_TOKEN:-}" ] && [ -n "${HFTOKEN:-}" ]; then
  export HF_TOKEN="$HFTOKEN"
fi
if [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ] && [ -n "${HF_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi
```

## Исправление 4: strict job id parser

`parse_job_id()` теперь принимает только явные DataSphere id из `created job ...`, `/job/<id>` или `job id: <id>`, и не принимает `/tmp/datasphere/job_...`.

## Исправление 5: no cancel after completed failed execute

Если `datasphere project job execute` завершился с non-zero кодом, managed launcher больше не делает `cancel`. Он:

1. сохраняет `job_id` в manifest;
2. выставляет TTL, если id найден;
3. пытается скачать доступные outputs;
4. возвращает исходный non-zero code.

## Проверки

```text
[OK] python -m py_compile
[OK] bash -n wrapper scripts
[OK] parse_job_id unit sample
[OK] PyYAML load job configs
[OK] managed launcher dry-run
```
