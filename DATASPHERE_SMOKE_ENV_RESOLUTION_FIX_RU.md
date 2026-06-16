# Исправление smoke-падения DataSphere на создании Python env

Дата: 2026-06-16.

## Симптом

Свежий smoke-запуск `hf-smoke-managed` создал DataSphere job `bt14qrstk9mlnfdv622g`, но не дошёл до `run_hf_top_papers_sft_grpo_full.sh`. DataSphere завершила job на стадии подготовки окружения:

```text
Cannot create env: Failed to create python env.
ReturnCode: 2
See 'system.log' for more info.
```

Так как падение происходит до запуска wrapper script, исправление должно быть в `env.python.requirements-file` / Python dependency surface, а не в SFT/GRPO runtime-коде.

## Причина

`experiments/vlm_finetuning/datasphere/requirements.txt` оставлял несколько heavy ML-зависимостей с открытым верхним диапазоном:

```text
datasets>=2.20.0
huggingface_hub>=0.30.0
accelerate>=1.8.0
transformers>=4.57.0
peft>=0.17.0
bitsandbytes>=0.48.1
```

После появления новых major/minor releases pip мог собрать другой dependency graph, чем тот, на котором уже проходил smoke. Для DataSphere Jobs это особенно рискованно: ошибка resolver/install на этой стадии выглядит как generic `Cannot create env` и скрывает конкретный pip-конфликт в `system.log`.

## Исправление

Файл `experiments/vlm_finetuning/datasphere/requirements.txt` стабилизирован для текущего Qwen3-VL smoke/full пайплайна:

```text
datasets>=4.7.0,<5
huggingface_hub>=0.34.0,<1.0
hf-xet>=1.1.0,<2
accelerate>=1.8.0,<2
transformers>=4.57.0,<4.58
trl>=1.4.0,<1.7
peft>=0.17.0,<0.20
bitsandbytes==0.48.1
qwen-vl-utils>=0.0.14,<0.1
evaluate>=0.4.2,<0.5
wandb>=0.18.0,<0.23
```

CUDA/PyTorch pins сохранены:

```text
torch==2.5.1+cu121
torchvision==0.20.1+cu121
torchaudio==2.5.1+cu121
```

`requirements.txt` по-прежнему не содержит комментариев, пустых строк и pip flags: локальный `datasphere==0.10.0` валидирует каждую строку как package requirement перед отправкой job.

## Regression tests

В `tests/test_datasphere_job_configs.py` добавлены проверки:

1. каждая строка DataSphere requirements валидна для `packaging.requirements.Requirement`;
2. в requirements есть upper bounds для нестабильной runtime-поверхности;
3. `bitsandbytes` зафиксирован на версии `0.48.1`, чтобы новый релиз не менял env без изменения репозитория.

## Проверки

Локально выполнено:

```text
[OK] python compileall для experiments/vlm_finetuning/scripts и experiments/vlm_finetuning/datasphere
[OK] bash -n для launch_examples.sh и datasphere/bin/*.sh
[OK] pytest tests/test_datasphere_job_configs.py tests/test_vlm_training_format_normalization.py
```

## Следующая проверка в DataSphere

```bash
cd top-papers-graph-main
source .venv/bin/activate
export DATASPHERE_PROJECT_ID='bt18pnosk97i8n24ddnv'
bash experiments/vlm_finetuning/datasphere/launch_examples.sh hf-smoke-managed
```

Ожидаемый результат: job должен пройти стадию `creating python env` и дойти до wrapper/preflight, где уже будут писаться pipeline-артефакты в `reports/hf_top_papers_qwen3vl_8b_smoke_datasphere/`.
