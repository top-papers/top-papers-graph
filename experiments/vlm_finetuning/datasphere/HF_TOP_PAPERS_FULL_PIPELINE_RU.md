# Полный DataSphere Jobs pipeline для `top-papers/top-papers-graph-experts-data`

Этот модуль запускает полный цикл VLM fine-tuning через DataSphere Jobs:

1. скачивает датасет `top-papers/top-papers-graph-experts-data` из Hugging Face;
2. разворачивает изображения в локальные файлы внутри рабочей директории job;
3. собирает SFT JSONL и GRPO JSONL;
4. запускает SFT LoRA для `Qwen/Qwen3-VL-8B-Instruct`;
5. запускает GRPO/RL поверх SFT-адаптера с reward за точное совпадение класса;
6. упаковывает адаптеры, логи, summary и manifest;
7. при запуске через managed CLI скачивает результаты локально и ставит TTL данных job в 1 день.

## Почему выбран `Qwen/Qwen3-VL-8B-Instruct`

Целевой запуск переведен с `Qwen/Qwen2.5-VL-7B-Instruct` на Qwen3-VL-8B как основной dense Instruct-кандидат:

- датасет небольшой: 389 строк, 32 класса, около 5.76 GB файлов;
- `g2.2` дает 2 A100 по 80 GB, 56 vCPU и 238 GB RAM, этого достаточно для LoRA SFT и короткого GRPO на 8B при batch size 1 и gradient checkpointing;
- Qwen3-VL сохраняет удобный Transformers API (`Qwen3VLForConditionalGeneration` + `AutoProcessor`) и совместим с существующим multimodal JSONL форматом `messages` + `image`/`images`;
- 8B обычно практичнее 30B+ teacher-run для первого полного цикла: ниже риск OOM, дешевле итерации и проще уложиться в бюджет 100 000 ₽.


## Полный пошаговый tutorial

Для запуска от начала до конца используйте отдельный пошаговый документ:

- `TUTORIAL_FULL_EXPERIMENT_RU.md` — подготовка DataSphere CLI, preflight-проверки, запуск, мониторинг, скачивание outputs и troubleshooting.

## Основной запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install datasphere
export DATASPHERE_PROJECT_ID=<project_id>

python experiments/vlm_finetuning/datasphere/run_full_pipeline.py \
  --project-id "$DATASPHERE_PROJECT_ID" \
  --config experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_full_g2_2.yaml \
  --ttl-days 1
```

Короткая форма через helper:

```bash
export DATASPHERE_PROJECT_ID=<project_id>
bash experiments/vlm_finetuning/datasphere/launch_examples.sh hf-full-managed
```

## Ресурсы DataSphere Jobs

Файл `job_configs/hf_top_papers_sft_grpo_full_g2_2.yaml` задает:

```yaml
cloud-instance-types:
  - g2.2

working-storage:
  type: SSD
  size: 1024Gb
```

`g2.2` — это конфигурация DataSphere с 56 vCPU, 2 GPU A100, 238 GB RAM и 160 GB VRAM суммарно.

## Бюджетный guard

В job env зафиксированы:

```yaml
BUDGET_RUB: 100000
G2_2_RUB_PER_HOUR: 1085.76
MAX_SFT_STEPS: 180
MAX_GRPO_STEPS: 80
DATA_TIMEOUT_HOURS: 4
SFT_TIMEOUT_HOURS: 30
GRPO_TIMEOUT_HOURS: 45
```

Это дает локальный hard-stop по фазам. Для абсолютного лимита на стороне аккаунта дополнительно настройте лимиты потребления проекта DataSphere в интерфейсе/админке проекта: локальный timeout закрывает job-процесс, но официальный billing guard должен жить в DataSphere.

## Что появится после запуска

DataSphere job outputs:

- `data/derived/hf_top_papers_graph_experts/sft_train.jsonl`
- `data/derived/hf_top_papers_graph_experts/sft_eval.jsonl`
- `data/derived/hf_top_papers_graph_experts/grpo_train.jsonl`
- `data/derived/hf_top_papers_graph_experts/grpo_eval.jsonl`
- `outputs/hf_top_papers_qwen3vl_8b_sft_lora.tar.gz`
- `outputs/hf_top_papers_qwen3vl_8b_grpo_lora.tar.gz`
- `reports/hf_top_papers_qwen3vl_8b_datasphere_reports.tar.gz`
- `reports/hf_top_papers_qwen3vl_8b_datasphere/final_summary.json`

## Закрытие ресурсов

DataSphere Jobs выделяет вычислительную ВМ на время выполнения job. После завершения команды job вычислительные ресурсы освобождаются. `run_full_pipeline.py` дополнительно:

- вызывает `datasphere project job download-files --id <job_id>` для скачивания outputs;
- вызывает `datasphere project job set-data-ttl --id <job_id> --days 1`, чтобы кеш, логи и результаты job не хранились 14 дней по умолчанию;
- при ошибке или прерывании пытается вызвать `datasphere project job cancel --id <job_id>`.

## Настройки качества/стоимости

Можно менять env-переменные в job config:

- дешевле/быстрее: `MAX_SFT_STEPS=80`, `MAX_GRPO_STEPS=30`, `BASE_MODEL=Qwen/Qwen3-VL-8B-Instruct`;
- качественнее: `MAX_SFT_STEPS=300`, `MAX_GRPO_STEPS=120`, но следите за budget/timeouts;
- если нужен только SFT, временно закомментируйте GRPO-блок в `bin/run_hf_top_papers_sft_grpo_full.sh`.

## Важные совместимость-фиксы

- Job configs используют `local-paths` без одновременного `root-path`, потому что эти параметры несовместимы в DataSphere config.
- Все runtime wrappers стартуют из корня репозитория через `bin/common.sh`, поэтому команды вида `python experiments/vlm_finetuning/...` работают одинаково локально и внутри job.
- `working-storage` задан как `type: SSD` и `size: 1024Gb`, что соответствует синтаксису DataSphere config.
- Managed launcher стримит длинные логи job в `reports/datasphere_cli_runs/`, не удерживая весь stdout в памяти.

