# Patch report: полный Yandex DataSphere Jobs pipeline для VLM SFT/RL

## Что добавлено

Добавлен production-oriented контур для запуска полного дообучения VLM на `top-papers/top-papers-graph-experts-data` через Yandex DataSphere Jobs.

### Новые файлы

- `experiments/vlm_finetuning/scripts/build_hf_graph_experts_dataset.py`  
  Скачивает HF dataset, сохраняет изображения локально, строит `sft_*.jsonl` и `grpo_*.jsonl`.

- `experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_full_g2_2.yaml`  
  DataSphere job config под `g2.2`, 2 × A100 80 GB, 56 vCPU, 238 GB RAM, SSD 1024 GB.

- `experiments/vlm_finetuning/datasphere/bin/run_hf_top_papers_sft_grpo_full.sh`  
  Runtime wrapper: dataset build → SFT LoRA → GRPO/RL → упаковка outputs/reports.

- `experiments/vlm_finetuning/datasphere/run_full_pipeline.py`  
  Локальный managed launcher: запускает `datasphere project job execute`, скачивает outputs, ставит TTL job data в 1 день, при ошибке пытается cancel.

- `experiments/vlm_finetuning/datasphere/HF_TOP_PAPERS_FULL_PIPELINE_RU.md`  
  Подробная инструкция по запуску и параметрам.

- `experiments/vlm_finetuning/datasphere/TUTORIAL_FULL_EXPERIMENT_RU.md`  
  Пошаговый tutorial для полного цикла эксперимента: подготовка CLI, preflight, запуск, мониторинг, скачивание outputs и troubleshooting.

### Измененные файлы

- `experiments/vlm_finetuning/scripts/train_vlm_sft.py`  
  Исправлена логика подготовки SFT: если JSONL уже содержит `messages`, они не перетираются числовым `label`.

- `experiments/vlm_finetuning/scripts/train_vlm_grpo.py`  
  Добавлен reward `reward_label_exact_match` для image-label RL, поддержка `--sft-adapter-path` и загрузка SFT LoRA adapter перед GRPO.

- `experiments/vlm_finetuning/datasphere/launch_examples.sh`  
  Добавлены команды `hf-full` и `hf-full-managed`.

- `experiments/vlm_finetuning/datasphere/requirements.txt`  
  Обновлены зависимости для VLM/TRL/GRPO/Hugging Face Xet.

- `experiments/vlm_finetuning/README.md`  
  Добавлен раздел про полный HF top-papers DataSphere pipeline.

## Выбранная схема

- модель: `Qwen/Qwen3-VL-8B-Instruct`;
- обучение: LoRA SFT → GRPO/RL поверх SFT adapter;
- DataSphere config: `g2.2`;
- диск: `working-storage.type=SSD`, `working-storage.size=1024Gb`;
- бюджетный guard: phase timeouts + short max steps + TTL 1 день после завершения;
- outputs скачиваются локально через `datasphere project job download-files --id <job_id>`.

## Почему так

Датасет небольшой, но мультимодальный: 389 строк, 32 класса, около 5.76 GB. Полное дообучение большой VLM нерационально для первого цикла. LoRA SFT + короткий GRPO дает проверяемый результат в пределах бюджета и не требует ручного управления Compute VM.

## Проверки в этой среде

В контейнере выполнены:

- YAML parse для job configs;
- `bash -n` для новых shell wrappers;
- `python -m py_compile` для измененных Python-файлов;
- dry-run локального launcher.

Полный запуск обучения не выполнялся, потому что в текущем контейнере нет доступа к DataSphere/GPU и не установлены runtime-зависимости `datasets/torch/trl`.

## Дополнительные исправления совместимости

- `datasphere/bin/common.sh` исправлен так, чтобы переходить в корень репозитория, а не в каталог `experiments/`.
- DataSphere YAML configs больше не смешивают `root-path` и `local-paths`.
- `working-storage` во всех configs приведен к явному `type: SSD` и размерам `...Gb`.
- `run_full_pipeline.py` стримит длинные логи и хранит только ограниченный tail в памяти.
- SFT/GRPO normalizers устойчиво обрабатывают `image`, `images`, `messages`, `chat.messages` и multimodal blocks.

См. также `DATASPHERE_VLM_JOBS_FIX_REPORT_RU.md`.

