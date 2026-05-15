# Отчет о правках DataSphere Jobs для VLM fine-tuning

Дата ревизии: 2026-05-15.

## Цель

Подготовить архив репозитория так, чтобы полный эксперимент `experiments/vlm_finetuning` запускался через Yandex DataSphere Jobs на датасете Hugging Face `top-papers/top-papers-graph-experts-data` максимально воспроизводимо: сборка данных, SFT, GRPO, упаковка и скачивание артефактов.

## Ключевые исправления

1. Исправлен `experiments/vlm_finetuning/datasphere/bin/common.sh`: теперь wrappers переходят в настоящий корень репозитория, а не в `experiments/`.
2. Все DataSphere YAML configs приведены к совместимой схеме `local-paths` без одновременного `root-path`.
3. `working-storage` нормализован: `type: SSD`, размеры в формате `...Gb`.
4. Полный HF job использует `working-storage.size: 1024Gb` и `cloud-instance-types: [g2.2]`.
5. `build_hf_graph_experts_dataset.py` теперь сохраняет изображения строго в `--out-dir/images` и пишет в JSONL пути, которые корректно разрешаются из job working directory.
6. `train_vlm_sft.py` получил устойчивую нормализацию `messages`/`chat.messages`, `image`/`images`, HF imagefolder rows и multimodal blocks.
7. `train_vlm_grpo.py` получил такую же устойчивую нормализацию prompt/image-полей и понятную ошибку при отсутствии `peft.PeftModel` для `--sft-adapter-path`.
8. `run_hf_top_papers_sft_grpo_full.sh` сохраняет статус выхода при `trap`, собирает manifest и упаковывает SFT/GRPO/report artifacts.
9. `run_full_pipeline.py` больше не копит весь stdout длинной DataSphere job в памяти: вывод стримится в терминал и log-файл, а в памяти держится только хвост для диагностики и парсинга job id.
10. `launch_examples.sh` валидирует обязательные аргументы для lifecycle-команд.
11. Добавлен полный tutorial: `experiments/vlm_finetuning/datasphere/TUTORIAL_FULL_EXPERIMENT_RU.md`.

## Проверки, выполненные локально

```bash
pytest -q tests/test_vlm_training_format_normalization.py tests/test_vlm_dataset_build_input_retention.py -q
python -m py_compile \
  experiments/vlm_finetuning/scripts/build_hf_graph_experts_dataset.py \
  experiments/vlm_finetuning/scripts/train_vlm_sft.py \
  experiments/vlm_finetuning/scripts/train_vlm_grpo.py \
  experiments/vlm_finetuning/scripts/train_vlm_dpo.py \
  experiments/vlm_finetuning/datasphere/run_full_pipeline.py
bash -n experiments/vlm_finetuning/datasphere/launch_examples.sh
for f in experiments/vlm_finetuning/datasphere/bin/*.sh; do bash -n "$f"; done
python experiments/vlm_finetuning/datasphere/run_full_pipeline.py --project-id dummy-project --dry-run
```

Также была выполнена YAML-проверка всех `experiments/vlm_finetuning/datasphere/job_configs/*.yaml`: configs читаются, `root-path` не используется вместе с `local-paths`, `working-storage` задан как SSD и размер оканчивается на `Gb`.

## Ограничение проверки

В текущем контейнере не было DataSphere credentials, GPU и интернет-доступа для запуска реальной удаленной job и загрузки HF/model artifacts. Поэтому выполнены статические, unit-level и CLI-dry-run проверки. Реальный запуск должен выполняться пользователем в своем DataSphere проекте по tutorial.
