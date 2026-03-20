# Адаптация пайплайна под Yandex DataSphere CLI

Этот слой переводит эксперименты из режима «ноутбук/ручной запуск» в режим **DataSphere Jobs + CLI**.

## Что теперь поддерживается

- сбор SFT- и preference-датасетов как отдельная job;
- `SFT smoke` и `SFT pilot` как отдельные jobs;
- `DPO pilot` как отдельная job;
- `teacher SFT` для `Qwen3-VL-30B-A3B-Instruct`;
- `student distillation` для `Qwen3-VL-4B-Instruct`;
- offline validation как отдельная job;
- lifecycle-команды: `execute`, `attach`, `list`, `get`, `cancel`, `set-data-ttl`, `download-files`.

## Что находится в этой папке

- `job_configs/*.yaml` — DataSphere job-конфиги.
- `bin/*.sh` — shell entrypoints, которые вызываются внутри jobs.
- `requirements.txt` — training/runtime стек под TRL + PEFT + Qwen-VL.
- `launch_examples.sh` — удобная оболочка над CLI-командами.

## Базовая последовательность

```bash
export DATASPHERE_PROJECT_ID="<project_id>"

bash experiments/vlm_finetuning/datasphere/launch_examples.sh build-datasets
bash experiments/vlm_finetuning/datasphere/launch_examples.sh sft-smoke
bash experiments/vlm_finetuning/datasphere/launch_examples.sh sft-pilot
bash experiments/vlm_finetuning/datasphere/launch_examples.sh dpo-pilot
bash experiments/vlm_finetuning/datasphere/launch_examples.sh teacher-sft-30b
bash experiments/vlm_finetuning/datasphere/launch_examples.sh student-distill-4b
bash experiments/vlm_finetuning/datasphere/launch_examples.sh validate
```

## Как это связано с ML-слоем

- `scripts/build_vlm_sft_dataset.py` и `scripts/build_vlm_preference_dataset.py` собирают datasets.
- `scripts/train_vlm_sft.py` и `scripts/train_vlm_dpo.py` — реальные training entrypoints под TRL.
- `bin/run_*.sh` — связывают dataset build, `torchrun`, сохранение outputs и упаковку артефактов.
- `job_configs/*.yaml` — описывают DataSphere runtime, storage, inputs/outputs и instance types.

## Почему используется torchrun

Для `g2.2` и `g2.4` важно не просто «получить доступ к нескольким GPU», а реально задействовать их в training job. Поэтому shell entrypoints вызывают `torchrun --standalone --nproc_per_node=<num_gpus> ...`, а не обычный `python ...`.

## Что важно помнить

1. `teacher-sft-30b` — это схема под большой teacher в лимите до `4x A100`, а не обещание, что именно эта конфигурация всегда будет самой дешёвой.
2. `student-distill-4b` ожидает заранее подготовленный silver corpus `data/derived/training/vlm_distill_train.jsonl`.
3. DPO job лучше запускать после появления реальных expert preference pairs; пустой preference dataset не даёт meaningful run.
4. Для длинных запусков полезно скачивать outputs и продлевать TTL данных через CLI.
