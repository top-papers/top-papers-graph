# Исправление `launch_examples.sh hf-full-managed` для полного SFT -> DPO -> GRPO цикла

## Что было

Команда:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh hf-full-managed
```

вызывала `run_full_pipeline.py` без явного `--config`. В самом `run_full_pipeline.py` дефолтным конфигом оставался legacy-compatible файл `hf_top_papers_sft_grpo_full_g2_2.yaml`. Этот файл уже делегировал в v2 pipeline, но путь запуска был непрозрачным: managed launcher всё ещё выглядел как SFT+GRPO full job.

## Что изменено

Теперь managed full launcher явно запускает production v2 config:

```bash
experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_dpo_grpo_v2_g2_2.yaml
```

То есть команда:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh hf-full-managed
```

создаёт DataSphere job для полного цикла:

```text
text-SFT -> VLM-SFT -> robust mixed DPO -> optional GRPO polish -> HF upload
```

## Изменённые файлы

- `experiments/vlm_finetuning/datasphere/launch_examples.sh`
  - action `hf-full-managed` теперь передаёт `--config experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_dpo_grpo_v2_g2_2.yaml`.
  - Для экспериментов можно переопределить конфиг через `HF_FULL_MANAGED_CONFIG=...`.

- `experiments/vlm_finetuning/datasphere/run_full_pipeline.py`
  - `DEFAULT_CONFIG` теперь тоже указывает на v2 SFT+DPO+GRPO config.
  - Логи и manifest называются `hf_top_papers_sft_dpo_grpo_*`, чтобы не вводить в заблуждение.
  - CLI description обновлён под полный SciReason pipeline.

## Проверка

Выполнен dry-run:

```bash
DATASPHERE_PROJECT_ID=dummy \
  bash experiments/vlm_finetuning/datasphere/launch_examples.sh hf-full-managed --dry-run --no-download
```

Результат manifest:

```json
{
  "config": "experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_dpo_grpo_v2_g2_2.yaml",
  "expected_pipeline": [
    "text-SFT",
    "VLM-SFT",
    "robust mixed DPO",
    "optional GRPO polish",
    "Hugging Face upload"
  ],
  "managed_entrypoint": "bash experiments/vlm_finetuning/datasphere/launch_examples.sh hf-full-managed",
  "commands": {
    "execute": [
      "datasphere",
      "project",
      "job",
      "execute",
      "-p",
      "dummy",
      "-c",
      "experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_dpo_grpo_v2_g2_2.yaml"
    ]
  },
  "status": "planned"
}
```

## Итог

`hf-full-managed` больше не зависит от legacy-compatible обходного конфига. Основной managed entrypoint теперь напрямую запускает рекомендованный full pipeline с robust mixed DPO и включённым по умолчанию GRPO polish (`ENABLE_GRPO_POLISH=1` в v2 job config).
