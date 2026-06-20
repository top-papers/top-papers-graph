# Исправление падения SFT после анализа `smoke_test_logs(20).txt`

Дата: 2026-06-20

## Что показал лог

Прогон дошёл до конца SFT-стадии: `480/480`, после чего финальная оценка дала `eval_loss=1.2321186065673828`. Падение произошло не во время forward/backward, а на этапе финальной загрузки лучшего чекпойнта в `Trainer._load_best_model()`:

```text
ImportError: cannot import name 'EmbeddingParallel' from 'transformers.integrations.tensor_parallel'
```

Стек проходил через `model.load_adapter(...)` и `peft.utils.save_and_load._maybe_shard_state_dict_for_tp(...)`. Это указывает на несовместимость установленной пары PEFT/Transformers в managed runtime при нативном `load_best_model_at_end=True` для LoRA/PEFT adapter checkpoint в DDP.

Дополнительно в логе были видны признаки неполного использования данных:

- `max_images_per_example_sft=3`, `max_images_per_example_grpo=2`;
- `rows_with_truncated_images=430` для SFT и `1603` для GRPO;
- `SFT_MAX_TEXT_CHARS=12000` отфильтровал 64 train rows и 6 eval rows;
- `MAX_SFT_STEPS=480` остановил SFT до полного `SFT_EPOCHS=4`.

Также во время чекпойнтов возникали повторные сетевые HEAD-запросы к Hugging Face (`RemoteDisconnected`), которые не были основной причиной падения, но замедляли и зашумляли run.

## Что изменено

### 1. Safe best checkpoint вместо нативного PEFT reload

Файлы:

- `experiments/vlm_finetuning/scripts/train_vlm_sft.py`
- `experiments/vlm_finetuning/scripts/train_vlm_dpo.py`

Изменения:

- добавлен флаг `--native-load-best-model-at-end`, по умолчанию `False`;
- `load_best_model_at_end` в `SFTConfig`/`DPOConfig` теперь включается нативно только если это не PEFT adapter model или явно разрешён native mode;
- для LoRA/PEFT сохраняется best checkpoint через безопасное копирование adapter artifacts из `trainer.state.best_model_checkpoint`, без вызова `PeftModel.load_adapter()`;
- создаются manifest-файлы:
  - `sft_best_checkpoint_manifest.json`
  - `dpo_best_checkpoint_manifest.json`
- `run_config.json` теперь содержит:
  - `best_checkpoint_selection_enabled`;
  - `native_best_checkpoint_reload_enabled`;
  - `best_checkpoint_selection_mode`.

### 2. Full-data defaults в DataSphere jobs

Файлы:

- `experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_full_g2_2.yaml`
- `experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_dpo_grpo_v2_g2_2.yaml`
- `experiments/vlm_finetuning/datasphere/bin/run_hf_top_papers_sft_grpo_full.sh`
- `experiments/vlm_finetuning/datasphere/bin/run_hf_top_papers_sft_dpo_grpo_v2.sh`

Изменения:

- `MAX_IMAGES_PER_EXAMPLE_SFT=0`;
- `MAX_IMAGES_PER_EXAMPLE_GRPO=0`;
- `SFT_MAX_TEXT_CHARS=0`;
- legacy full SFT job теперь использует `MAX_SFT_STEPS=-1`, чтобы не прерывать эпохи фиксированным step cap.

### 3. Prefetch + offline training mode

DataSphere wrappers теперь перед training stages делают prefetch base model через `huggingface_hub.snapshot_download`, затем включают:

```bash
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

Это должно убрать повторные сетевые HEAD-запросы во время checkpoint/model-card side effects после того, как модель уже загружена в cache. Перед upload stage offline-флаги снимаются.

## Проверки

Выполнено локально:

```bash
python -m py_compile \
  experiments/vlm_finetuning/scripts/train_vlm_sft.py \
  experiments/vlm_finetuning/scripts/train_vlm_dpo.py \
  experiments/vlm_finetuning/scripts/train_vlm_grpo.py \
  experiments/vlm_finetuning/scripts/audit_full_data_usage.py

bash -n experiments/vlm_finetuning/datasphere/bin/run_hf_top_papers_sft_grpo_full.sh
bash -n experiments/vlm_finetuning/datasphere/bin/run_hf_top_papers_sft_dpo_grpo_v2.sh

python -m pytest \
  tests/test_datasphere_job_configs.py \
  tests/test_vlm_training_format_normalization.py \
  tests/test_scireason_alignment_dataset_v2.py -q
```

Результат:

```text
50 passed in 0.84s
```

## Ожидаемый эффект при следующем запуске

- SFT не должен падать после последнего шага на `EmbeddingParallel` / `PeftModel.load_adapter`.
- Best checkpoint всё равно будет выбран и сохранён, но через безопасное копирование adapter artifacts.
- Датасет не должен терять изображения из-за cap `3/2`.
- SFT не должен выкидывать длинные строки по `12000 chars` по умолчанию.
- Legacy full job не должен останавливаться на `480` шагах, если задано обучение по эпохам.
- Повторные Hugging Face HEAD-запросы после prefetch должны исчезнуть или существенно сократиться.
