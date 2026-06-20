# SciReason fine-tuning v2 patch report

Дата: 2026-06-19

## Что изменено

В репозиторий добавлена v2-схема дообучения, которая реализует рекомендации после анализа полного SFT+GRPO-прогона от 19 июня 2026 года, архива репозитория и HF-датасета `top-papers/top-papers-graph-experts-data`.

## Ключевые изменения

1. **Export-only dataset build**
   - Новый скрипт: `experiments/vlm_finetuning/scripts/build_scireason_alignment_datasets.py`.
   - Скрипт жёстко использует `exports/colab-run-001/sft.jsonl` и `grpo.jsonl`.
   - Generic HF `imagefolder`-viewer больше не используется как основной источник instruction/reasoning-данных.

2. **Защита от неправильного imagefolder fallback**
   - В `build_hf_graph_experts_dataset.py` добавлен флаг `--allow-imagefolder-fallback`.
   - Если запустить `--source-mode imagefolder` без этого флага, скрипт завершится с ошибкой и пояснением.

3. **Leakage-safe split**
   - Split выполняется по грубому source group: `paper/source_file/submission/expert`, а не только по `task_family`.
   - Генерируется `leakage_report.json` с проверкой пересечения групп между train/eval.

4. **Relevance-based image selection**
   - Изображения выбираются не просто как первые `N`, а по лёгкому deterministic score: evidence tokens, figure/table/page hints, filename overlap.
   - Генерируется `image_resolution_report.json`.

5. **Новый alignment dataset layout**
   - `sft_text_train.jsonl` / `sft_text_eval.jsonl` для text-only SFT с `assistant_only_loss`.
   - `sft_vlm_train.jsonl` / `sft_vlm_eval.jsonl` для multimodal SFT.
   - `dpo_train.jsonl` / `dpo_eval.jsonl` для DPO.
   - `grpo_train_verified.jsonl` / `grpo_eval_verified.jsonl` только для строк с явными reward targets.

6. **SFT continuation from adapter**
   - В `train_vlm_sft.py` добавлен `--init-adapter-path`.
   - Это позволяет запускать multimodal SFT от text-SFT LoRA-адаптера.

7. **Новый DataSphere pipeline**
   - Скрипт: `experiments/vlm_finetuning/datasphere/bin/run_hf_top_papers_sft_dpo_grpo_v2.sh`.
   - Job config: `experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_dpo_grpo_v2_g2_2.yaml`.
   - High-level config: `experiments/vlm_finetuning/configs/sft_dpo_grpo_v2_qwen3vl_8b_lora.yaml`.

## Новая рекомендуемая схема

```text
1. build_scireason_alignment_datasets.py
2. text-only SFT + assistant_only_loss=True
3. multimodal SFT от text-SFT adapter
4. DPO от multimodal SFT adapter
5. optional GRPO polish только при ENABLE_GRPO_POLISH=1
```

GRPO по умолчанию отключён, потому что предыдущий полный прогон показал слабую reward variance и нулевые/слабые correctness-компоненты. Новый builder всё равно создаёт verified GRPO split и reward audit, чтобы GRPO можно было включить после проверки.

## Как запустить DataSphere v2 job

```bash
python experiments/vlm_finetuning/datasphere/run_full_pipeline.py \
  --project-id "$DATASPHERE_PROJECT_ID" \
  --config experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_dpo_grpo_v2_g2_2.yaml
```

Для включения короткого GRPO polish добавьте в job config или окружение:

```bash
ENABLE_GRPO_POLISH=1
```

## Проверки

Выполнены локальные проверки:

```bash
python -m py_compile \
  experiments/vlm_finetuning/scripts/build_scireason_alignment_datasets.py \
  experiments/vlm_finetuning/scripts/build_hf_graph_experts_dataset.py \
  experiments/vlm_finetuning/scripts/train_vlm_sft.py \
  experiments/vlm_finetuning/scripts/train_vlm_dpo.py \
  experiments/vlm_finetuning/scripts/train_vlm_grpo.py

pytest -q \
  tests/test_scireason_alignment_dataset_v2.py \
  tests/test_vlm_training_format_normalization.py \
  tests/test_datasphere_job_configs.py
```

Результат: `47 passed`.

## Важное ограничение

Патч не запускал дорогое обучение. Он подготавливает репозиторий к следующему корректному эксперименту и добавляет quality gates, но фактическое качество новой модели должно быть подтверждено отдельным DataSphere запуском и blind A/B eval.
