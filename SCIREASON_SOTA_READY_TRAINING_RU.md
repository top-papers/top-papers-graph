# SciReason Qwen3-VL SOTA-ready fine-tuning package

Этот архив подготовлен как готовая к запуску версия репозитория после аудита `deep-research-report(15).md`.

## Что изменено относительно предыдущего SFT→GRPO контура

1. **Схема обучения заменена на более устойчивую цепочку:**

```text
leakage-safe export build
→ text-only SFT with assistant-only loss
→ multimodal SFT on evidence-bearing images
→ DPO/cDPO-style preference alignment
→ optional short KL-constrained GRPO polish
```

2. **GRPO больше не является основным alignment-шагом.** Он выключен по умолчанию (`ENABLE_GRPO_POLISH=0`) и должен запускаться только после preflight-аудита данных и reward-ready subset.

3. **Full-data режим теперь включён по умолчанию.**

- `MAX_SFT_SAMPLES=0` и `MAX_GRPO_SAMPLES=0`: строки из export JSONL не сэмплируются.
- `MAX_IMAGES_PER_EXAMPLE_SFT=0` и `MAX_IMAGES_PER_EXAMPLE_GRPO=0`: сохраняются все image refs из HF export, а не только top-k изображений.
- `TEXT_SFT_MAX_STEPS=-1`, `VLM_SFT_MAX_STEPS=-1`, `GRPO_MAX_STEPS=-1`: обучение идёт по эпохам и не останавливается раньше прохода по датасету.

4. **Добавлены quality gates перед запуском обучения:**

```bash
python experiments/vlm_finetuning/scripts/audit_alignment_readiness.py \
  --data-dir data/derived/hf_top_papers_scireason_v2 \
  --strict

python experiments/vlm_finetuning/scripts/audit_full_data_usage.py \
  --data-dir data/derived/hf_top_papers_scireason_v2 \
  --strict \
  --require-all-images
```

Проверяются leakage-safe split, минимальные размеры SFT/DPO/GRPO subset, reward-ready coverage, отсутствие row subsampling, совпадение raw row counts с export summary и отсутствие усечения image refs.

5. **Добавлен post-run reward audit для GRPO:**

```bash
python experiments/vlm_finetuning/scripts/audit_reward_trace.py \
  --trace outputs/<run>_grpo_lora/grpo_reward_trace.jsonl \
  --strict
```

Он ловит ключевой дефект из отчёта: вырожденную внутригрупповую дисперсию reward.

6. **В `train_vlm_grpo.py` добавлен явный KL-control:** `--beta`, default `0.02`. Для smoke/debug можно поставить `--beta 0`, но для production-запуска это не рекомендуется.

7. **В SFT и DPO включён best-checkpoint selection** при наличии eval split: `load_best_model_at_end=True`, `metric_for_best_model=eval_loss`, `greater_is_better=False`.

## Основной запуск в DataSphere

```bash
bash experiments/vlm_finetuning/datasphere/bin/run_hf_top_papers_sft_dpo_grpo_v2.sh
```

Рекомендуемые переменные окружения уже прописаны в:

```text
experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_dpo_grpo_v2_g2_2.yaml
```

Для запуска через DataSphere CLI:

```bash
python experiments/vlm_finetuning/datasphere/run_full_pipeline.py \
  --project-id "$DATASPHERE_PROJECT_ID" \
  --config experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_dpo_grpo_v2_g2_2.yaml
```

## Главный config-файл схемы

```text
experiments/vlm_finetuning/configs/scireason_sota_ready_qwen3vl_8b_lora.yaml
```

Он отражает рекомендуемую production-схему и параметры:

- dataset source: `top-papers/top-papers-graph-experts-data`, `exports/colab-run-001`;
- leakage-safe split;
- full-data mode: all export rows and all image refs, unless caps are explicitly set;
- text SFT: assistant-only loss;
- VLM SFT: continuation from text adapter;
- DPO: main alignment stage;
- GRPO: optional, `beta=0.02`, `fail_on_weak_reward=true`.

## Важные outputs

```text
data/derived/hf_top_papers_scireason_v2/summary.json
data/derived/hf_top_papers_scireason_v2/leakage_report.json
data/derived/hf_top_papers_scireason_v2/image_resolution_report.json
data/derived/hf_top_papers_scireason_v2/reward_audit_by_task_family.json
data/derived/hf_top_papers_scireason_v2/alignment_readiness_report.json
data/derived/hf_top_papers_scireason_v2/full_data_usage_report.json
outputs/hf_top_papers_qwen3vl_8b_v2_text_sft_lora.tar.gz
outputs/hf_top_papers_qwen3vl_8b_v2_vlm_sft_lora.tar.gz
outputs/hf_top_papers_qwen3vl_8b_v2_dpo_lora.tar.gz
reports/hf_top_papers_qwen3vl_8b_v2_datasphere/final_summary.json
```

## Практическая рекомендация

Не включайте `ENABLE_GRPO_POLISH=1`, пока:

- `alignment_readiness_report.json` не проходит error gates;
- `full_data_usage_report.json` не подтверждает отсутствие subsampling и image truncation;
- `reward_audit_by_task_family.json` показывает достаточный reward-ready subset;
- GRPO reward trace на smoke/probe запуске не даёт `group_zero_std_fraction` выше допустимого порога.

Главная production-кандидатная модель после этой схемы — DPO adapter. GRPO следует использовать как короткий verified polish, а не как основной способ исправления модели.
