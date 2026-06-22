# Полное включение схемы SFT → DPO → GRPO

Дата: 2026-06-22

## Что проверено

После проверки `scireason-dpo-first-full-pipeline-repo.zip` выяснилось, что DPO-first путь был активен, но GRPO polish всё ещё был выключен по умолчанию через `ENABLE_GRPO_POLISH=0` в v2 job config и legacy-compatible full job config. Это означало, что фактический production-запуск давал `SFT → DPO`, а не полный `SFT → DPO → GRPO`, если пользователь явно не переопределял переменную окружения.

## Что изменено

1. `experiments/vlm_finetuning/datasphere/bin/run_hf_top_papers_sft_dpo_grpo_v2.sh`
   - GRPO stage теперь включён по умолчанию: `ENABLE_GRPO_POLISH:-1`.
   - Комментарий обновлён: это full `SFT → DPO → GRPO` схема; `ENABLE_GRPO_POLISH=0` оставлен только для DPO-only ablations.
   - `final_summary.json` ищет `GRPO_DIR/run_config.json` после реального GRPO run и fallback на `planned_run_config.json` только если GRPO был отключён.

2. `experiments/vlm_finetuning/datasphere/bin/run_hf_top_papers_sft_grpo_full.sh`
   - Legacy-compatible wrapper теперь делегирует в v2 и экспортирует `ENABLE_GRPO_POLISH=1` по умолчанию.

3. `experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_dpo_grpo_v2_g2_2.yaml`
   - `ENABLE_GRPO_POLISH: '1'`.
   - `GRPO_MAX_STEPS: 120`, чтобы GRPO был активен, но ограничен как short polish.
   - Описание обновлено: GRPO включён после DPO по умолчанию.

4. `experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_full_g2_2.yaml`
   - `ENABLE_GRPO_POLISH: '1'`.
   - Добавлен `GRPO_MAX_STEPS: 120`, так как v2 wrapper читает именно это имя, а не только legacy `MAX_GRPO_STEPS`.
   - Outputs приведены ближе к делегированному v2-пайплайну и включают DPO/GRPO adapter archives.

## Итог

Default production path теперь:

```text
text-only SFT → multimodal VLM-SFT → robust mixed DPO → GRPO polish → upload final adapter
```

GRPO можно отключить только явно:

```bash
ENABLE_GRPO_POLISH=0
```

