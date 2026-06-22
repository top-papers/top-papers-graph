# Исправление: полноценный запуск DPO-first пайплайна

Дата: 2026-06-22

## Проблема

Последний успешный DataSphere-прогон использовал legacy job `hf_top_papers_sft_grpo_full_g2_2.yaml`, то есть фактически запускал `SFT -> GRPO` и обходил новую DPO-first схему. В результате DPO-improvements присутствовали в репозитории, но не применялись в production/full запуске.

Дополнительная потенциальная проблема была в формате DPO-данных: v2 builder хранит `prompt` как список chat messages, а `chosen`/`rejected` — как компактные строки. Для TRL conversational DPO эти completion поля должны быть приведены к assistant message list. Пока DPO реально не запускался, этот риск не проявлялся в логах.

## Что изменено

1. `run_hf_top_papers_sft_grpo_full.sh`
   - добавлен backward-compatible safety rail;
   - non-smoke legacy full jobs теперь по умолчанию делегируют в `run_hf_top_papers_sft_dpo_grpo_v2.sh`;
   - чтобы принудительно вернуться к старому поведению, нужно явно задать `ENABLE_DPO_FIRST_PIPELINE=0`.

2. `hf_top_papers_sft_grpo_full_g2_2.yaml`
   - оставлен совместимым по имени, но теперь явно включает `ENABLE_DPO_FIRST_PIPELINE=1`;
   - добавлены DPO-параметры: `robust sft`, `loss_weights`, `use_weighting`, `precompute_ref_log_probs`, `label_smoothing`, `DPO_EPOCHS=1.5`;
   - outputs дополнены DPO/SFT-v2 архивами.

3. `run_hf_top_papers_sft_dpo_grpo_v2.sh`
   - добавлена optional Hugging Face upload стадия;
   - если GRPO polish выключен, upload автоматически публикует DPO adapter как final adapter;
   - если GRPO включён и даёт adapter files, final adapter выбирается автоматически как GRPO.

4. `upload_hf_finetuned_artifacts.py`
   - добавлена поддержка `--dpo-dir`, `--final-stage`, `--final-dir`;
   - upload bundle теперь сохраняет `artifacts/dpo_lora/`;
   - root репозитория может быть DPO adapter, если GRPO пропущен;
   - model card обновлён под `SFT -> DPO -> optional GRPO`.

5. `train_vlm_dpo.py`
   - DPO formatter теперь приводит строковые `chosen`/`rejected` к conversational assistant messages, когда `prompt` является chat message list;
   - это делает JSONL из v2 builder совместимым с TRL conversational DPO format.

6. Tests
   - добавлен regression test `test_dpo_formatter_wraps_string_completions_for_conversational_prompt`.

## Рекомендуемый запуск

```bash
datasphere project job execute \
  -p <PROJECT_ID> \
  -c experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_dpo_grpo_v2_g2_2.yaml
```

Legacy-compatible full config теперь также запускает DPO-first pipeline:

```bash
datasphere project job execute \
  -p <PROJECT_ID> \
  -c experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_full_g2_2.yaml
```

## Ожидаемый результат

Production/full запуск больше не должен публиковать только `SFT -> GRPO` checkpoint. По умолчанию главным кандидатом становится DPO adapter после `text-SFT -> VLM-SFT -> robust mixed DPO`. GRPO остаётся optional short polish и запускается только при `ENABLE_GRPO_POLISH=1`.
