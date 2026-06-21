# Исправление последнего DataSphere прогона и улучшение alignment-пайплайна

Дата: 2026-06-21

## Что показал последний лог

Последний запуск прошел дальше предыдущих аварийных стадий:

- build сохранил полный raw export: SFT `2548` строк, GRPO `1960` строк, без build-time усечения image refs;
- SFT завершился полностью и безопасно сохранил лучший чекпойнт `checkpoint-540`;
- GRPO завершил train/eval и упал уже после обучения, на post-run reward audit;
- критическая ошибка: `NameError: name 'active_zero_std_frac' is not defined` в `train_vlm_grpo.py`.

То есть это не ошибка CUDA/OOM и не ошибка PEFT best-checkpoint reload. Модель уже доучивалась почти до конца, но финальная упаковка/аудит падали из-за опечатки в inline-версии reward-audit кода.

## Исправления

### 1. Исправлен `NameError` в GRPO post-run audit

Файл: `experiments/vlm_finetuning/scripts/train_vlm_grpo.py`

Что сделано:

- добавлен расчет `active_group_stds`;
- добавлен расчет `active_zero_std_frac`;
- `weak_reward` теперь основан на active-group zero-std, как в standalone `audit_reward_trace.py`;
- audit JSON теперь получает `status: pass/fail`;
- post-run audit и сохранение processor выполняются только на main process, чтобы DDP-rank'и не писали один и тот же файл одновременно.

### 2. Legacy full SFT+GRPO job теперь явно включает KL/beta

Файл: `experiments/vlm_finetuning/datasphere/bin/run_hf_top_papers_sft_grpo_full.sh`

Добавлено:

```bash
--beta "${GRPO_BETA:-0.02}"
--min-reward-std "${GRPO_MIN_REWARD_STD:-0.03}"
--max-zero-std-frac "${GRPO_MAX_ZERO_STD_FRAC:-0.85}"
```

Это делает legacy full job ближе к v2-пайплайну: KL-регуляризация становится явной, а reward audit получает управляемые пороги.

### 3. Перенастроены GRPO reward weights

Файлы:

- `experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_full_g2_2.yaml`
- `experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_dpo_grpo_v2_g2_2.yaml`

Новые веса:

```text
0.0 0.25 0.5 0.0 0.25 2.0
```

Порядок соответствует reward-функциям:

1. `label_exact_match` — 0.0, потому что в текущем GRPO split это неактивная family;
2. `schema_validity` — 0.25, чтобы JSON/schema не доминировала;
3. `temporal_consistency` — 0.5;
4. `graph_consistency` — 0.0, потому что в assertion-review GRPO она фактически неактивна;
5. `evidence_presence` — 0.25, потому что компонент почти насыщается;
6. `expert_override_match` — 2.0, потому что после нормализации verdict aliases это основной живой reward-сигнал.

### 4. V2-пайплайн остается рекомендуемым

Рекомендуемый production path остается:

```text
full raw export build
→ text SFT
→ multimodal SFT
→ robust DPO / cDPO-style alignment
→ optional short KL-constrained GRPO polish
```

Legacy `hf_top_papers_sft_grpo_full_g2_2.yaml` оставлен рабочим для совместимости с текущими запусками, но новый качественный путь — `hf_top_papers_sft_dpo_grpo_v2_g2_2.yaml`.

## Проверки

Выполнено локально:

```text
py_compile: OK
bash -n run_hf_top_papers_sft_grpo_full.sh: OK
bash -n run_hf_top_papers_sft_dpo_grpo_v2.sh: OK
pytest: 183 passed, 6 skipped
```

Полный GPU/DataSphere run здесь не запускался: в текущем окружении нет g2.2/A100 runtime.

## Ожидаемый эффект следующего запуска

Следующий legacy full job не должен падать на `active_zero_std_frac` после GRPO. Если reward-сигнал останется слабым, pipeline сохранит `post_run_reward_audit.json` и отчеты, но не упадет из-за ошибки Python. Для качества рекомендуется запускать v2 job, где DPO является основным alignment stage, а GRPO остается optional polish.
