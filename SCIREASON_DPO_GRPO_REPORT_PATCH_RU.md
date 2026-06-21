# Патч по итогам аудита DPO/GRPO для SciReason

Этот репозиторий обновлён по рекомендациям отчёта `reports/deep-research-report-2026-06-21.md`.

## Что изменено

1. **DPO-first alignment усилен**
   - `train_vlm_dpo.py` теперь по умолчанию использует robust DPO: `loss_type=robust`, `beta=0.06`, `label_smoothing=0.05`.
   - Добавлены CLI-флаги: `--loss-type` с несколькими значениями, `--loss-weights`, `--use-weighting/--no-use-weighting`, `--precompute-ref-log-probs`, `--precompute-ref-batch-size`, `--padding-free`, `--activation-offloading`.
   - В DataSphere entrypoint добавлены multi-loss настройки `robust sft` с весами `1.0 0.15` и WPO-style weighting.

2. **DPO pair mining стал hard-negative oriented**
   - `build_scireason_alignment_datasets.py` теперь строит до нескольких preference-пар на source row.
   - Добавлены типы negative pairs: `verdict_flip`, `evidence_drop`, `task_hard_synthetic`, `policy_negative`.
   - В metadata каждой пары добавлены `pair_type` и `pair_hardness`.
   - Preflight-аудит проверяет `DPO hard-pair ratio`.

3. **GRPO reward стал task-aware**
   - В `train_vlm_grpo.py` добавлен task-aware routing активных reward-компонент по `task_family`.
   - Когда TRL передаёт grouped generations с повторяющимся `sample_id`, активные компоненты проходят robust group normalization через median/MAD и `tanh`.
   - Inactive компоненты остаются audit-only и не загрязняют полезный GRPO advantage.

4. **GRPO hyperparameters приведены к gated polish режиму**
   - Добавлены CLI-флаги: `--num-iterations`, `--epsilon`, `--epsilon-high`, `--top-entropy-quantile`.
   - GRPO defaults в DataSphere: `beta=0.005`, `learning_rate=2e-6`, `num_iterations=2`, `num_generations_eval=4`, `max_completion_length=640`, `temperature=0.85`, `top_entropy_quantile=0.2`.
   - `mask_truncated_completions` теперь включён по умолчанию.

5. **Динамический image cap**
   - Builder теперь оценивает `page/figure/table` anchors и выбирает per-row `dynamic_cap` в пределах глобального лимита.
   - Image scoring сильнее учитывает locator/evidence overlap.

6. **Offline-safe loading**
   - `train_vlm_sft.py`, `train_vlm_dpo.py`, `train_vlm_grpo.py` теперь добавляют `local_files_only=True`, когда выставлены `HF_HUB_OFFLINE=1` или `TRANSFORMERS_OFFLINE=1`.

## Проверки

Выполнены:

```bash
python -m py_compile experiments/vlm_finetuning/scripts/*.py
bash -n experiments/vlm_finetuning/datasphere/bin/run_hf_top_papers_sft_dpo_grpo_v2.sh
python -m pytest -q
```

Результат полного pytest:

```text
183 passed, 6 skipped, 13 warnings
```

Ограничение: полноценное обучение Qwen/Qwen3-VL-8B-Instruct не запускалось в этом sandbox, потому что для него нужны GPU, доступ к Hugging Face датасету/модели и DataSphere-like окружение. Проверены синтаксис, unit/regression tests и orchestration shell syntax.
