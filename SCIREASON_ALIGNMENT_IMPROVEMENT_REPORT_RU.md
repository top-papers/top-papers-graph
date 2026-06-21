# Улучшение схемы дообучения SciReason после успешного, но слабого GRPO-прогона

Дата патча: 2026-06-21.

## Причина изменения

Последний DataSphere-прогон впервые прошёл end-to-end, но GRPO-аудит показал слабый reward-сигнал: высокая доля групп с нулевой дисперсией reward и несколько почти константных reward-компонентов. Поэтому схема изменена с «SFT → GRPO»/«GRPO как основной alignment» на более устойчивую последовательность:

```text
full raw export build
→ text-only SFT
→ multimodal SFT
→ DPO / robust preference alignment as the main alignment stage
→ optional short KL-constrained GRPO polish only if reward audit passes
```

## Что изменено

### 1. DPO теперь получает больше качественных preference pairs

`experiments/vlm_finetuning/scripts/build_scireason_alignment_datasets.py` теперь строит DPO не только из SFT rows с synthetic hard negative, но и из GRPO/RL rows с явными целями:

- `assertion_review_rl` / `mm_review_rl`: chosen = экспертный verdict, rejected = противоположный verdict;
- `trajectory_reasoning_rl`: chosen = reference assertions, rejected = пустой/unsupported graph;
- `temporal_fix_rl`: chosen = reference temporal target, rejected = unknown temporal output;
- `image_label_rl`: chosen = reference label, rejected = unknown label.

Новые функции:

- `make_dpo_rows_from_grpo(...)`
- `dedupe_dpo_rows(...)`
- `canonical_verdict(...)`

В `summary.json` добавлены counts `dpo_from_sft` и `dpo_from_grpo`.

### 2. DPO stage стал robust/cDPO-style

`train_vlm_dpo.py` получил флаг:

```bash
--label-smoothing
```

DataSphere v2 wrapper теперь передаёт:

```bash
--loss-type "${DPO_LOSS_TYPE:-robust}"
--label-smoothing "${DPO_LABEL_SMOOTHING:-0.03}"
```

Это снижает риск переобучения на synthetic/bootstrap preference pairs.

### 3. Training-time image cap стал evidence-aware

SFT/DPO/GRPO entrypoints теперь при `--max-images-per-example > 0` выбирают не первые N картинок, а evidence-aware top-k:

- совпадение имени файла с `evidence`, `metadata`, `reference_assertions_json`, `reference_temporal_json`;
- совпадение с `prompt`, `messages`, `claim`, `chosen`, `rejected`;
- bonus для `figure`, `fig`, `table`, `page`;
- стабильный tie-break по исходному порядку.

Raw full-data JSONL по-прежнему сохраняет все image refs. Ограничивается только training projection для памяти g2.2.

Изменённые entrypoints:

- `train_vlm_sft.py`
- `train_vlm_dpo.py`
- `train_vlm_grpo.py`

### 4. Reward functions для GRPO стали менее вырожденными

`train_vlm_grpo.py`:

- добавлена нормализация verdict aliases: `unsupported → reject`, `approved → accept`, `needs_revision → revise`;
- `reward_expert_override_match` теперь различает correct parsed JSON, wrong parsed JSON, partial/truncated JSON и missing verdict более плавно;
- `reward_label_exact_match` использует canonical label parsing;
- `reward_graph_consistency` активен только там, где действительно есть graph target, чтобы review-only rows не создавали искусственно мёртвый компонент.

### 5. Reward trace audit стал строже и честнее

`audit_reward_trace.py` и встроенный audit в `train_vlm_grpo.py` теперь дополнительно считают:

- `active_group_zero_std_fraction`
- `active_group_count`

Fail gate теперь ориентирован на активные, ненулевые reward groups, а не на компоненты, которые для данного task family должны быть нейтральными.

V2 wrapper передаёт:

```bash
--min-reward-std "${GRPO_MIN_REWARD_STD:-0.03}"
--max-zero-std-frac "${GRPO_MAX_ZERO_STD_FRAC:-0.80}"
```

## Рекомендуемый запуск

Используйте v2 job config, а не legacy full SFT→GRPO config:

```bash
datasphere project job execute \
  -p <PROJECT_ID> \
  -c experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_dpo_grpo_v2_g2_2.yaml
```

По умолчанию GRPO polish выключен:

```yaml
ENABLE_GRPO_POLISH: '0'
```

Это намеренно: DPO является основным alignment stage. Включайте GRPO только после проверки DPO-кандидата и reward variance:

```yaml
ENABLE_GRPO_POLISH: '1'
GRPO_MIN_REWARD_STD: '0.03'
GRPO_MAX_ZERO_STD_FRAC: '0.80'
```

## Проверки

Выполнено в текущем архиве:

```text
python -m py_compile build_scireason_alignment_datasets.py train_vlm_sft.py train_vlm_dpo.py train_vlm_grpo.py audit_reward_trace.py
bash -n run_hf_top_papers_sft_dpo_grpo_v2.sh
python -m pytest -q
```

Результат:

```text
183 passed, 6 skipped
```

## Ограничения

Полный GPU-прогон в этом окружении не выполнялся: здесь нет DataSphere g2.2 runtime и A100 GPU. Патч проверен статически, unit/regression тестами и совместимостью CLI/YAML.
