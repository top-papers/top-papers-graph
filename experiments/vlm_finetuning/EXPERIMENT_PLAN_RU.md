# План экспериментов и валидации

## 1. Базовая матрица запусков

### Run 0 — Baseline

Цель: зафиксировать исходное качество текущего пайплайна без дообучения.

Считаем:

- extraction metrics;
- temporal metrics;
- graph acceptance metrics;
- downstream hypothesis metrics.

### Run 1 — Smoke SFT

Цель: проверить, что весь контур обучения корректно работает end-to-end.

Конфигурация:

- 1x A100;
- 1–5k train samples;
- 200–500 val samples;
- 0.5–1 epoch equivalent;
- один adapter run.

Критерий успеха:

- обучение стабильно;
- loss убывает;
- модель не ломает JSON contract;
- inference после обучения проходит benchmark без деградации.

### Run 2 — Pilot SFT

Цель: получить первый реальный quality lift.

Конфигурация:

- full currently available SFT mixture;
- 2x A100;
- несколько seed или хотя бы 2 learning-rate setting;
- evaluation на fixed holdout.

Критерий успеха:

- рост по extraction и temporal metrics;
- снижение hallucination rate;
- улучшение expert validation acceptance.

### Run 3 — Comparison SFT

Сравниваем:

- Qwen2.5-VL-3B vs Qwen3-VL-4B;
- LoRA vs QLoRA;
- text-only SFT vs multimodal SFT;
- with vs without graph repair samples.

### Run 4 — DPO Pilot

Цель: понять, даёт ли preference alignment дополнительный выигрыш поверх лучшего SFT.

Сравниваем:

- best SFT checkpoint;
- best SFT + DPO.

### Run 5 — Optional Narrow GRPO

Цель: проверить только reward-verifiable improvements.

Только после Run 4.

## 2. Какими группами мерить успех

### Group A. Extraction quality

- triplet EM/F1
- temporal EM/F1
- evidence grounding precision
- hallucination rate

### Group B. Graph quality

- assertion acceptance rate
- corrected-after-review rate
- contradiction rate
- missing-time rate

### Group C. Downstream utility

- graph analytics stability
- hypothesis support quality
- expert usefulness score

## 3. Наборы holdout

### Holdout 1. Unseen experts
Проверяет, не выучила ли модель стиль конкретных аннотаторов.

### Holdout 2. Unseen domains
Проверяет перенос на новые научные области.

### Holdout 3. Multimodal hard subset
Figure/table-heavy статьи.

### Holdout 4. Temporal hard subset
Сложные кейсы со временем: интервалы, delayed effects, historical comparisons.

## 4. Рекомендуемые абляции

1. Убрать image input и оставить только текст.
2. Убрать temporal-fix samples.
3. Убрать graph-repair samples.
4. Убрать DPO и оставить только SFT.
5. Добавить/не добавлять hard negatives.

Это позволит понять, что именно даёт прирост.

## 5. Принципы оценки

- сравнивать модели на одном и том же frozen benchmark;
- хранить per-sample outputs;
- валидировать не только средние метрики, но и длинный хвост ошибок;
- отдельно смотреть figure/table кейсы.

## 6. Error taxonomy

Для каждого запуска нужен breakdown по типам ошибок:

- hallucinated triplet;
- wrong predicate;
- wrong temporal scope;
- wrong evidence linkage;
- missed figure/table fact;
- over-extraction;
- under-extraction;
- invalid JSON/schema.

## 7. Go / No-Go критерии

### Go for DPO

- SFT уже улучшил core metrics;
- качество пар достаточное;
- есть хотя бы несколько сотен качественных chosen/rejected examples.

### Go for RL

- reward стабилен и коррелирует с expert judgments;
- DPO уже не даёт нужного прироста;
- есть бюджет на несколько итераций.

### No-Go

- если прирост виден только на seen experts;
- если растёт формальная валидность JSON, но не semantic quality;
- если сильная деградация на multimodal hard subset.


## Привязка к DataSphere CLI

Для запуска экспериментов в Yandex DataSphere Jobs теперь добавлен отдельный orchestration layer:

- `datasphere/job_configs/*.yaml` — job-конфиги под `datasphere project job execute -p <project_ID> -c <config.yaml>`.
- `datasphere/bin/*.sh` — исполняемые shell entrypoints внутри job.
- `datasphere/launch_examples.sh` — обёртка над `execute`, `list`, `get`, `attach`, `cancel`, `set-data-ttl`, `download-files`.

Рекомендуемый порядок запуска:
1. `build_datasets.yaml`
2. `sft_smoke.yaml`
3. `sft_pilot.yaml`
4. `dpo_pilot.yaml`
5. `validate_extraction.yaml`

Таким образом, ML-конфиги в `configs/*.yaml` остаются source-of-truth для гиперпараметров,
а DataSphere job-конфиги становятся source-of-truth для cloud orchestration.


## 8. Дополнительные teacher / distillation runs

### Run 6 — Teacher SFT 30B-A3B

Цель: получить большой teacher для silver generation и оценки потолка качества.

Конфигурация:

- `Qwen3-VL-30B-A3B-Instruct + LoRA`;
- `g2.4`;
- полный gold SFT mixture.

### Run 7 — Student Distill 4B

Цель: проверить, можно ли перенести большую часть выигрыша teacher в более дешёвый student.

Конфигурация:

- teacher-generated silver corpus + gold;
- `Qwen3-VL-4B-Instruct + LoRA`;
- сравнение `gold-only` vs `gold+silver`.

### Run 8 — Optional Student Distill 2B

Только если 4B student уже показал хороший transfer.
