# Дизайн дообучения VLM для извлечения научных данных в темпоральный граф знаний

## 1. Цель

Нужно дообучить VLM так, чтобы модель лучше извлекала из научных статей:

- утверждения в формате `(subject, predicate, object)`;
- временные атрибуты (`start_date`, `end_date`, `valid_from`, `valid_to`, `time_source`);
- условия эксперимента;
- связь между текстом, таблицами, рисунками и итоговым утверждением;
- локальное научное рассуждение вида `claim -> evidence -> inference -> next_question`.

Система должна улучшить не только локальную extraction quality, но и downstream-качество:

- качество автоматически собранного temporal KG;
- долю корректных рёбер после экспертной валидации;
- качество аналитики поверх графа;
- качество гипотез, построенных на графе.

## 2. Что уже есть в репозитории

В репозитории уже присутствуют важные заготовки:

- expert trajectories (`data/experts/trajectories`) как источник SFT-данных;
- graph reviews, MM reviews и temporal corrections как источник corrective supervision;
- exporter `scripts/data/export_training_datasets.py` для текстовых SFT-наборов;
- мультимодальный контур с `qwen2_vl`/`qwen3_vl` backend;
- Task 2 validation pipeline и notebook для сравнения reference graph и auto graph;
- rule-based reward / overrides из экспертных review.

То есть репозиторий уже содержит data flywheel, но не хватает специализированного контура именно
для VLM fine-tuning, multimodal preferences и системной экспериментальной матрицы.

## 3. Что именно должна учить модель

### 3.1 Основной навык

Модель должна учиться не “писать всё про статью”, а выполнять узко определённые шаги:

1. увидеть релевантный текст, таблицу или рисунок;
2. извлечь структурированный факт;
3. привязать факт к evidence;
4. определить temporal scope;
5. не выходить за пределы evidence;
6. уметь отказаться от уверенного ответа при слабой evidence.

### 3.2 Подзадачи обучения

Подзадачи лучше разделить на отдельные task family:

- **Step completion**: по claim + sources + conditions предсказать inference и next_question.
- **Structured extraction**: по странице PDF / figure / table + prompt вернуть JSON с assertion(s).
- **Temporal grounding**: дополнить утверждение корректными temporal fields.
- **Multimodal evidence linking**: связать объект extraction с figure/table/page/snippet.
- **Graph repair**: по auto assertion + expert feedback выдать corrected assertion.
- **Preference alignment**: предпочитать expert-corrected output вместо baseline-output.

## 4. Как использовать данные от 205 экспертов

Предполагается минимум 205 trajectory-like артефактов, по одному и более от каждого эксперта,
и дополнительная доля валидационных артефактов из Task 2.

### 4.1 Что считаем единицей обучения

Один экспертный YAML — это не один train sample. Его нужно развернуть в несколько sample unit:

- sample на каждый `step`;
- sample на каждый `source` внутри step;
- sample на каждое temporal assertion;
- sample на каждое corrective review событие;
- sample на каждую multimodal verification задачу.

Практически один качественный YAML на 6–12 шагов может дать 10–40 обучающих примеров,
если грамотно распаковать его по task family.

### 4.2 Типы сигналов

#### A. SFT-сигнал
Из trajectories и MM reviews:

- правильное форматирование;
- дисциплина evidence grounding;
- научный стиль краткой интерпретации;
- распознавание figure/table-driven facts.

#### B. Preference / correction signal
Из graph reviews, temporal corrections, Task 2 validation:

- rejected vs accepted;
- original vs corrected;
- weakly grounded vs well grounded;
- wrong time scope vs fixed time scope.

#### C. RL-compatible signal
Из автоматических проверок:

- валиден ли JSON;
- парсится ли temporal schema;
- совпадает ли assertion с expert-approved review;
- нет ли temporal contradictions;
- есть ли evidence reference.

## 5. Рекомендуемая стратегия: SFT -> DPO -> узкий RL

## 5.1 Почему не full fine-tuning первым этапом

При лимите до `4x A100` и первом цикле экспериментов full fine-tuning выглядит не лучшей стартовой
точкой: выше стоимость, больше риск нестабильности, выше требования к объёму и чистоте данных.
Для первого контура лучше использовать adapter-based fine-tuning.

## 5.2 Почему основной контур должен быть SFT + DPO

Потому что доступные экспертные данные по своей природе именно такие:

- trajectories дают supervised targets;
- reviews и corrections дают natural preference signal;
- notebook validation даёт corrective supervision.

То есть основной signal уже офлайн и хорошо ложится на:

1. **SFT** — чтобы научить формат, структуру и evidence discipline;
2. **DPO** — чтобы научить предпочитать экспертно-корректные extraction outputs.

## 5.3 Где RL уместен

RL уместен только на узком, проверяемом контуре, где reward можно формализовать.
Например:

- reward за schema-valid JSON;
- reward за корректное заполнение temporal fields;
- reward за совпадение с expert-approved assertion после canonicalization;
- penalty за hallucinated fields;
- penalty за отсутствие evidence.

Но RL не должен быть первым шагом. Иначе велик риск reward hacking: модель научится выдавать
формально валидный JSON без реального улучшения semantic extraction.

## 6. Выбор модели

### 6.1 Практический выбор для первого цикла

**Предпочтительный кандидат:** `Qwen3-VL-4B-Instruct`.

Причины:

- достаточно компактная модель для PEFT-обучения в лимите `4x A100`;
- современная архитектура с улучшенной пространственно-временной обработкой;
- сильный multimodal backbone;
- лучше соответствует задаче temporal grounding и анализу figure/table-rich статей.

### 6.2 Операционный fallback

Если стек в конкретной среде ещё нестабилен под Qwen3-VL, то операционный baseline для первого
smoke/pilot запуска — `Qwen2.5-VL-3B-Instruct`.

Причины:

- стабильные примеры обучения в TRL;
- ниже риск несовместимости кода/версий;
- достаточно дешёвый smoke baseline.

## 7. Режимы обучения

### 7.1 Stage A — SFT (обязательный)

Обучаем на смеси:

- trajectory step completion;
- multimodal extraction;
- temporal extraction;
- graph repair demonstrations.

#### Целевые форматы output

Нужно обучать не скрытую chain-of-thought, а контролируемый observable output:

- JSON assertions;
- краткий rationale максимум 1–3 предложения;
- evidence pointers;
- temporal fields.

Причина: скрытые reasoning traces трудно валидировать и они плохо подходят как основной production
contract. Основной контракт должен быть структурированным.

### 7.2 Stage B — DPO (рекомендуется)

DPO строим на парах:

- `chosen = expert-corrected output`
- `rejected = baseline output` или `expert-rejected output`

Источники пар:

- graph reviews;
- temporal corrections;
- MM reviews;
- notebook validation bundles.

#### Что именно оптимизируем

- меньше hallucinations;
- лучше evidence alignment;
- лучше temporal grounding;
- меньше лишних утверждений;
- больше точность на figure/table-based facts.

### 7.3 Stage C — Optional RL / GRPO (узко и после DPO)

Запускаем только после того, как:

- SFT и DPO уже дали выигрыш на offline validation;
- reward проверен на correlation с expert preference;
- есть стабильная inference generation loop.

GRPO имеет смысл для задач, где можно быстро и автоматически считать reward на больших партиях.
Например:

- schema validity;
- canonicalized triplet match;
- temporal consistency;
- graph consistency;
- abstention when evidence is weak.

## 8. Как строить обучающие датасеты

## 8.1 SFT dataset families

### Family 1. Trajectory reasoning SFT

Вход:

- topic;
- list of papers;
- step claim;
- source summary;
- conditions;
- page/figure context, если доступен.

Выход:

- inference;
- next_question;
- optional structured extraction block.

### Family 2. Page-to-assertion VLM SFT

Вход:

- image страницы / figure / table;
- page text or OCR/snippet;
- prompt на извлечение.

Выход:

- JSON assertions + evidence locator + temporal fields.

### Family 3. Review-to-repair SFT

Вход:

- auto assertion;
- evidence;
- expert verdict / rationale.

Выход:

- corrected assertion или explicit rejection.

### Family 4. Temporal fix SFT

Вход:

- assertion;
- evidence;
- wrong temporal fields.

Выход:

- corrected temporal fields.

## 8.2 Preference dataset families

Каждая запись должна содержать:

- prompt;
- images или image;
- chosen;
- rejected;
- task_type;
- optional scalar metadata for analysis.

Наиболее ценные preference пары:

1. baseline extraction vs expert-fixed extraction;
2. evidence-grounded vs hallucinated;
3. correct temporal scope vs wrong temporal scope;
4. concise structured output vs verbose unstructured output.

## 8.3 Splits

Нельзя сплитить случайно по sample. Нужно сплитить по экспертам и по темам.

Рекомендуемые срезы:

- **train / val / test by expert**;
- **leave-domain-out** test;
- **multimodal-heavy** subset;
- **text-only** subset;
- **figure/table** subset;
- **temporal-hard** subset.

Рекомендация:

- 70% экспертов train
- 15% экспертов val
- 15% экспертов test

Плюс отдельный маленький **gold holdout**, который не участвует ни в SFT, ни в DPO.

## 9. Data mixture

На старте не надо делать равномерную смесь “всё со всем”. Нужен управляемый mixture.

Рекомендуемый старт:

- 45% trajectory SFT
- 20% multimodal extraction SFT
- 15% graph repair SFT
- 10% temporal fix SFT
- 10% hard negative / abstention SFT

Для DPO:

- 40% graph review preferences
- 30% temporal correction preferences
- 20% multimodal review preferences
- 10% hypothesis-support extraction preferences

## 10. Practical PEFT design

### 10.1 Базовый вариант

- QLoRA или LoRA;
- rank 16–64;
- target modules: attention + MLP text stack; при необходимости часть vision projection modules;
- bf16/fp16 depending environment;
- gradient checkpointing;
- packing отключить, если есть риск потери image tokens;
- `max_length=None` для VLM SFT.

### 10.2 Что не делать в первом цикле

- full FT всей модели;
- aggressive RL на сырых reward;
- смешивание слишком большого числа losses одновременно;
- обучение на длинных свободных CoT как основном target.

## 11. Как встроить multimodal input

Есть два принципиальных режима.

### Режим 1. Page-centric

Используем страницу PDF как изображение + текст страницы.
Подходит для:

- рисунков и таблиц, встроенных в страницу;
- extraction в стилях page QA / page grounding.

### Режим 2. Figure-centric

Предварительно режем PDF на figure/table crops.
Подходит для:

- figure-heavy papers;
- точного обучения на конкретных графиках и таблицах.

Рекомендация:

- стартовать с page-centric режимом;
- добавить figure-centric subset для hardest multimodal samples.

## 12. Метрики

## 12.1 Локальные extraction metrics

- Assertion exact match / macro F1.
- Subject / predicate / object component F1.
- Temporal field exact match / partial F1.
- Evidence grounding precision.
- Hallucination rate.
- Abstention quality on low-evidence samples.

## 12.2 Graph-level metrics

- precision / recall рёбер относительно reference graph;
- accept rate на expert validation;
- repair success rate;
- temporal contradiction rate;
- graph density drift;
- cycle / consistency diagnostics.

## 12.3 Downstream metrics

- качество complex-network analytics на auto KG vs reference KG;
- изменение ranking узлов / рёбер;
- качество hypothesis generation при одном и том же hypothesis pipeline, но разном extractor;
- экспертная оценка гипотез по шкале полезности / новизны / обоснованности.

## 13. Offline validation loop

Для каждого run нужно сохранять:

- model adapter/checkpoint;
- exact data mixture config;
- prompt template version;
- extraction outputs on fixed benchmark;
- graph build outputs;
- validation summaries.

Сравнение делаем не по одной метрике, а по каскаду:

1. extraction;
2. temporal grounding;
3. graph review acceptance;
4. downstream hypothesis utility.

## 14. Что считать успешным первым этапом

Первый цикл можно считать успешным, если одновременно выполнены условия:

- +10–20% relative improvement на temporal field correctness;
- -20% hallucination rate;
- заметный рост expert-accepted assertions на Task 2 validation;
- улучшение не только на text-only, но и на image/table subset;
- отсутствие деградации на unseen experts/domains.

## 15. Go / No-Go для RL

Запускать RL только если:

- reward коррелирует с expert verdict;
- DPO уже дало стабильный выигрыш;
- schema-validity почти насыщена и нужно добирать harder behavior;
- есть ресурсы минимум под несколько последовательных trial run.

Если reward correlation слабая, RL откладываем и усиливаем DPO/curation.

## 16. Итоговая рекомендация

### Лучший стартовый путь

1. `Qwen3-VL-4B-Instruct` + LoRA/QLoRA SFT.
2. Затем DPO на expert corrections/preferences.
3. Только потом narrow GRPO на reward-verifiable subset.

### Почему это оптимально

- соответствует типу доступных данных;
- помещается в лимит до `4x A100`;
- даёт быстрый feedback loop;
- минимизирует риск дорогостоящего и шумного RL на старте;
- напрямую привязан к существующим артефактам репозитория.


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


## 13. Реальные training entrypoints и orchestration

В bundle добавлены рабочие entrypoints:

- `scripts/train_vlm_sft.py`
- `scripts/train_vlm_dpo.py`

Они построены на `TRL` и поддерживают:

- `LoRA` и `QLoRA`;
- `Qwen2.5-VL` и `Qwen3-VL`;
- text-only режим для trajectory-heavy datasets;
- VLM-режим для datasets с колонкой `image`;
- запуск через `torchrun` в DataSphere jobs.

Дополнительно добавлены:

- dataset builders `build_vlm_sft_dataset.py` и `build_vlm_preference_dataset.py`;
- DataSphere shell-entrypoints в `datasphere/bin/`;
- job-конфиги для `sft_smoke`, `sft_pilot`, `dpo_pilot`, `teacher_sft_qwen3vl_30b_a3b`, `student_distill_qwen3vl_4b`.

## 14. Схема для большой модели и дистилляции

Подробная схема вынесена в отдельный документ:

- `LARGE_MODEL_AND_DISTILLATION_RU.md`

Кратко:

- большой teacher-кандидат: `Qwen3-VL-30B-A3B-Instruct + LoRA`;
- локально под `4x A100` это реалистичнее, чем 235B;
- дистилляцию лучше строить как sequence-level distillation `teacher -> silver corpus -> student 4B -> optional 2B`.
