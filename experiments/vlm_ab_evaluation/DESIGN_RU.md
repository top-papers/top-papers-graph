<!-- SPDX-FileCopyrightText: 2026 top-papers-graph contributors -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# Дизайн публикационного эксперимента

## 1. Исследовательский вопрос

Улучшает ли адаптер `top-papers/Qwen3-VL-8B-Instruct-scireason` качество ответов на задачи
извлечения мультимодальных научных свидетельств по сравнению с тем же
`Qwen/Qwen3-VL-8B-Instruct` без адаптера?

Сравнение оценивает **эффект применения опубликованного LoRA**, а не эффект всего end-to-end
конвейера `top-papers-graph`. Retrieval, OCR, список статей, изображения, prompt, processor,
precision, decoding и лимит ответа фиксируются. Меняются только веса активного адаптера.

## 2. Аудируемые артефакты

| Артефакт | Зафиксированная ревизия |
| --- | --- |
| Код `top-papers-graph` на момент исходного аудита, до evaluation implementation | `99ac6ca0b000494a3ba2438b3ea0854b4da863b0` |
| Benchmark | `33ccc5ed08e314c6457dcaa23e7f7508406cb4f8` |
| Train dataset | `1f1968b6604ffe12b9f6a4d1d5425ba19f097f14` |
| SciReason adapter | `45936868f4b2acdbbc4245044137072411fd11cc` |
| Evaluation base reconstruction | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |

`adapter_config.json` не фиксирует ревизию base model (`revision: null`). Поэтому указанная base
ревизия является воспроизводимой реконструкцией для оценки, но не криптографически доказанной
ревизией, использованной при обучении. В A/B это не создает межгрупповой confound, поскольку обе
группы используют одну реконструкцию, но ограничение обязательно указывается в статье.

Для новой publication revision `R` это ограничение больше не допускается: strict `prepare`
скачивает `adapter_config.json` из `R` и требует, чтобы `base_model_name_or_path` и `revision` точно
совпадали с immutable base model в config.

Каждый новый `prepare_manifest.json` отдельно фиксирует Git HEAD, dirty state и SHA256 inventory
всего evaluation source tree. Strict run разрешен только из clean tree; SHA исходного аудита выше
не является SHA добавленного evaluation implementation.

Contamination gate читает фактические `sft_all.jsonl`/`grpo_all.jsonl` из adapter revision `R`.
После публикации `R` release owner публикует в том же adapter repository отдельную immutable
attestation revision `M` с transformation-lineage manifest, который объявляет `R`. Inference
продолжает оценивать `R`, а `prepare` независимо фиксирует `M`. Разделение обязательно: manifest в
`R`, содержащий SHA самого `R`, создал бы неразрешимую самоссылку Git commit. Manifest фиксирует
path, SHA256 и число строк каждого training JSONL, а также непустые полные множества
paper/source/creator IDs и image/prompt SHA256. До этого полноту lineage нельзя считать доказанной
автоматически.

## 3. Почему текущий benchmark заблокирован

Authenticated `prepare` по скачанным байтам зафиксировал:

- 386 строк, 360 уникальных `sample_id`, 1 544 image references, 1 488 уникальных путей и только 30
  уникальных image contents;
- 1 546 critical findings, 938 warnings, 386 исключенных строк и ноль eligible samples;
- cross-paper reuse всех 360 samples в 29 группах одинаковых bytes;
- 26 лишних строк в 20 группах duplicate IDs; все 46 строк этих групп различаются как полные JSON,
  поэтому автоматически выбрать запись для сохранения нельзя;
- residual comparison wording во всех 386 строках и likely answer/evaluation leakage в 342 строках,
  затрагивающих 319 уникальных samples;
- schema errors во всех строках: prompt не совпадает с canonical user message и отсутствует
  `split_provenance`; дополнительно 29 строк имеют неверный `primary_endpoint`, одна строка не имеет
  canonical paper ID;
- 1 496 provenance rows без обязательных SHA256, page, locator, source URL, license и двух
  независимых verifier IDs; complete provenance отсутствует для всех 360 samples;
- пересечение 22 paper IDs между benchmark и фактическими SFT/GRPO export, затрагивающее 70
  уникальных samples и 71 benchmark row;
- отсутствие complete immutable training-lineage manifest в adapter release.

Проверенные hashes, детальные counts, ownership исправлений и acceptance criteria зафиксированы в
`remote_audit_baseline_20260717.json` и `REMEDIATION.md`. Ни один результат на этой ревизии нельзя
публиковать как доказательство превосходства модели. Режим `--exploratory` проверяет только
инженерный путь.

## 4. Требования к исправленному benchmark

1. Каждая единица имеет уникальный `sample_id` и неизменяемый canonical `paper_id`.
2. Каждое изображение связано с `(paper_id, page, figure/table locator, sha256, source URL,
   license)` и независимо проверено минимум двумя кураторами по исходной статье.
3. Prompt является самостоятельной задачей одной модели и не содержит A/B wording или ответа.
4. Ни article, source document, image bytes, prompt, expert/creator, ни их производные не входят ни
   в SFT, ни в GRPO. Код проверяет exact canonical paper IDs и exact normalized prompts по
   фактическим export; source/image/creator holdout дополнительно подтверждается обязательным
   split-lineage manifest и не выводится автоматически из имен файлов.
5. Split проводится на уровне статьи и источника, не строки.
6. Для automatic semantic metrics есть независимо adjudicated gold facts и rubric. При их
   отсутствии primary endpoint остается человеческим.
7. Для каждого изображения документированы лицензия, право распространения и citation.
8. До inference проходит schema, provenance, duplicate, hash и contamination gate.

Машинные контракты приведены в `schemas/publication_benchmark_row.schema.json`,
`schemas/publication_provenance_row.schema.json` и
`schemas/training_lineage_manifest.schema.json`. Benchmark и provenance публикуются отдельными
JSONL; для каждого уникального `sample_id` требуется ровно одна provenance row со всеми images.
`gold_answer` и `rubric` опциональны для primary human endpoint, но обязательны при
`benchmark.require_gold=true` и любых claims об automatic semantic quality.

## 5. Единицы и выборка

- **Experimental unit:** один frozen benchmark sample, одновременно поданный A и B.
- **Primary analysis cluster:** canonical paper ID.
- **Primary subset:** `primary_endpoint=true`, condition `original`, заранее определенные
  `multimodal_hard` и `temporal_hard` cases.
- **Reviewer design:** ровно два независимых эксперта; оба оценивают каждую единицу.
- **Counterbalancing:** для каждой единицы left/right у второго эксперта инвертирован, а общий
  случайный порядок примеров показан второму эксперту в обратной последовательности.
- **Easy controls:** анализируются отдельно и не входят в primary endpoint.

Все доступные прошедшие gate единицы включаются без post-hoc отбора по ответам моделей.
Preregistered minimum составляет 240 независимых primary papers при ожидаемой evaluable доле 0.90.

## 6. Estimand и гипотезы

Для каждого review:

- tuned win = 1;
- tie = 0.5;
- base win = 0;
- skip = missing и не входит в знаменатель.

Сначала оценки экспертов усредняются внутри sample, затем samples внутри paper, затем papers с
одинаковым весом. Primary estimand `theta` есть paper-macro вероятность предпочтения tuned с
половинным весом tie.

- weak-null `H0: theta = 0.5`;
- `H1: theta != 0.5` (двусторонний preregistered test, alpha 0.05).

Превосходство tuned можно заявить только когда audit/completeness gates пройдены, `theta > 0.5`,
двусторонний `p < 0.05` и нижняя 95% CI выше 0.5. Те же условия должны выполняться после
worst-case подстановки 0 за каждый пропущенный reviewer judgment. Статистическая незначимость не
доказывает эквивалентность; для equivalence нужен отдельно заданный margin и TOST.

## 7. Outcomes

### Primary

Слепое `overall_preference` на primary subset.

### Secondary

- `evidence_preference`;
- `visual_preference`;
- `temporal_preference`;
- hallucination, unsupported claim, wrong/missed evidence, visual и temporal error tags;
- JSON parse rate и output-schema validity;
- generation success и runtime как операционные показатели;
- breakdown по `stratum` и `evidence_kind`.

Secondary p-values корректируются Holm. Формальная валидность JSON не интерпретируется как
семантическое качество.

### Grounding controls

Обе модели дополнительно запускаются в `text_only` и `shuffled_images`. Число shuffled images
совпадает с original, а donor выбирается дерangement на уровне paper. Эти условия не входят в
primary endpoint. Семантическое падение на controls можно заявлять только по gold rubric либо по
отдельно ослепленной оценке; одних schema metrics недостаточно.

## 8. Inference protocol

- один и тот же pinned base model в A и B;
- один shared processor из adapter repository, включая `max_pixels=1003520`;
- явная загрузка PEFT и runtime-проверка активного adapter;
- confirmatory primary: unquantized FP16 base/compute, native FP32 PEFT LoRA, SDPA и
  `device_map=balanced` на двух T4; runtime отклоняет quantized base, не-FP16 base weights,
  не-FP32 LoRA weights, несовместимые missing/unexpected checkpoint keys, auxiliary adapter state,
  CPU/disk offload и использование не обеих GPU;
- primary decoding: greedy, `do_sample=false`, `max_new_tokens=768`;
- один и тот же prompt bytes и image SHA256;
- arms запускаются последовательно в отдельных процессах;
- outputs append-only, resumable только при полном совпадении fingerprints.

Complete inference manifest содержит SHA256 prediction JSONL и связывает его с config, frozen
benchmark, audit и evaluation source tree. Mock/limit overrides запрещены в strict mode.

Sampled decoding допустим как sensitivity analysis в отдельных run IDs: минимум пять
предопределенных seeds, одинаковых для A и B. Он не заменяет deterministic primary run.

Отдельный NF4 sensitivity run использует тот же FP32 LoRA и только автоматические diagnostics
(generation success, parse/schema validity и runtime). Он не выдается экспертам и не используется
для семантических выводов. Переход на NF4 при OOM/timeout primary run запрещен: это protocol
deviation и новый run, а не fallback. Его config содержит `precision_mode=nf4-sensitivity`, inference
manifest получает `result_scope=automatic_sensitivity_only`, а CLI отклоняет `blind`, `aggregate` и
`run`.

## 9. Blinding

Reviewer package не содержит model/adapter IDs, arm truth, source output filenames, creator
rationale, expected errors, sample IDs или owner key. Для каждого reviewer отдельно
контрбалансируются порядок и позиция. Reviewer видит исходную задачу, evidence images и два raw
ответа. В HTML встроена версионированная rubric; экспорт невозможен без всех обязательных полей,
краткого обоснования и подтверждения независимой работы.

Owner mapping хранится отдельно и HMAC-подписан закрытым randomization secret. Assignment IDs и
review exports связаны с fingerprint конкретных task/image/response bytes. Каждый пакет имеет
персональный nonce, поэтому экспорт одного эксперта нельзя принять под ID другого. Раскрытие проводится
только после получения всех запланированных review exports. При нарушении blinding затронутый
reviewer/run исключается целиком по заранее описанному protocol deviation, а не по направлению
результата.

## 10. Статистический анализ

- point estimate: reviewer -> sample -> paper macro average;
- 95% percentile bootstrap CI с resampling paper clusters;
- двусторонний large-sample paper-cluster mean test для weak-null `theta=0.5`;
- sign-flip test по paper effects только как sensitivity test более сильной sharp
  label-exchangeability null;
- nominal Krippendorff alpha для overlap reviews;
- exact binomial position-bias diagnostic;
- Holm correction для secondary endpoints;
- обязательные counts для wins/losses/ties/skips и всех исключений.

При малом числе papers CI и p-value считаются exploratory. Не следует заменять кластерный анализ
naive test по всем reviewer rows: это псевдорепликация.

## 11. Power

`plan` фиксирует conservative normal-approximation MDE по доле evaluable papers. Поскольку primary
unit является paper, повторные reviewers не считаются дополнительными независимыми units;
`n_items` трактуется как число независимых papers. ICC сохраняется как informational reviewer
design parameter. Значения variance после pilot можно обновить один раз до inference и
зафиксировать новым run ID/preregistration revision. Нельзя менять target N после просмотра arm
truth.

## 12. Исключения и missingness

До inference исключаются только заранее определенные технические нарушения: неуникальный ID,
небезопасный/отсутствующий image path, schema/placeholder mismatch. Любая contamination,
cross-paper byte reuse или residual comparison wording блокирует весь publication run.

После inference:

- generation errors сохраняются и считаются failures, а не тихо удаляются;
- invalid JSON показывается reviewer как raw response;
- `skip` разрешен только для действительно неоцениваемой единицы, требует текстовой причины и
  полностью репортится;
- worst/best-case sensitivity подставляет 0/1 за каждый отдельный `skip` до агрегации
  reviewer -> sample -> paper; claim gate требует worst-case point estimate, CI и p-value;
- missing assigned reviews блокируют publication result;
- никакого удаления outlier papers/reviewers после deblinding.

## 13. Публикационный пакет

Статья и supplement должны включать protocol/config, immutable revisions, полный audit,
CONSORT-like flow количества единиц, per-paper scores, primary/secondary table, agreement и
position diagnostics, error taxonomy, controls, compute/software environment и все protocol
deviations. Raw owner mapping и персональные данные reviewers публиковать нельзя; de-identified
review rows можно открыть при согласии и проверке лицензий.
