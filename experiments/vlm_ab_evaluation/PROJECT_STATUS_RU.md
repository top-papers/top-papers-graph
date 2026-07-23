<!-- SPDX-FileCopyrightText: 2026 top-papers-graph contributors -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# Статус подготовки A/B-теста на 2026-07-20

## Готово

- Проверены публичные ревизии base, SciReason LoRA, обучающих exports и benchmark.
- Реализован strict paired pipeline: audit, freeze, base/tuned inference, blinding, offline review,
  deblinding, paper-clustered analysis и immutable manifests.
- Confirmatory design требует ровно двух экспертов; оба оценивают каждый пример.
- Для второго эксперта стороны каждой пары инвертируются, а порядок примеров разворачивается.
- В `review.html` встроены rubric `vlm-ab-paired-v1.0`, обязательное обоснование, независимая
  аттестация и reviewer-specific nonce.
- Подготовлен private Kaggle API workflow для последовательных base/tuned kernels на двух T4.
- Подготовлен capacity protocol на 150 независимых primary papers.
- Зафиксирован основной endpoint: unquantized FP16 base/compute с native FP32 PEFT LoRA;
  `device_map=balanced` обязан использовать обе T4 без CPU/disk offload.
- NF4 зафиксирован как отдельный sensitivity run только с автоматическими diagnostics, без повторной
  оценки экспертами.
- Precision mode, exact `N=150`, two-reviewer overlap, model dtypes/device map и PEFT checkpoint keys
  проверяются исполняемыми gates; NF4 human-review commands отклоняются.
- Локальные профильные tests и Ruff проходят.

## Текущая remediation-поставка

Корень:

```text
runs/vlm_ab/qwen3vl-cap150-remediation-working-v1
```

Контрольные значения:

| Артефакт | SHA256 / значение |
| --- | --- |
| `prepare_manifest.json` | `a84cc9f68ba889e5e69a425ff9248a9de9a7d888e7e5e25fc8dd62178c6e5d50` |
| Queue fingerprint | `0143079579454f5181d0c02481975a242ed972df02ded6121b55930bacfbd5da` |
| `curation_queue/queue_manifest.json` | `7816a526be72b25885eb2b2b7fc30f217403c20854a749842ed5b0aeff81143d` |
| `curation_queue/tasks.jsonl` | `05e9ca069a3fba8cc6860090e260d8b9d8d88dfd35e3c7b775004ebd8789324a` |
| `curation_queue/decision_template.jsonl` | `d30eb357e5c37dd869205ac2af4db5ca28b3acfa5fcb0d31147520b8255583aa` |
| `curator_workspace_v2/workspace_manifest.json` | `39405f45fea20d82975c12149354ee5f823c2feb4577d0cb546b63e5ec53ca72` |
| `curator_workspace_v2/curator.html` | `d1f25914321af979342d7e98a0235a4d157985afd5334b71cf56eee1263e6f45` |
| Queue tasks | `386` |
| Training-overlap proposals | `74` tasks, `22` canonical paper IDs |
| Exact capacity candidates | `150`, `capacity_exact_target_available=true` |

Queue и уже выданные remediation-пакеты нельзя перегенерировать или редактировать после начала
ручной работы. Изменения evaluation-кода применяются только к будущему strict run с новыми IDs.

## Почему inference пока запрещен

Текущий benchmark commit `33ccc5ed08e314c6457dcaa23e7f7508406cb4f8` имеет ноль eligible
samples. Среди блокеров: comparison wording во всех 386 строках, schema errors, неполная provenance,
повторное использование image bytes между статьями, duplicate IDs, вероятная утечка evaluation
материала и 22 training-overlap paper IDs.

Также текущий adapter commit `45936868f4b2acdbbc4245044137072411fd11cc` не доказывает exact base
revision и не содержит полного immutable training-lineage manifest. Поэтому сейчас нельзя создавать
научно интерпретируемые base/tuned predictions или выдавать финальные пакеты экспертам.

## Следующий исполнимый этап

1. Назначить benchmark curators, не участвующих в финальной слепой оценке, если это возможно.
2. Завершить решения для 386 queue tasks и собрать ровно 150 проверенных non-overlap primary papers.
3. Опубликовать corrected benchmark как immutable revision `B`.
4. Опубликовать проверенный adapter/training release `R`, затем отдельную lineage attestation `M`.
5. Создать новый strict config с `B/R/M` и двумя pseudonymous expert IDs.
6. Зафиксировать exact FP16/FP32 model kwargs в новом config и preregister primary и NF4 protocols до
   просмотра outputs.
7. Выполнить clean `plan`, strict `prepare`, затем private Kaggle FP16 base и tuned kernels.
8. Выполнить отдельный NF4 run только для автоматических diagnostics.
9. Создать два blind packages только из FP16 primary, провести отдельную calibration на не-benchmark
   примерах и передать
   каждому эксперту только его opaque каталог плюс `EXPERT_REVIEW_RU.md`.

## Требуются решения и доступы

- Два ASCII pseudonymous ID финальных экспертов.
- Отдельные люди для ручной курации benchmark либо явное решение использовать тех же экспертов с
  документированным риском знакомства с материалом.
- Размер confirmatory набора подтвержден: exact `N=150`.
- Основной Kaggle T4 режим подтвержден: unquantized FP16 base/compute с native FP32 LoRA; NF4 только
  automatic sensitivity diagnostics.
- Hugging Face write access для публикации `B`, `R` и `M`.
- Kaggle API credential в `%USERPROFILE%\.kaggle\kaggle.json`, доступ к Internet и квота T4x2.
- Подтверждение прав на распространение изображений и лицензий corrected benchmark/adapter.
- Небольшой отдельный calibration-набор, не входящий в confirmatory benchmark.
