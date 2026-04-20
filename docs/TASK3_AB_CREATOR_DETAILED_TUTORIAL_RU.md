# Task 3 A/B: подробный туториал для эксперта-создателя тестового набора

## 1. Новая схема ролей

В новой постановке A/B теста роли разделены:

- **эксперт-создатель набора** собирает curated hard subset и формализует кейсы;
- **CLI runner** автоматически строит blind A/B пакет по этим кейсам;
- **эксперт-участник** заполняет уже готовую blind-форму и не знает, где baseline, а где SFT/DPO.

Это нужно для того, чтобы эффект VLM мерился именно на тех участках пайплайна, где он обязан проявляться: multimodal evidence extraction, evidence linking и temporal grounding.

## 2. Что должен подготовить эксперт-создатель

До начала работы подготовьте:

1. Репозиторий `top-papers-graph` или ZIP-архив репозитория.
2. Curated `processed_papers` для поднабора статей.
3. Минимальный список статей / identifiers для trajectory YAML.
4. Черновое понимание, какие кейсы у вас относятся к:
   - `multimodal_hard`
   - `temporal_hard`
   - `easy_control`

## 3. Что делает Colab notebook

В репозитории добавлен notebook:

- `notebooks/task3_ab_testset_authoring_colab.ipynb`

Он делает два независимых артефакта:

1. **trajectory bundle** — минимальный совместимый YAML/ZIP для Task 3 runner;
2. **authoring bundle** — offline HTML + JSON template, в которых эксперт-создатель описывает кейсы будущего A/B теста.

## 4. Какие файлы должны получиться

После Colab-шага у вас должны быть:

- `task1_minimal_for_ab_bundle.zip`
- `task3_ab_creator_bundle.zip`
- `task3_ab_case_manifest.template.json`
- `task3_ab_creator_offline_form.html`

После заполнения offline формы должен появиться ещё один файл:

- `task3_ab_case_manifest.filled.json`

Именно `task3_ab_case_manifest.filled.json` — главный вход для Kaggle/CLI/DataSphere runner.

## 5. Какого состава должен быть тестовый набор

Рекомендуемая пропорция:

- 60% `multimodal_hard`
- 25% `temporal_hard`
- 15% `easy_control`

Практически:

- для пилота: 12–20 кейсов;
- для полноценного прогона: 40–80 кейсов.

## 6. Что такое один кейс

Один кейс — это **одна сравнительная единица**, на которой участник потом сравнит Variant A и Variant B.

Кейс должен описывать:

- какую именно hypothesis unit / candidate unit вы хотите поймать;
- какой тип сложности там важен;
- на что участник должен смотреть в первую очередь.

## 7. Какие поля обязательно заполнить в каждом кейсе

Обязательные поля:

- `case_id`
- `stratum`
- `paper_title`
- `paper_id`
- `evidence_kind`
- `page_hint`
- `creator_prompt`
- `review_focus`
- `expected_error_modes`
- хотя бы одно поле из блока `match.*`

## 8. Как правильно заполнять `creator_prompt`

`creator_prompt` — это короткая инструкция для эксперта-участника.

Хорошие примеры:

- «Проверь, извлечён ли факт из таблицы и не потеряна ли численная тенденция»
- «Проверь временной интервал и связь evidence с figure/table»
- «Смотри, не появился ли визуальный факт, которого нет на странице»

Плохие примеры:

- «Здесь должен победить tuned VLM»
- «A лучше, потому что baseline часто ошибается»

Участник не должен видеть ожидаемого победителя.

## 9. Как правильно заполнять `review_focus`

Записывайте туда только то, что реально нужно оценивать на этом кейсе.

Например:

- `evidence, visual_fact`
- `temporal, evidence`
- `overall`

## 10. Как правильно заполнять `expected_error_modes`

Допустимые категории в новом дизайне:

- `missed_visual_fact`
- `wrong_evidence_linkage`
- `needs_time_fix`
- `hallucinated_visual_inference`

Они потом прямо появятся в blind-форме участника как отдельные поля оценки.

## 11. Как работает блок `match.*`

Runner автоматически сопоставляет кейс с hypothesis rows по match-полям.

Лучший вариант — точный `candidate_signature`, если вы его знаете.

Если нет, используйте комбинацию:

- `candidate_source_contains`
- `candidate_predicate_contains`
- `candidate_target_contains`
- `hypothesis_title_contains`
- `time_scope_contains`
- `rank_hint`

### Важное правило

Не оставляйте кейс совсем без `match.*`.

Если кейс не содержит ни одного usable match-поля, runner не сможет надёжно подобрать соответствующий output у обеих моделей.

## 12. Пошаговый сценарий работы

### Шаг 1. Запустите Colab notebook

Откройте:

- `notebooks/task3_ab_testset_authoring_colab.ipynb`

Заполните:

- тему
- cutoff year
- список identifiers
- имя/ID эксперта
- размер будущего набора

### Шаг 2. Скачайте trajectory bundle

После запуска notebook скачайте:

- `task1_minimal_for_ab_bundle.zip`

Это вход для Task 3 runner.

### Шаг 3. Скачайте authoring bundle

Также скачайте:

- `task3_ab_creator_bundle.zip`

Или хотя бы:

- `task3_ab_creator_offline_form.html`
- `task3_ab_case_manifest.template.json`

### Шаг 4. Откройте offline HTML локально

Откройте `task3_ab_creator_offline_form.html` в браузере.

Форма:

- хранит черновик локально;
- позволяет добавлять и удалять кейсы;
- умеет экспортировать заполненный JSON.

### Шаг 5. Заполните hard cases

Приоритетно создавайте:

- figure/table-heavy кейсы;
- temporal-hard кейсы;
- случаи, где baseline, скорее всего, теряет visual fact или ломает evidence linkage.

### Шаг 6. Экспортируйте `task3_ab_case_manifest.filled.json`

Это основной результат работы создателя набора.

### Шаг 7. Передайте runner'у три входа

Нужно передать:

1. trajectory YAML/ZIP;
2. filled case manifest JSON/ZIP;
3. curated `processed_papers`.

## 13. Как запускать CLI locally / on Kaggle

В репозитории добавлен runner:

- `scripts/task3/run_task3_case_based_blind_ab.py`

Пример:

```bash
python scripts/task3/run_task3_case_based_blind_ab.py \
  --trajectory inputs/trajectory.yaml \
  --case-manifest inputs/task3_ab_case_manifest.filled.json \
  --processed-dir inputs/processed_papers \
  --out-dir runs/task3_case_based_blind_ab \
  --model-a-vlm-model-id <baseline_vlm> \
  --model-b-vlm-model-id <finetuned_vlm> \
  --run-vlm
```

## 14. Что появится после запуска runner'а

Runner создаёт:

- `task3_case_based_run_manifest.json`
- `expert_review/offline_review/task3_case_based_blind_review.html`
- `owner_only/task3_case_based_blind_key.json`
- `expert_case_based_blind_review_bundle.zip`

## 15. Что можно передавать участнику, а что нельзя

### Можно передавать участнику

- `task3_case_based_blind_review.html`
- `task3_case_based_blind_review_manifest.json`
- `expert_case_based_blind_review_bundle.zip`

### Нельзя передавать участнику

- `task3_case_based_blind_key.json`
- любые заметки, где раскрыто, какая модель baseline, а какая tuned
- ваши внутренние ожидания победителя по кейсам

## 16. Что оценивает эксперт-участник

На каждом кейсе участник должен заполнить:

- `preferred_variant`
- `better_evidence`
- `better_temporal`
- `missed_visual_fact_by`
- `wrong_evidence_linkage_by`
- `needs_time_fix_by`
- `hallucinated_visual_inference_by`
- `confidence`
- `comments`

## 17. Как считать метрики

### Primary endpoint

Считать только по кейсам, где:

- `primary_endpoint = true`
- `stratum in {multimodal_hard, temporal_hard}`

### Secondary metrics

- частота `missed_visual_fact`
- частота `wrong_evidence_linkage`
- частота `needs_time_fix`
- частота `hallucinated_visual_inference`

### Easy controls

Используйте как sanity-check, но не как главный KPI.

## 18. DataSphere Jobs

Для DataSphere добавлены:

- `scripts/datasphere/run_task3_case_based_blind_ab_job.py`
- `configs/datasphere/task3_case_based_blind_ab_job.config.yaml`

Общий сценарий:

1. установить `datasphere` CLI;
2. подготовить job config с `cmd`, `inputs`, `outputs`, `env`;
3. запустить:

```bash
datasphere project job execute -p <project_id> -c configs/datasphere/task3_case_based_blind_ab_job.config.yaml
```

## 19. Короткий checklist перед handoff

- [ ] у каждого кейса заполнен `creator_prompt`
- [ ] у каждого кейса есть хотя бы одно `match.*`
- [ ] hard-кейсы составляют большинство
- [ ] `primary_endpoint` не включён на easy-control кейсах
- [ ] ожидаемый победитель нигде не раскрыт
- [ ] owner key не попадёт участнику
