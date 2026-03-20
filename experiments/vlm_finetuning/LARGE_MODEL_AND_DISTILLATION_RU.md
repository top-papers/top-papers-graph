# Схема дообучения большой Qwen3-VL и дистилляции в маленькую модель

## 1. Какие крупные модели стоит рассматривать

Практически значимы:

- `Qwen3-VL-30B-A3B-Instruct` — основной локальный teacher-кандидат в лимите до `4x A100`.
- `Qwen3-VL-32B-Instruct` — плотный teacher-кандидат, но менее комфортный по памяти и цене.
- `Qwen3-VL-235B-A22B-Instruct` — сильный teacher, но не реалистичен для локального дообучения в DataSphere при лимите `4x A100`.

## 2. Рекомендуемая схема для большой модели

### Teacher

- **`Qwen3-VL-30B-A3B-Instruct + LoRA`**.

Последовательность:

1. `SFT` на gold expert records.
2. `DPO` на corrections и reviews.
3. Узкий `GRPO` только на reward-verifiable subset.

Teacher используется как high-precision extractor и labeler для silver-данных.

## 3. Почему teacher надо учить отдельно

Teacher позволяет расширить датасет за счёт:

- teacher-generated silver labels;
- self-consistency filtering;
- disagreement mining;
- high-agreement pseudo-labels.

Это главный способ масштабировать обучение при 205 экспертах.

## 4. Схема дистилляции

### Выбор типа дистилляции

Для этого кейса лучше **sequence-level distillation**, а не offline logit distillation:

- меньше storage/I-O;
- проще для multimodal pipeline;
- проще смешивать gold и silver;
- лучше сочетается с экспертным review loop.

### Поток данных

1. Дообучить teacher на gold.
2. Сгенерировать silver corpus на дополнительных страницах / фигурах / таблицах.
3. Отфильтровать silver по quality gates.
4. Смешать `gold + silver`.
5. Обучить student через обычный `SFT`.

### Рекомендуемый student

- Первый student: **`Qwen3-VL-4B-Instruct`**.
- Второй этап после валидации 4B: **`Qwen3-VL-2B-Instruct`**.

### Смесь данных

- `gold_weight = 1.0`
- `silver_weight = 0.25–0.40`

## 5. Критерий успеха дистилляции

4B student должен сохранять >=85–90% выигрыша teacher относительно базового 4B/3B baseline по:

- triplet F1
- temporal F1
- evidence grounding precision
- graph acceptance rate
- contradiction rate
- downstream usefulness

## 6. Что добавлено в bundle

- `scripts/train_vlm_sft.py`
- `scripts/train_vlm_dpo.py`
- `configs/teacher_sft_qwen3vl_30b_a3b_lora.yaml`
- `configs/student_distill_qwen3vl_4b_from_teacher.yaml`
- `configs/student_distill_qwen3vl_2b_from_teacher.yaml`
- `datasphere/bin/run_teacher_sft_30b_a3b.sh`
- `datasphere/bin/run_student_distill_4b.sh`
