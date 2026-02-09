# Масштабирование на 200–300 экспертов

Этот документ — практический “операционный” план, как организовать работу студентов и аспирантов
для создания датасетов и улучшения SciReason.

## 1) Организационная схема

Рекомендуемый формат:

- **10–15 squads** по 15–25 человек
- в каждом squad:
  - 1 **Tech Lead** (инструменты, CI, блокеры)
  - 1 **Science Lead** (качество, терминология, онтология)
  - 1–2 **Reviewers** (дополнительный слой контроля качества)

## 2) Пул задач (что делают эксперты)

### A. Reasoning trajectories (SFT на “как думать”)

Цель: зафиксировать *правильный ход мыслей* на материалах литературы “до открытия”.

Артефакт: `data/experts/trajectories/**.yaml`.

**Definition of done:**
- 6–15 шагов
- в каждом шаге: claim + evidence + inference + next_question
- в `conditions`: все доменно‑важные граничные условия (температура/SOC/материал/...)

### B. Graph reviews (верификация KG)

Цель: обучить агента извлекать корректные утверждения и отбрасывать мусор.

Артефакт: `data/experts/graph_reviews/**.json`.

**Definition of done:**
- 10+ утверждений
- минимум 3 отклонённых
- минимум 1 добавленное/исправленное

### C. Hypothesis reviews (red teaming)

Цель: сформировать датасет “научной рецензии” и улучшить генерацию гипотез.

Артефакт: `data/experts/hypothesis_reviews/**.json`.

### D. MM reviews (multimodal)

Цель: улучшить извлечение смысла из графиков/таблиц/формул.

Артефакт: `data/experts/mm_reviews/**.json`.

### E. Temporal corrections (time grounding)

Цель: повышать качество временных привязок (год/период/эволюция эффекта).

Артефакт: `data/experts/temporal_corrections/**.json`.

## 3) Контроль качества

### Двойной контроль (2-pass)
- 1-й эксперт делает артефакт
- 2-й эксперт подтверждает/исправляет (не переписывая целиком)

### Gold set
- 1–3 “эталонных” файла на каждый тип артефакта
- используется для калибровки новых участников

### Автоматическая проверка

Перед PR:

```bash
pytest -q
python scripts/ci/validate_expert_artifacts.py
```

## 4) Как избегать хаоса в Git

### Naming convention
- `reviewer_id`: короткий стабильный ID (например, `mipt_squad03_07`)
- имена файлов:
  - trajectories: `<topic>__<reviewer_id>.yaml`
  - graph_reviews: `<paper_id>__<reviewer_id>.json`
  - hypothesis_reviews: `<run_id>__<reviewer_id>.json`

### PR policy
- маленькие PR (до 20 файлов)
- обязательный checklist: “evidence present?”, “conditions present?”, “no hallucinations?”

## 5) Data flywheel: от артефактов к fine-tuning

```bash
python scripts/data/export_training_datasets.py
```

Далее:
- SFT: учим извлечение/структурирование и стиль научного рассуждения
- RM/Preference: учим отбрасывать ошибки, наказывать галлюцинации

## 6) Минимизация юридических рисков

- не копировать большие фрагменты статей
- использовать короткие цитаты/пересказ
- хранить PDF отдельно (object storage), в git — только метаданные и маленькие артефакты
