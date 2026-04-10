# Отчёт по реализации модуля Task 3

## Что было проанализировано

### Репозиторий

Внутри репозитория уже существовали необходимые базовые строительные блоки:

- поиск и нормализация статей
- загрузка PDF и multimodal ingest
- построение temporal KG
- базовый TGNN/TGN-style temporal link prediction
- генерация гипотез
- готовый Task 2 bundle-паттерн

Из этого следует, что для Task 3 правильнее было делать не отдельный разрозненный скрипт, а **полноценный orchestrator/bundle**, совместимый с текущей архитектурой.

### Блокноты заданий

#### Task 2 notebook
Показал, что в проекте уже есть рабочий паттерн:
- `task2-bundle`
- поддержка `g4f`
- поддержка локальных HF/transformers VLM
- offline-first режим
- сохранение bundle-артефактов

Этот паттерн был перенесён на Task 3.

#### Task 1 notebook
Показал ожидаемый входной артефакт траектории/темы. Поэтому Task 3 умеет принимать `trajectory YAML` напрямую.

### Экспертные данные

Экспертные YAML-файлы показали, что целевой workflow должен уметь работать с:
- темпоральной аргументацией
- мультимодальными свидетельствами (текст + изображения + таблицы)
- длинными reasoning-цепочками
- выходом в виде проверяемых гипотез

Особенно важны были следующие особенности экспертных данных:
- траектории состоят из временно упорядоченных шагов и переходов между ними;
- evidence может идти не только из текста, но и из figure/table источников;
- конечный результат должен быть пригоден для формулирования testable hypotheses.

## Что добавлено в кодовую базу

### 1. File-backed индекс чанков

Новый модуль:
- `src/scireason/index/annoy_store.py`

Возможности:
- построение индекса Annoy для чанков
- сохранение sidecar-артефактов (`vectors.npy`, ids, metadata, manifest)
- fallback на NumPy retrieval, если `annoy` не установлен

### 2. PyTorch Geometric Temporal backend

Новый модуль:
- `src/scireason/tgnn/pygt_temporal_link_prediction.py`

Возможности:
- snapshot-based temporal link prediction
- использование `torch-geometric-temporal`, если библиотека установлена
- прозрачный fallback на уже существующий TGNN/TGN predictor

### 3. Multimodal triplet extraction

Новый модуль:
- `src/scireason/mm/multimodal_triplets.py`

Возможности:
- формирование триплетов не только из plain text
- объединение `text + tables + equations + VLM captions`
- единый multimodal analysis text для temporal triplet extraction

### 4. Основной orchestrator Task 3

Новые entrypoints:
- `src/scireason/pipeline/task3_hypothesis_generation.py`
- `src/scireason/task3_hypothesis_generation.py`

Функция:
- `prepare_task3_hypothesis_bundle(...)`

Поддерживаемые входы:
- `query`
- `trajectory YAML`
- список идентификаторов (`DOI/URL/PMID/PMCID/arXiv/OpenAlex`)
- `processed_papers/` для offline режима

Поддерживаемые выходы:
- chunk registry
- Annoy bundle
- temporal KG
- event stream
- multimodal triplets
- temporal link predictions
- ranked hypotheses
- manifest + markdown export

### 5. CLI

Добавлены команды:
- `top-papers-graph prepare-task3-hypotheses`
- `top-papers-graph task3-bundle`

### 6. Extras и документация

Добавлено:
- extra `.[task3]` в `pyproject.toml`
- `README_TASK3_HYPOTHESES.md`
- обновление основного `README.md`

### 7. Рабочий notebook Task 3

Добавлены notebook-файлы:
- `task3_multimodal_temporal_hypothesis_generation.ipynb`
- `notebooks/task3_multimodal_temporal_hypothesis_generation_colab.ipynb`

Возможности notebook:
- вход из `query`, `Task 1 YAML`, списка `identifier` и `commands`
- загрузка `processed_papers.zip` для более офлайн режима
- выбор routing для `g4f` и локальных HF/Transformers VLM
- генерация офлайн `A/B` формы для эксперта
- генерация `expert_hypothesis_artifacts_bundle.zip`

### 8. Тесты

Добавлены тесты:
- `tests/test_task3_annoy_store.py`
- `tests/test_task3_pipeline.py`

Проверяется:
- построение Annoy/fallback индекса
- регистрация CLI-команд Task 3
- offline Task 3 bundle на synthetic `processed_papers`

## Как усилена исходная схема пользователя

Изначальная схема была хорошей, но её стоило усилить в нескольких местах.

### Было
1. скачивание статей
2. парсинг текста/картинок/таблиц
3. эмбеддинги + Annoy
4. TGNN prediction
5. VL-триплеты
6. VL-анализ триплетов/чанков/эмбеддингов/TGNN
7. гипотезы + ранжирование

### Стало
1. **гибкий входной слой**: query / trajectory / identifiers / processed_dir
2. **robust acquisition**: PDF ingest + metadata-only fallback
3. **multimodal chunk registry** как единый контракт артефактов
4. **embeddings + Annoy sidecar + NumPy fallback**
5. **temporal KG + event stream** как центральное представление
6. **multimodal triplets** из text/page/table/formula/VLM captions
7. **temporal link prediction** через `torch-geometric-temporal` или fallback TGNN
8. **candidate-centric evidence assembly**:
   - temporal context
   - link prediction support
   - Annoy neighbors
   - matched multimodal triplets
   - candidate-specific VLM re-analysis
9. **ranked falsifiable hypotheses**
10. **bundle contract** для воспроизводимости и экспертной проверки

## Что было проверено

- `python -m py_compile` для новых модулей
- `python -m compileall -q src/scireason`
- `pytest -q tests/test_task3_annoy_store.py tests/test_task3_pipeline.py`
- CLI smoke run через `python -m scireason.cli task3-bundle ... --processed-dir ...`

## Ограничения и принятые решения

1. `torch-geometric-temporal` сделан **опциональным**, так как он не всегда есть в среде.
2. `annoy` тоже сделан **опциональным**, чтобы pipeline не ломался без дополнительных бинарных зависимостей.
3. Локальная VL-модель и `g4f` не дублируют новую реализацию, а используют уже существующий routing layer репозитория.
4. В Task 3 сохранён offline-first режим, чтобы модуль можно было тестировать без обязательного доступа к интернету и внешним сервисам.

## Ключевой результат

Task 3 теперь реализован как полноценный, воспроизводимый модуль репозитория, совместимый с уже существующими Task 1/Task 2 workflow и поддерживающий:

- локальные HF VL модели
- `g4f` модели
- мультимодальный parsing
- temporal link prediction
- время-осознанную генерацию и ранжирование проверяемых гипотез
