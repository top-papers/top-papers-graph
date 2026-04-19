# scidatapipe bridge for top-papers-graph

Новый модуль добавляет в `top-papers-graph` экспорт в совместимый со `scidatapipe` формат датасетов.

## Что делает

Команда:

```bash
top-papers-graph export-scidatapipe \
  --input-dir path/to/mixed_input_dir \
  --processed-papers-dir path/to/processed_papers \
  --out-dir data/derived/scidatapipe_export
```

или с авто-скачиванием статей и автозагрузкой результата в Hugging Face Hub:

```bash
top-papers-graph export-scidatapipe \
  --input-dir path/to/mixed_input_dir \
  --download-papers \
  --download-unpaywall-email you@example.com \
  --hf-upload \
  --hf-repo-id your-org/your-dataset \
  --out-dir data/derived/scidatapipe_export
```

Результат:

- `normalized_task1/<submission>/<submission>.yaml`
- `normalized_task2/<submission>/gold.json`
- `normalized_task2/<submission>/auto.json`
- `sft.jsonl`
- `grpo.jsonl`
- `export_summary.json`
- при `--download-papers`: `downloads/{pdfs,html,meta}` и `downloaded_processed_papers/`
- при `--hf-upload`: загрузка папки `out_dir` в HF dataset repo

## Что перенесено из scidatapipe

- нормализация Task 1 (`artifact_version 2/3 -> 4`)
- нормализация Task 2 bundle -> `gold.json` / `auto.json`
- pydantic-схемы `SFTSample` / `GRPOSample`
- форматы chat/content/image блоков
- логика загрузки статей по DOI/arXiv/wiki/url с fallbacks по open-access источникам

## Что улучшено относительно базового scidatapipe

### 1. Единый mixed input режим

Bridge умеет принимать не только отдельные `--task1-dir` и `--task2-dir`, но и один или несколько `--input-dir`.

Внутри таких директорий он автоматически находит:

- все `*.yaml` / `*.yml` как Task 1
- все Task 2 bundle-папки по маркерам (`edge_reviews.json`, `review_templates`, `review_state_latest.json`, `temporal_corrections.json`)
- все `*.zip` как Task 2 bundle archives

Если найден bundle directory, bridge не пытается дополнительно спускаться внутрь неё и повторно индексировать вложенные файлы.

### 2. Пакетная обработка директорий

Поддержаны все три режима:

- `--task1-dir` — только YAML
- `--task2-dir` — только bundle zip/папки
- `--input-dir` — смешанный пакетный режим

Поиск по умолчанию рекурсивный. Его можно отключить через `--recursive false`.

### 3. Скачивание статей прямо из полей разметки

Если включить `--download-papers`, bridge:

- собирает DOI / URL / arXiv / wiki / PMCID / OpenAlex из Task 1 YAML и Task 2 bundle
- скачивает PDF или HTML в `downloads/`
- при наличии PDF прогоняет его через `top-papers-graph` ingest (`parse-mm`-совместимая логика)
- складывает новые paper dirs в `downloaded_processed_papers/`

Таким образом можно запускать экспорт даже если `processed_papers` ещё не были собраны заранее.

### 4. Мультимодальные данные берутся не только из ручных `image_path`

Bridge сканирует:

- `processed_papers/*/mm/pages.jsonl`
- `processed_papers/*/mm/images/page_XXX.png`
- локальные `edge_reviews.json` bundle-артефакты
- локальные HTML-страницы (`html/*.html`) как fallback для wiki/HTML источников
- автоматически скачанные `downloads/html/*.html`

### 5. В prompt попадает **весь multimodal context** по процитированным статьям

Даже если к sample не прикреплены все изображения, в user message добавляется текстовый блок:

- page text
- `vlm_caption`
- `tables_md`
- `equations_md`
- locator / page / paper id

Это позволяет не терять мультимодальную информацию при экспорте датасета.

### 6. Автозагрузка результата в Hugging Face Hub

Если включить `--hf-upload`, bridge после сборки:

- при необходимости создаёт dataset repo
- при необходимости генерирует базовый `README.md`
- загружает весь `out_dir` через `huggingface_hub` как dataset repo upload

Это удобно для versioning и совместного доступа к `sft.jsonl`, `grpo.jsonl`, summary и multimodal assets.

## Полезные опции

- `--task1`, `--task2` — передать конкретные YAML / bundle zip / bundle dir
- `--task1-dir`, `--task2-dir` — пакетный режим по специализированным директориям
- `--input-dir` — смешанный пакетный режим
- `--download-papers` — включить скачивание статей по идентификаторам
- `--download-unpaywall-email` — email для Unpaywall DOI lookup
- `--download-root` — куда складывать PDF/HTML кеш
- `--download-processed-papers-dir` — куда складывать `processed_papers` для скачанных PDF
- `--ingest-downloaded-papers false` — только скачать, без parse-mm ingest
- `--download-multimodal false` — использовать обычный ingest вместо multimodal
- `--download-run-vlm false` — отключить VLM для скачанных PDF
- `--prefer-cached-downloads false` — форсировать повторные загрузки
- `--hf-upload` — загрузить результат в Hugging Face Hub
- `--hf-repo-id namespace/name` — целевой dataset repo
- `--hf-token` — токен доступа, если не хочется полагаться на `huggingface-cli login`
- `--hf-private` / `--hf-public` — создать repo как private/public
- `--hf-path-in-repo` — путь внутри dataset repo
- `--hf-commit-message`, `--hf-commit-description` — описание коммита
- `--hf-create-repo-if-missing false` — не пытаться создавать repo
- `--hf-generate-readme false` — не создавать dataset card

## Рекомендованный режим использования

Если нужен максимально богатый multimodal export, есть три режима:

1. **Предварительно** прогнать статьи через основной `top-papers-graph` pipeline / `parse-mm` и передать готовый `--processed-papers-dir`.
2. Или включить `--download-papers`, чтобы bridge сам скачал статьи по идентификаторам из разметки и построил дополнительный `downloaded_processed_papers` перед экспортом.
3. Если нужен versioned artefact-sharing, сразу включить `--hf-upload`, чтобы export folder после сборки ушёл в HF dataset repo.

Во втором и третьем режиме экспорт становится самодостаточным для больших смешанных папок с YAML и bundle-архивами.


## Import robustness in notebooks

The repository now includes a small root-level ``scireason`` compatibility shim so that
notebooks and ad-hoc scripts executed directly from a source checkout can import
``scireason.scidatapipe_bridge`` even before an editable install finishes. The canonical
installation path is still ``pip install -e .``.

