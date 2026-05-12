# Task 3 → Hugging Face VLM benchmark dataset

Этот патч добавляет отдельный пайплайн для выгрузки экспертных Task 3 case manifests из Google Sheets в Hugging Face dataset repository `top-papers/top-papers-graph-benchmark`.

## Что было проанализировано

1. `expert_task3_yaml_bundle_for_ab_test_fixed.ipynb` — notebook для эксперта-создателя набора. Он создаёт минимальный Task 1 YAML bundle, офлайн-форму и JSON-манифест `task3_ab_case_manifest_v1`.
2. Три заполненных экспертных манифеста:
   - Екатерина Кубракова: 12/12 заполненных cases;
   - Дмитрий Высоких: 8/12 заполненных cases, 4 пустых CASE-009..CASE-012 пропускаются по умолчанию;
   - Артем Васин: 12/12 заполненных cases.
3. `top_papers_graph_scidatapipe_hf_colab_from_csv_only_fixed_gdown_scope.ipynb` — существующий Colab uploader для Task 1/Task 2. Новый notebook повторяет его схему: скачать CSV из Google Sheet, выбрать последние отправки по `Timestamp`, скачать файлы из URL-колонок, собрать dataset folder, загрузить folder в Hugging Face Hub.

## Главное отличие Task 3 от Task 1/Task 2

Task 1/Task 2 uploader нормализует уже готовые экспертные артефакты в `sft.jsonl` / `grpo.jsonl`.

Task 3 creator notebook создаёт не training rows, а **case manifest для A/B проверки**. Поля `creator_prompt` часто сформулированы для эксперта: «Сравните, какой вариант лучше ...». Поэтому новый builder превращает эти формулировки в **прямую задачу для одной VL-модели**:

- `Сравните, какой вариант лучше извлекает X` → `Извлеките X`;
- `Сравните, как каждый VLM определяет X` → `Определите X`;
- уже прямые prompts (`Найти`, `Выявить`, `По рис. ...`) сохраняются.

`creator_rationale`, `expected_error_modes` и `match.*` не попадают в model prompt, чтобы не подсказывать модели критерии A/B проверки. Эти поля сохраняются отдельно в `review_metadata/task3_case_rationales.jsonl`.

## Добавленные файлы

- `src/scireason/task3_hf_benchmark.py` — библиотечный builder dataset folder.
- `scripts/data/build_task3_hf_benchmark_dataset.py` — CLI wrapper.
- `notebooks/top_papers_graph_task3_hf_benchmark_colab.ipynb` — Google Colab notebook.
- `tests/test_task3_hf_benchmark.py` — smoke tests.
- `examples/task3_ab_case_manifests/*.json` — три предоставленных образца экспертных манифестов.

## Выходной формат dataset repository

```text
<dataset_root>/
  README.md
  .gitattributes
  data/
    task3_vlm_generation.jsonl
    task3_cases_flat.jsonl
    task3_cases_summary.csv
  metadata/
    build_summary.json
  review_metadata/
    task3_case_rationales.jsonl
  source_manifests/
    *.json
  assets/
    images/
      <submission>/<case>/*.png
```

Основной файл: `data/task3_vlm_generation.jsonl`.

Каждая строка содержит:

- `messages`: system/user prompt с `{"type": "image"}` placeholders;
- `images`: относительные пути к rendered pages / explicit images;
- `model_task_prompt`: адаптированный prompt для одной VL-модели;
- `input_text`: полный user prompt;
- `generation_target_schema`: ожидаемая JSON-структура ответа;
- `review_metadata`: diagnostic metadata, не предназначенная для подачи модели.

## Локальная сборка на примерах

```bash
PYTHONPATH=src python scripts/data/build_task3_hf_benchmark_dataset.py \
  --input examples/task3_ab_case_manifests \
  --out-dir /tmp/top_papers_graph_benchmark_task3 \
  --dataset-repo-id top-papers/top-papers-graph-benchmark \
  --no-render-pdf-pages
```

На трёх приложенных манифестах smoke build даёт:

- manifests: 3;
- total cases: 36;
- written cases: 32;
- skipped incomplete: 4;
- written cases with images: 0, потому что в sample manifests не было PDF/images.

Если в Google Sheet или ZIP-архиве есть PDF/images, builder прикрепляет их к cases. Если включён `--download-papers-from-ids`, arXiv PDF скачиваются напрямую, DOI PDF пробуются через Unpaywall при наличии `--unpaywall-email`.

## Colab workflow

Откройте `notebooks/top_papers_graph_task3_hf_benchmark_colab.ipynb` и заполните:

- `TASK3_SHEET_URL` — Google Sheet с отправками Task 3;
- при необходимости `TASK3_URL_COLUMNS` — имена колонок с JSON/ZIP/PDF/images;
- `HF_REPO_ID = "top-papers/top-papers-graph-benchmark"`;
- `HF_TOKEN` или `notebook_login()`.

Если публичная ветка GitHub ещё не содержит этот патч, загрузите архив репозитория в Google Drive и укажите ссылку в `REPO_OVERLAY_GDRIVE_URL`.

## Загрузка после upload

```python
from datasets import load_dataset
from huggingface_hub import snapshot_download
from pathlib import Path

repo_id = "top-papers/top-papers-graph-benchmark"
ds = load_dataset(repo_id, "task3_vlm_generation", split="test")
repo_root = Path(snapshot_download(repo_id, repo_type="dataset"))
row = ds[0]
messages = row["messages"]
image_paths = [repo_root / rel for rel in row["images"]]
```

`messages` и `image_paths` можно передавать в runner выбранной VL-модели; затем сгенерированные ответы двух моделей отдавать эксперту в blind A/B интерфейс.
