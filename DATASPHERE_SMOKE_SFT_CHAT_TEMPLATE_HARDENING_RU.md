# Исправление smoke run: Qwen3-VL SFT chat template hardening

Дата: 2026-06-15

## Что показали новые логи

Smoke job `bt1150kq3cljko609750` успешно прошёл раннюю GPU-проверку: DataSphere выдал 2 x NVIDIA A100-SXM4-80GB, PyTorch `2.5.1+cu121`, CUDA build `12.1`, `cuda_available=true`, `bf16_supported=true`.

Новая ошибка возникает уже при подготовке SFT датасета внутри `TRL SFTTrainer`:

```text
TypeError: argument of type 'int' is not iterable
...
processing_class.apply_chat_template(...)
...
File "<template>", line 50, in top-level template code
```

Это означает, что в Qwen3-VL Jinja chat template попал несанитизированный элемент `message.content`, например число, а template выполняет проверки вида:

```jinja2
content.type == 'image' or 'image' in content or 'image_url' in content
```

Для `int` выражение `'image' in content` падает.

## Что исправлено

### 1. Усилена SFT-нормализация перед `SFTTrainer`

Файл:

```text
experiments/vlm_finetuning/scripts/train_vlm_sft.py
```

Добавлены функции:

- `_strict_qwen_safe_messages(...)`
- `_sanitize_sft_row_for_trl(...)`
- `_keep_only_trl_sft_columns(...)`
- `_validate_sft_dataset_for_qwen(...)`

Теперь перед созданием `SFTTrainer` датасет принудительно приводится к узкой схеме:

```text
messages
images / image
```

Все посторонние колонки вроде `label`, `label_text`, `id`, `task_family` удаляются, чтобы TRL не выбрал альтернативный формат `prompt`/`completion` или не протащил сырые JSON/int поля в chat template.

Каждый `message.content` после sanitization содержит только:

```python
{"type": "image"}
{"type": "text", "text": "..."}
```

### 2. Smoke caps сделаны гарантированными

Файл:

```text
experiments/vlm_finetuning/datasphere/bin/run_hf_top_papers_sft_grpo_full.sh
```

Если `OUT_PREFIX` содержит `smoke`, wrapper теперь выставляет дефолтные ограничения даже если YAML/env vars не попали в remote job:

```bash
MAX_SFT_SAMPLES=96
MAX_GRPO_SAMPLES=48
MAX_DATASET_SAMPLES=0
```

Это нужно, потому что в новых логах dataset builder снова скачивал полный export asset tree и дошёл до ~15k files, а не до маленькой smoke-подвыборки.

### 3. Снижен параллелизм Hugging Face Hub download

Файл:

```text
experiments/vlm_finetuning/scripts/build_hf_graph_experts_dataset.py
```

Добавлен параметр:

```bash
--hf-download-max-workers
```

Он прокидывается в `snapshot_download(..., max_workers=...)`. Для smoke config задано:

```yaml
HF_DOWNLOAD_MAX_WORKERS: 1
```

Для full config:

```yaml
HF_DOWNLOAD_MAX_WORKERS: 2
```

Это снижает риск `HTTP Error 429` / `Rate limited`, которые видны в smoke logs.

### 4. Команда `hf-smoke-managed` стала более явной

Файл:

```text
experiments/vlm_finetuning/datasphere/launch_examples.sh
```

Для локального manifest/dry-run теперь явно экспортируются smoke defaults:

```bash
OUT_PREFIX=hf_top_papers_qwen3vl_8b_smoke
MAX_SFT_SAMPLES=96
MAX_GRPO_SAMPLES=48
MAX_DATASET_SAMPLES=0
```

## Как проверить

```bash
cd top-papers-graph-main
source .venv/bin/activate

export DATASPHERE_PROJECT_ID='bt18pnosk97i8n24ddnv'

datasphere project get --id "$DATASPHERE_PROJECT_ID"

bash experiments/vlm_finetuning/datasphere/launch_examples.sh hf-smoke-managed
```

В новых логах ожидается:

```text
[gpu-check] OK
"max_sft_samples": 96
"max_grpo_samples": 48
"hf_download_max_workers": 1
sample_limited_asset_download: true
[train_vlm_sft] validated ... train rows for Qwen3-VL chat template safety
Tokenizing train dataset ... без TypeError
```

## Локальные проверки

Выполнены:

```text
py_compile для training scripts и DataSphere launcher
bash -n для launch_examples.sh и datasphere/bin/*.sh
requirements.txt через packaging.Requirement
YAML load всех DataSphere job configs
pytest для:
  tests/test_vlm_training_format_normalization.py
  tests/test_hf_export_smoke_sampling.py
  tests/test_datasphere_job_configs.py
smoke dry-run через launch_examples.sh hf-smoke-managed --dry-run
```
