# Исправления после smoke-прогона DataSphere Jobs от 2026-06-15

## Что показал smoke-прогон

Smoke job `bt1150kq3cljko609750` стартовал через `hf-smoke-managed`, успешно прошёл ранний GPU preflight и подтвердил корректное окружение:

```text
torch_version: 2.5.1+cu121
torch_cuda_build: 12.1
cuda_available: true
device_count: 2
bf16_supported: true
device_name_0: NVIDIA A100-SXM4-80GB
```

Это означает, что предыдущее исправление PyTorch CUDA wheel с `cu121` сработало.

После этого dataset build дошёл до генерации локальных JSONL, но при SFT tokenization упал внутри `processing_class.apply_chat_template`:

```text
TypeError: argument of type 'int' is not iterable
```

Причина: Qwen3-VL chat template ожидает, что каждый элемент `message.content` будет строкой или словарём content-block формата. В export встречались сырые значения, например числа или произвольные JSON-объекты внутри `content`, и Jinja-шаблон падал на проверках вида `'image' in content`.

В логах также были массовые `HTTP Error 429` от Hugging Face Hub во время smoke dataset download. Smoke скачивал весь export asset tree, хотя для smoke нужны только небольшие SFT/GRPO подвыборки.

## Что изменено

### 1. Безопасная нормализация multimodal message content

Файлы:

```text
experiments/vlm_finetuning/scripts/train_vlm_sft.py
experiments/vlm_finetuning/scripts/train_vlm_grpo.py
```

Добавлена нормализация, которая перед TRL/Qwen3-VL tokenization приводит каждый `message.content` к безопасным блокам:

```python
{"type": "text", "text": "..."}
{"type": "image"}
```

Любые `int`, `float`, `bool`, `dict` неизвестной структуры и вложенные JSON-like значения превращаются в текстовый блок, а image references переносятся в top-level `images`.

### 2. Smoke dataset теперь скачивает только нужные assets

Файл:

```text
experiments/vlm_finetuning/scripts/build_hf_graph_experts_dataset.py
```

Для debug/smoke caps теперь используется двухфазная загрузка:

1. сначала скачиваются только metadata-файлы export (`sft.jsonl`, `grpo.jsonl`, README, summary);
2. затем детерминированно выбирается smoke subset;
3. затем скачиваются только assets, реально упомянутые в выбранных строках.

Это снижает риск Hugging Face `429 Rate limited` и ускоряет smoke run.

### 3. Smoke config получил dataset caps

Файл:

```text
experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_smoke_g2_2.yaml
```

Добавлено:

```yaml
MAX_SFT_SAMPLES: 96
MAX_GRPO_SAMPLES: 48
MAX_DATASET_SAMPLES: 0
```

Full config оставлен без caps:

```yaml
MAX_SFT_SAMPLES: 0
MAX_GRPO_SAMPLES: 0
MAX_DATASET_SAMPLES: 0
```

### 4. Wrapper прокидывает sample caps в dataset builder

Файл:

```text
experiments/vlm_finetuning/datasphere/bin/run_hf_top_papers_sft_grpo_full.sh
```

Добавлена передача:

```bash
--max-samples
--max-sft-samples
--max-grpo-samples
```

только если соответствующие env vars больше нуля.

## Локальные проверки

Выполнены локально:

```text
[OK] py_compile для build_hf_graph_experts_dataset.py, train_vlm_sft.py, train_vlm_grpo.py, train_vlm_dpo.py, run_full_pipeline.py, check_gpu_before_pipeline.py
[OK] bash -n для launch_examples.sh и datasphere/bin/*.sh
[OK] все строки requirements.txt проходят packaging.Requirement
[OK] все DataSphere YAML configs загружаются через pyyaml
[OK] все configs с requirements-file имеют env.python.pip.extra-index-urls=https://download.pytorch.org/whl/cu121
[OK] parse_job_id корректно отличает DataSphere job id от /tmp/datasphere/job_...
[OK] pytest: tests/test_vlm_training_format_normalization.py, tests/test_hf_export_smoke_sampling.py, tests/test_datasphere_job_configs.py
[OK] dry-run для hf-smoke-managed строит корректную команду datasphere project job execute
```

Удалённый DataSphere smoke run не выполнялся в этой среде, потому что здесь нет доступа к вашему проекту, GPU quota и секретам.

## Следующая команда для проверки

```bash
cd top-papers-graph-main
source .venv/bin/activate
export DATASPHERE_PROJECT_ID='<project_id_from_datasphere_ui>'

datasphere project get --id "$DATASPHERE_PROJECT_ID"
bash experiments/vlm_finetuning/datasphere/launch_examples.sh hf-smoke-managed
```

В новом smoke run ожидается:

1. `[gpu-check] OK`;
2. dataset build с `sample_limited_asset_download: true`;
3. SFT tokenization без `TypeError: argument of type 'int' is not iterable`;
4. создание `outputs/hf_top_papers_qwen3vl_8b_smoke_sft_lora.tar.gz`;
5. запуск GRPO smoke и создание `outputs/hf_top_papers_qwen3vl_8b_smoke_grpo_lora.tar.gz`.
