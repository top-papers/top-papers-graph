# Tutorial: smoke/full запуск Qwen3-VL-8B в Yandex DataSphere после исправления `requirements-file`

Дата ревизии: 2026-06-15.

Этот tutorial относится к исправленной версии репозитория `top-papers-graph-main` после ошибки:

```text
packaging.requirements.InvalidRequirement: Expected package name at the start of dependency specifier
    # Pin CUDA wheels to avoid installing a PyTorch build newer than the DataSphere NVIDIA driver.
    ^
```

Причина: локальный `datasphere==0.10.0` перед созданием job валидирует строки из `env.python.requirements-file` как Python package requirements. Поэтому комментарии и pip-опции внутри `experiments/vlm_finetuning/datasphere/requirements.txt` могут падать ещё до загрузки job в DataSphere.

Исправление в этом архиве:

1. `requirements.txt` теперь содержит **только валидные package specifiers**, без комментариев, пустых строк и pip flags.
2. PyTorch CUDA wheel index перенесён из `requirements.txt` в поддерживаемую секцию YAML:

```yaml
env:
  python:
    pip:
      extra-index-urls:
        - https://download.pytorch.org/whl/cu121
```

3. PyTorch по-прежнему зафиксирован на CUDA 12.1 wheels:

```text
torch==2.5.1+cu121
torchvision==0.20.1+cu121
torchaudio==2.5.1+cu121
```

Официальная документация DataSphere показывает, что для manual Python environment можно указывать `requirements-file` и параметры pip, включая `extra-index-urls`, в секции `env.python.pip`. Документация DataSphere CLI показывает запуск job командой `datasphere project job execute -p <идентификатор_проекта> -c <файл_конфигурации>`. PyTorch публикует отдельные CUDA wheels, включая `cu121`, через свой package index.

---

## 1. Что изменено в репозитории

### 1.1. `experiments/vlm_finetuning/datasphere/requirements.txt`

Теперь файл выглядит так:

```text
pyyaml>=6.0
pillow>=10.0
numpy>=1.26,<2
datasets>=2.20.0
huggingface_hub>=0.30.0
hf-xet>=1.1.0
accelerate>=1.8.0
transformers>=4.57.0
trl>=1.4.0,<2
peft>=0.17.0
bitsandbytes>=0.48.1
sentencepiece>=0.2.0
qwen-vl-utils>=0.0.14
evaluate>=0.4.2
safetensors>=0.4.3
wandb>=0.18.0
torch==2.5.1+cu121
torchvision==0.20.1+cu121
torchaudio==2.5.1+cu121
```

Важно: здесь **нет** строк вида:

```text
# comment
--extra-index-url https://download.pytorch.org/whl/cu121
```

Именно это исправляет ошибку `InvalidRequirement` на локальной стадии `datasphere project job execute`.

### 1.2. Все DataSphere YAML configs с этим requirements-file

Во все job configs, которые используют:

```yaml
requirements-file: experiments/vlm_finetuning/datasphere/requirements.txt
```

добавлена секция:

```yaml
pip:
  extra-index-urls:
    - https://download.pytorch.org/whl/cu121
```

Проверяемые файлы:

```text
experiments/vlm_finetuning/datasphere/job_configs/build_datasets.yaml
experiments/vlm_finetuning/datasphere/job_configs/dpo_pilot.yaml
experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_full_g2_2.yaml
experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_smoke_g2_2.yaml
experiments/vlm_finetuning/datasphere/job_configs/sft_pilot.yaml
experiments/vlm_finetuning/datasphere/job_configs/sft_smoke.yaml
experiments/vlm_finetuning/datasphere/job_configs/student_distill_qwen3vl_4b.yaml
experiments/vlm_finetuning/datasphere/job_configs/teacher_sft_qwen3vl_30b_a3b.yaml
experiments/vlm_finetuning/datasphere/job_configs/validate_extraction.yaml
```

### 1.3. Ранняя GPU/CUDA/BF16 проверка

Файл:

```text
experiments/vlm_finetuning/datasphere/bin/check_gpu_before_pipeline.py
```

Он запускается в начале full/smoke wrapper и пишет:

```text
reports/<OUT_PREFIX>_datasphere/gpu_preflight_status.json
```

Для smoke run это будет:

```text
reports/hf_top_papers_qwen3vl_8b_smoke_datasphere/gpu_preflight_status.json
```

Проверяются:

```text
nvidia-smi
torch.__version__
torch.version.cuda
torch.cuda.is_available()
torch.cuda.device_count()
torch.cuda.get_device_name(0)
torch.cuda.is_bf16_supported()
```

Если CUDA/BF16 недоступны, job теперь падает сразу, до скачивания датасета.

### 1.4. Managed launcher

Файл:

```text
experiments/vlm_finetuning/datasphere/run_full_pipeline.py
```

Исправления:

- parser больше не принимает `/tmp/datasphere/job_2026-...` за DataSphere job id;
- `cancel` вызывается только при `KeyboardInterrupt`, а не после already-failed blocking execute;
- если настоящий `job_id` найден, launcher всё равно пытается выставить TTL и скачать доступные outputs.

### 1.5. Smoke команда

В `launch_examples.sh` есть команда:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh hf-smoke-managed
```

Она запускает:

```text
experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_smoke_g2_2.yaml
```

Smoke config использует:

```yaml
OUT_PREFIX: hf_top_papers_qwen3vl_8b_smoke
HF_UPLOAD_AFTER_TRAINING: "0"
MAX_SFT_STEPS: 20
MAX_GRPO_STEPS: 5
GRPO_NUM_GENERATIONS: 1
GRPO_MAX_COMPLETION_LENGTH: 32
```

---

## 2. Подготовка локального окружения

Перейдите в распакованный репозиторий:

```bash
cd top-papers-graph-main
```

Создайте virtualenv:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U datasphere pyyaml packaging
```

Проверьте CLI:

```bash
datasphere version
yc version
```

Авторизуйтесь в Yandex Cloud:

```bash
yc init
```

Проверьте IAM token:

```bash
yc iam create-token >/dev/null && echo OK
```

---

## 3. Настройка DataSphere project id без `community_id`

Скопируйте project id из DataSphere UI.

Локально выполните:

```bash
export DATASPHERE_PROJECT_ID='<project_id_from_datasphere_ui>'
```

Проверьте доступ:

```bash
datasphere project get --id "$DATASPHERE_PROJECT_ID"
```

Не используйте старый путь:

```bash
datasphere project list -c <community_id>
```

В этом tutorial все job запускаются напрямую через `DATASPHERE_PROJECT_ID`.

---

## 4. Проверка Hugging Face token для full run

Для smoke run upload выключен. Для full run создайте в DataSphere project secret:

```text
Name: HFTOKEN
Value: hf_ваш_huggingface_write_token
```

Wrapper внутри job делает bridge:

```bash
if [ -z "${HF_TOKEN:-}" ] && [ -n "${HFTOKEN:-}" ]; then
  export HF_TOKEN="$HFTOKEN"
fi
if [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ] && [ -n "${HF_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi
```

То есть Hugging Face libraries увидят стандартную переменную `HF_TOKEN`.

---

## 5. Локальные проверки перед запуском

### 5.1. Проверка, что `requirements.txt` совместим с parser DataSphere CLI

```bash
python - <<'PY'
from pathlib import Path
from packaging.requirements import Requirement

path = Path('experiments/vlm_finetuning/datasphere/requirements.txt')
for lineno, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
    if not line.strip():
        raise SystemExit(f'blank line at {lineno}')
    Requirement(line)
print('[OK] requirements.txt contains only valid package specifiers')
PY
```

Ожидаемый результат:

```text
[OK] requirements.txt contains only valid package specifiers
```

### 5.2. Python compile checks

```bash
python -m py_compile \
  experiments/vlm_finetuning/datasphere/run_full_pipeline.py \
  experiments/vlm_finetuning/datasphere/bin/check_gpu_before_pipeline.py \
  experiments/vlm_finetuning/scripts/build_hf_graph_experts_dataset.py \
  experiments/vlm_finetuning/scripts/train_vlm_sft.py \
  experiments/vlm_finetuning/scripts/train_vlm_grpo.py \
  experiments/vlm_finetuning/scripts/train_vlm_dpo.py
```

### 5.3. Bash syntax checks

```bash
bash -n experiments/vlm_finetuning/datasphere/launch_examples.sh
for f in experiments/vlm_finetuning/datasphere/bin/*.sh; do bash -n "$f"; done
```

### 5.4. YAML checks

```bash
python - <<'PY'
from pathlib import Path
import yaml

for path in sorted(Path('experiments/vlm_finetuning/datasphere/job_configs').glob('*.yaml')):
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    py = cfg.get('env', {}).get('python', {})
    assert not ('root-path' in py and 'local-paths' in py), path
    storage = cfg.get('working-storage') or {}
    assert storage.get('type') == 'SSD', path
    assert str(storage.get('size', '')).endswith('Gb'), path
    if py.get('requirements-file') == 'experiments/vlm_finetuning/datasphere/requirements.txt':
        extra = py.get('pip', {}).get('extra-index-urls', [])
        assert 'https://download.pytorch.org/whl/cu121' in extra, path
    print('[OK]', path)
PY
```

### 5.5. Проверка parser job id

```bash
python - <<'PY'
from experiments.vlm_finetuning.datasphere.run_full_pipeline import parse_job_id

samples = {
    '2026-06-14 19:34:41,409 - [INFO] - created job `bt1rrni5s0e0ug22jail`': 'bt1rrni5s0e0ug22jail',
    'https://datasphere.yandex.cloud/communities/x/projects/y/job/bt1rrni5s0e0ug22jail': 'bt1rrni5s0e0ug22jail',
    'logs file path: /tmp/datasphere/job_2026-06-14T19:34:37.865135': None,
    'job id: bt1rrni5s0e0ug22jail': 'bt1rrni5s0e0ug22jail',
}

for text, expected in samples.items():
    actual = parse_job_id(text)
    print(text, '=>', actual)
    assert actual == expected, (actual, expected)
PY
```

---

## 6. Dry-run smoke managed launcher

```bash
python experiments/vlm_finetuning/datasphere/run_full_pipeline.py \
  --project-id "$DATASPHERE_PROJECT_ID" \
  --config experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_smoke_g2_2.yaml \
  --dry-run
```

Ожидаемый результат: JSON manifest с командой:

```text
datasphere project job execute -p <project_id> -c experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_smoke_g2_2.yaml
```

Job при `--dry-run` не запускается.

---

## 7. Запуск smoke run

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh hf-smoke-managed
```

В отличие от предыдущего сломанного варианта, команда больше не должна падать локально на строке `# Pin CUDA...`, потому что комментариев и pip flags в `requirements.txt` больше нет.

В начале remote job ищите блок:

```text
[gpu-check] nvidia-smi output:
[gpu-check] torch/CUDA status:
```

Ожидаемый успешный GPU preflight примерно такой:

```json
{
  "torch_version": "2.5.1+cu121",
  "torch_cuda_build": "12.1",
  "cuda_available": true,
  "device_count": 1,
  "bf16_supported": true,
  "device_name_0": "NVIDIA A100 ...",
  "ok": true
}
```

Если `cuda_available: false`, job завершится рано и сохранит диагностику в:

```text
reports/hf_top_papers_qwen3vl_8b_smoke_datasphere/gpu_preflight_status.json
```

---

## 8. Мониторинг и скачивание smoke outputs

Список jobs:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh list
```

Подключиться к логам:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh attach <job_id>
```

Скачать outputs вручную:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh download <job_id>
```

Проверить локальные artifacts:

```bash
ls -lh outputs/*smoke*tar.gz
ls -lh reports/*smoke*tar.gz
python -m json.tool reports/hf_top_papers_qwen3vl_8b_smoke_datasphere/gpu_preflight_status.json | head -120
python -m json.tool reports/hf_top_papers_qwen3vl_8b_smoke_datasphere/final_summary.json | head -120
```

---

## 9. Full run после успешного smoke

Запускайте full run только после успешного smoke.

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh hf-full-managed
```

Full config:

```text
experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_full_g2_2.yaml
```

Ожидаемые full artifacts:

```text
outputs/hf_top_papers_qwen3vl_8b_sft_lora.tar.gz
outputs/hf_top_papers_qwen3vl_8b_grpo_lora.tar.gz
reports/hf_top_papers_qwen3vl_8b_datasphere_reports.tar.gz
reports/hf_top_papers_qwen3vl_8b_datasphere/gpu_preflight_status.json
reports/hf_top_papers_qwen3vl_8b_datasphere/final_summary.json
reports/hf_top_papers_qwen3vl_8b_datasphere/hf_upload_summary.json
```

---

## 10. Если ошибка повторится

### 10.1. Снова `InvalidRequirement`

Проверьте, что в `requirements.txt` нет комментариев, пустых строк и pip flags:

```bash
nl -ba experiments/vlm_finetuning/datasphere/requirements.txt
```

И выполните:

```bash
python - <<'PY'
from pathlib import Path
from packaging.requirements import Requirement
path = Path('experiments/vlm_finetuning/datasphere/requirements.txt')
for i, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
    print(i, line)
    Requirement(line)
PY
```

### 10.2. `No matching distribution found for torch==2.5.1+cu121`

Проверьте, что в smoke/full YAML есть:

```yaml
pip:
  extra-index-urls:
    - https://download.pytorch.org/whl/cu121
```

Команда:

```bash
grep -R "extra-index-urls\|download.pytorch.org/whl/cu121" experiments/vlm_finetuning/datasphere/job_configs
```

### 10.3. CUDA всё ещё недоступна

Откройте:

```text
reports/<OUT_PREFIX>_datasphere/gpu_preflight_status.json
```

Если там:

```json
"cuda_available": false
```

то проблема уже не в `requirements.txt` parser. Проверьте:

```text
- действительно ли job запущен на g2.2;
- доступна ли GPU quota в проекте;
- что DataSphere выдал GPU VM;
- system.log / gpu_stats.tsv / docker_stats.tsv в job logs.
```

### 10.4. BF16 недоступен

Для Qwen3-VL-8B recipe ожидается BF16-capable GPU, например A100/H100-class. Не включайте CPU training для этой модели: это практически непригодно для полного обучения.

---

## 11. Короткая команда полного smoke-проверочного запуска

```bash
cd top-papers-graph-main
source .venv/bin/activate
export DATASPHERE_PROJECT_ID='<project_id_from_datasphere_ui>'

datasphere project get --id "$DATASPHERE_PROJECT_ID"

python - <<'PY'
from pathlib import Path
from packaging.requirements import Requirement
for line in Path('experiments/vlm_finetuning/datasphere/requirements.txt').read_text().splitlines():
    Requirement(line)
print('[OK] requirements parser')
PY

python experiments/vlm_finetuning/datasphere/run_full_pipeline.py \
  --project-id "$DATASPHERE_PROJECT_ID" \
  --config experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_smoke_g2_2.yaml \
  --dry-run

bash experiments/vlm_finetuning/datasphere/launch_examples.sh hf-smoke-managed
```

---

## 12. Что было проверено при сборке этого архива

Локально выполнены проверки:

```text
[OK] every requirements.txt line is accepted by packaging.Requirement
[OK] py_compile для run_full_pipeline.py, check_gpu_before_pipeline.py, build_hf_graph_experts_dataset.py, train_vlm_sft.py, train_vlm_grpo.py, train_vlm_dpo.py
[OK] bash -n для launch_examples.sh и datasphere/bin/*.sh
[OK] YAML configs загружаются через pyyaml
[OK] все configs с requirements-file имеют env.python.pip.extra-index-urls=https://download.pytorch.org/whl/cu121
[OK] parse_job_id больше не принимает /tmp/datasphere/job_2026-... как job id
[OK] smoke dry-run строит правильную DataSphere CLI команду
```

Полный удалённый DataSphere GPU run при сборке архива не выполнялся, потому что для этого нужны ваши project access, GPU quota и secrets.
