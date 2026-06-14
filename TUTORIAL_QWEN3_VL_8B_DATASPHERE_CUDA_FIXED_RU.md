# Tutorial: повторный smoke/full запуск Qwen3-VL-8B в Yandex DataSphere после CUDA/PyTorch fix

Дата ревизии: 2026-06-14.

Этот tutorial относится к исправленной версии репозитория `top-papers-graph-main`, в которую внесены правки по результатам smoke-логов:

- зафиксированы CUDA wheels PyTorch под `cu121`;
- добавлена ранняя GPU/CUDA/BF16-проверка до скачивания датасета;
- исправлен parser DataSphere `job_id`, чтобы локальный путь `/tmp/datasphere/job_2026-...` не принимался за настоящий id задания;
- managed launcher больше не вызывает `cancel` после уже завершившегося failed `datasphere project job execute`;
- добавлен отдельный smoke job config `hf_top_papers_sft_grpo_smoke_g2_2.yaml`;
- добавлена команда `hf-smoke-managed` в `launch_examples.sh`;
- добавлен bridge `HFTOKEN -> HF_TOKEN -> HUGGING_FACE_HUB_TOKEN` для Hugging Face upload.

Официальные документы для сверки:

- DataSphere Jobs CLI: https://yandex.cloud/ru/docs/datasphere/concepts/jobs/cli
- DataSphere запуск Jobs из проекта: https://yandex.cloud/ru/docs/datasphere/operations/projects/work-with-jobs
- PyTorch previous versions / CUDA wheels: https://pytorch.org/get-started/previous-versions/
- Hugging Face Hub environment variables: https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables

---

## 0. Что именно было исправлено

### 0.1. CUDA/PyTorch mismatch

В smoke-логе была ошибка вида:

```text
CUDA initialization: The NVIDIA driver on your system is too old (found version 12020)
ValueError: Your setup doesn't support bf16/gpu.
```

Это означает, что установленный в job PyTorch был собран под слишком новую CUDA runtime относительно NVIDIA driver в DataSphere image. В DataSphere вы обычно не обновляете host driver вручную, поэтому исправление сделано на уровне `requirements.txt`: теперь PyTorch, TorchVision и Torchaudio зафиксированы на CUDA 12.1 wheels.

Файл:

```text
experiments/vlm_finetuning/datasphere/requirements.txt
```

Ключевые строки:

```text
--extra-index-url https://download.pytorch.org/whl/cu121
torch==2.5.1+cu121
torchvision==0.20.1+cu121
torchaudio==2.5.1+cu121
```

### 0.2. Ранняя GPU-проверка

Добавлен файл:

```text
experiments/vlm_finetuning/datasphere/bin/check_gpu_before_pipeline.py
```

Он запускается до скачивания датасета и пишет:

```text
reports/<OUT_PREFIX>_datasphere/gpu_preflight_status.json
```

Проверяются:

- доступность `nvidia-smi`;
- `torch.__version__`;
- `torch.version.cuda`;
- `torch.cuda.is_available()`;
- `torch.cuda.device_count()`;
- имя GPU;
- `torch.cuda.is_bf16_supported()`.

Если CUDA или BF16 недоступны, job падает сразу, до скачивания тысяч файлов датасета.

### 0.3. Исправление managed launcher

Файл:

```text
experiments/vlm_finetuning/datasphere/run_full_pipeline.py
```

Теперь parser принимает только реальные DataSphere job ids, например:

```text
created job `bt1rrni5s0e0ug22jail`
/job/bt1rrni5s0e0ug22jail
job id: bt1rrni5s0e0ug22jail
```

и не принимает локальные пути:

```text
/tmp/datasphere/job_2026-06-14T19:34:37.865135
```

Также при failed `datasphere project job execute` launcher не делает `cancel`, потому что blocking execute уже завершился. Вместо этого он пытается выставить TTL и скачать доступные outputs, если настоящий `job_id` был распарсен.

---

## 1. Предварительные условия

На вашей локальной машине должны быть:

- Linux/macOS/WSL shell;
- Python 3.10+;
- Yandex Cloud CLI `yc`;
- DataSphere CLI `datasphere`;
- доступ к DataSphere project;
- project id, скопированный из DataSphere UI;
- DataSphere project secret `HFTOKEN`, если вы будете делать full upload в Hugging Face.

Проверка CLI:

```bash
yc version
datasphere version
```

Если `datasphere` не установлен:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U datasphere pyyaml
```

Аутентификация:

```bash
yc init
```

Проверка IAM token:

```bash
yc iam create-token >/dev/null && echo OK
```

---

## 2. Подготовьте Hugging Face token как DataSphere secret

Для smoke run upload отключён, но для full run нужен Hugging Face token с write-доступом к target model repo.

В DataSphere project создайте secret:

```text
Name: HFTOKEN
Value: hf_ваш_huggingface_write_token
```

Почему `HFTOKEN`, а не только `HF_TOKEN`: имя `HFTOKEN` безопасно проходит ограничения DataSphere secret names, а wrapper внутри job делает bridge:

```bash
if [ -z "${HF_TOKEN:-}" ] && [ -n "${HFTOKEN:-}" ]; then
  export HF_TOKEN="$HFTOKEN"
fi
if [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ] && [ -n "${HF_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi
```

То есть внутри Hugging Face libraries увидят стандартную переменную `HF_TOKEN`.

---

## 3. Перейдите в корень репозитория

После распаковки архива:

```bash
cd top-papers-graph-main
```

Проверьте, что вы видите нужные файлы:

```bash
ls -lh experiments/vlm_finetuning/datasphere/requirements.txt
ls -lh experiments/vlm_finetuning/datasphere/bin/check_gpu_before_pipeline.py
ls -lh experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_smoke_g2_2.yaml
```

---

## 4. Выставьте DataSphere project id

В DataSphere UI откройте нужный project и скопируйте project id.

Локально:

```bash
export DATASPHERE_PROJECT_ID='<project_id_from_datasphere_ui>'
```

Проверка:

```bash
datasphere project get --id "$DATASPHERE_PROJECT_ID"
```

Ожидаемый результат: CLI выводит информацию о проекте.

Не используйте:

```bash
datasphere project list -c <community_id>
```

В текущем сценарии запуск идёт напрямую по `DATASPHERE_PROJECT_ID`.

---

## 5. Локальные preflight-проверки репозитория

Эти команды не запускают обучение и не требуют GPU.

```bash
python -m py_compile \
  experiments/vlm_finetuning/datasphere/run_full_pipeline.py \
  experiments/vlm_finetuning/datasphere/bin/check_gpu_before_pipeline.py \
  experiments/vlm_finetuning/scripts/build_hf_graph_experts_dataset.py \
  experiments/vlm_finetuning/scripts/train_vlm_sft.py \
  experiments/vlm_finetuning/scripts/train_vlm_grpo.py \
  experiments/vlm_finetuning/scripts/train_vlm_dpo.py
```

Проверка shell wrappers:

```bash
bash -n experiments/vlm_finetuning/datasphere/launch_examples.sh
for f in experiments/vlm_finetuning/datasphere/bin/*.sh; do bash -n "$f"; done
```

Проверка YAML configs:

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
    print('[OK]', path)
PY
```

Проверка parser `job_id`:

```bash
python - <<'PY'
import importlib.util

spec = importlib.util.spec_from_file_location(
    'run_full_pipeline',
    'experiments/vlm_finetuning/datasphere/run_full_pipeline.py',
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

samples = {
    '2026-06-14 19:34:41,409 - [INFO] - created job `bt1rrni5s0e0ug22jail`': 'bt1rrni5s0e0ug22jail',
    'https://datasphere.yandex.cloud/x/job/bt1rrni5s0e0ug22jail': 'bt1rrni5s0e0ug22jail',
    'job id: bt1rrni5s0e0ug22jail': 'bt1rrni5s0e0ug22jail',
    'logs file path: /tmp/datasphere/job_2026-06-14T19:34:37.865135': None,
}

for text, expected in samples.items():
    actual = mod.parse_job_id(text)
    print(text, '=>', actual)
    assert actual == expected, (text, actual, expected)
PY
```

Ожидаемый результат: последний пример возвращает `None`, а не `job_2026-...`.

---

## 6. Dry-run smoke managed launcher

Dry-run показывает команду запуска, но не стартует DataSphere job:

```bash
python experiments/vlm_finetuning/datasphere/run_full_pipeline.py \
  --project-id "$DATASPHERE_PROJECT_ID" \
  --config experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_smoke_g2_2.yaml \
  --dry-run
```

Ожидаемый фрагмент:

```text
datasphere project job execute -p <project_id> -c experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_smoke_g2_2.yaml
```

---

## 7. Запустите smoke run

Рекомендуемая команда:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh hf-smoke-managed
```

Эквивалентно:

```bash
python experiments/vlm_finetuning/datasphere/run_full_pipeline.py \
  --project-id "$DATASPHERE_PROJECT_ID" \
  --config experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_smoke_g2_2.yaml
```

Smoke config использует:

```yaml
OUT_PREFIX: hf_top_papers_qwen3vl_8b_smoke
HF_UPLOAD_AFTER_TRAINING: "0"
MAX_SFT_STEPS: 20
MAX_GRPO_STEPS: 5
SFT_TIMEOUT_HOURS: 4
GRPO_TIMEOUT_HOURS: 4
GRPO_NUM_GENERATIONS: 1
GRPO_MAX_COMPLETION_LENGTH: 32
ENABLE_GPU_PREFLIGHT: "1"
```

---

## 8. Что должно произойти в начале smoke run

До скачивания датасета должен появиться блок:

```text
[gpu-check] nvidia-smi output:
...
[gpu-check] torch/CUDA status:
{
  "torch_version": "2.5.1+cu121",
  "torch_cuda_build": "12.1",
  "cuda_available": true,
  "device_count": 1 или 2,
  "bf16_supported": true,
  "device_name_0": "NVIDIA A100 ...",
  "ok": true
}
```

И файл:

```text
reports/hf_top_papers_qwen3vl_8b_smoke_datasphere/gpu_preflight_status.json
```

Если вместо этого будет:

```json
"cuda_available": false
```

значит проблема всё ещё на уровне DataSphere runtime/GPU/PyTorch wheel. В этом случае не надо ждать скачивания датасета: job завершится раньше и отдаст диагностический JSON.

---

## 9. Если GPU preflight снова падает

Сначала скачайте outputs failed job, если managed launcher сам не скачал их:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh list
bash experiments/vlm_finetuning/datasphere/launch_examples.sh download <job_id>
```

Посмотрите preflight JSON:

```bash
python -m json.tool reports/hf_top_papers_qwen3vl_8b_smoke_datasphere/gpu_preflight_status.json
```

Проверьте эти поля:

```text
torch_version
torch_cuda_build
cuda_available
device_count
bf16_supported
device_name_0
nvidia_smi.output_tail
```

Если `torch_version` не `2.5.1+cu121`, значит DataSphere поставил не те зависимости. Проверьте, что в job действительно загружен обновлённый файл:

```bash
grep -n "torch\|cu121" experiments/vlm_finetuning/datasphere/requirements.txt
```

Если `cuda_available=false`, но `nvidia-smi` показывает GPU, проблема почти наверняка в совместимости PyTorch wheel и driver.

Если `nvidia-smi` не найден или GPU не виден, проверьте:

```text
[ ] job config использует cloud-instance-types: g2.2
[ ] в DataSphere project доступна GPU quota
[ ] job действительно стартовал в GPU runtime
[ ] нет ограничений проекта/папки/организации на GPU
```

---

## 10. Проверка smoke outputs после успешного run

После успешного smoke run ожидаются файлы:

```bash
ls -lh outputs/hf_top_papers_qwen3vl_8b_smoke_sft_lora.tar.gz
ls -lh outputs/hf_top_papers_qwen3vl_8b_smoke_grpo_lora.tar.gz
ls -lh reports/hf_top_papers_qwen3vl_8b_smoke_datasphere_reports.tar.gz
python -m json.tool reports/hf_top_papers_qwen3vl_8b_smoke_datasphere/final_summary.json | head -120
```

Проверка adapter archives:

```bash
tar -tzf outputs/hf_top_papers_qwen3vl_8b_smoke_sft_lora.tar.gz | head -80
tar -tzf outputs/hf_top_papers_qwen3vl_8b_smoke_grpo_lora.tar.gz | head -80
```

Минимально ожидаемые файлы внутри adapter archive:

```text
adapter_config.json
adapter_model.safetensors
preprocessor_config.json или tokenizer/processor files
```

---

## 11. Full run после успешного smoke

Full run запускайте только после успешного smoke:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh hf-full-managed
```

Эквивалентно:

```bash
python experiments/vlm_finetuning/datasphere/run_full_pipeline.py \
  --project-id "$DATASPHERE_PROJECT_ID" \
  --config experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_full_g2_2.yaml
```

Full config использует:

```yaml
OUT_PREFIX: hf_top_papers_qwen3vl_8b
HF_UPLOAD_AFTER_TRAINING: "1"
MAX_SFT_STEPS: 480
MAX_GRPO_STEPS: 160
SFT_TIMEOUT_HOURS: 48
GRPO_TIMEOUT_HOURS: 35
ENABLE_GPU_PREFLIGHT: "1"
```

Для full run убедитесь, что DataSphere secret `HFTOKEN` создан и Hugging Face token имеет write-доступ к:

```text
top-papers/Qwen3-VL-8B-Instruct-scireason
```

---

## 12. Мониторинг и управление jobs

Список jobs:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh list
```

Информация по job:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh get <job_id>
```

Подключиться к логам:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh attach <job_id>
```

Остановить job:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh cancel <job_id>
```

Выставить TTL:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh ttl <job_id> 1
```

Скачать outputs:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh download <job_id>
```

---

## 13. Как временно отключить GPU preflight

Отключать preflight для обучения не рекомендуется. Но для диагностики можно временно поставить в YAML:

```yaml
- ENABLE_GPU_PREFLIGHT: "0"
```

или добавить в env vars:

```yaml
- SKIP_GPU_PREFLIGHT: "1"
```

Разница:

- `ENABLE_GPU_PREFLIGHT=0` вообще не запускает preflight script;
- `SKIP_GPU_PREFLIGHT=1` запускает script, пишет JSON, но не падает при ошибке.

Для нормального smoke/full run оставляйте:

```yaml
- ENABLE_GPU_PREFLIGHT: "1"
```

---

## 14. Что делать при повторе старой ошибки

Если снова появится:

```text
CUDA initialization: The NVIDIA driver on your system is too old
```

проверьте в первую очередь:

```bash
grep -n "torch\|torchvision\|torchaudio\|cu121" experiments/vlm_finetuning/datasphere/requirements.txt
```

Ожидаемо:

```text
--extra-index-url https://download.pytorch.org/whl/cu121
torch==2.5.1+cu121
torchvision==0.20.1+cu121
torchaudio==2.5.1+cu121
```

Если строки правильные, но DataSphere всё равно ставит другой torch, очистите/перезапустите job окружение или проверьте, не переопределяет ли зависимости другой requirements/bootstrap слой.

---

## 15. Контрольный чеклист

Перед smoke:

```text
[ ] DATASPHERE_PROJECT_ID выставлен
[ ] datasphere project get --id "$DATASPHERE_PROJECT_ID" проходит
[ ] requirements.txt содержит torch==2.5.1+cu121
[ ] check_gpu_before_pipeline.py существует
[ ] hf_top_papers_sft_grpo_smoke_g2_2.yaml существует
[ ] launch_examples.sh содержит hf-smoke-managed
[ ] py_compile проходит
[ ] bash -n проходит
[ ] run_full_pipeline.py --dry-run проходит
```

Перед full:

```text
[ ] smoke run прошёл успешно
[ ] gpu_preflight_status.json показывает cuda_available=true
[ ] gpu_preflight_status.json показывает bf16_supported=true
[ ] smoke SFT archive создан
[ ] smoke GRPO archive создан
[ ] DataSphere secret HFTOKEN создан
[ ] HF token имеет write-доступ к целевому repo
[ ] HF_UPLOAD_AFTER_TRAINING="1" только для full run
```

---

## 16. Команды одним блоком для повторного smoke

```bash
cd top-papers-graph-main

export DATASPHERE_PROJECT_ID='<project_id_from_datasphere_ui>'

datasphere project get --id "$DATASPHERE_PROJECT_ID"

python -m py_compile \
  experiments/vlm_finetuning/datasphere/run_full_pipeline.py \
  experiments/vlm_finetuning/datasphere/bin/check_gpu_before_pipeline.py \
  experiments/vlm_finetuning/scripts/build_hf_graph_experts_dataset.py \
  experiments/vlm_finetuning/scripts/train_vlm_sft.py \
  experiments/vlm_finetuning/scripts/train_vlm_grpo.py

bash -n experiments/vlm_finetuning/datasphere/launch_examples.sh
for f in experiments/vlm_finetuning/datasphere/bin/*.sh; do bash -n "$f"; done

python experiments/vlm_finetuning/datasphere/run_full_pipeline.py \
  --project-id "$DATASPHERE_PROJECT_ID" \
  --config experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_smoke_g2_2.yaml \
  --dry-run

bash experiments/vlm_finetuning/datasphere/launch_examples.sh hf-smoke-managed
```

---

## 17. Что было проверено при сборке этого архива

В этой среде были выполнены статические проверки:

```text
[OK] py_compile для run_full_pipeline.py, check_gpu_before_pipeline.py, build_hf_graph_experts_dataset.py, train_vlm_sft.py, train_vlm_grpo.py
[OK] bash -n для launch_examples.sh и datasphere/bin/*.sh
[OK] YAML configs загружаются через pyyaml
[OK] parser job id принимает created job `bt...` и не принимает /tmp/datasphere/job_2026-...
[OK] dry-run строит команду datasphere project job execute -p <project_id> -c hf_top_papers_sft_grpo_smoke_g2_2.yaml
```

Полный GPU/DataSphere run здесь не выполнялся: для него нужны ваш DataSphere project, GPU quota и ваши секреты.
