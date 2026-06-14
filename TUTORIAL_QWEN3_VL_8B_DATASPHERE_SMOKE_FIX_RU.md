# Tutorial: повторный smoke/full запуск Qwen3-VL-8B в Yandex DataSphere после исправления CUDA/PyTorch и managed launcher

Дата ревизии: 2026-06-14.

Этот tutorial относится к исправленному архиву репозитория `top-papers-graph-main` и заменяет проблемный smoke-прогон, где DataSphere job доходил до SFT stage, но падал на CUDA/BF16 из-за несовместимого PyTorch CUDA wheel и NVIDIA driver внутри DataSphere runtime.

## 0. Что было исправлено

В репозиторий внесены четыре практические правки.

### 0.1. PyTorch зафиксирован на CUDA 12.1 wheel

Файл:

```text
experiments/vlm_finetuning/datasphere/requirements.txt
```

Теперь в нём не используется плавающее `torch>=...`. Вместо этого добавлено:

```text
--extra-index-url https://download.pytorch.org/whl/cu121
torch==2.5.1+cu121
torchvision==0.20.1+cu121
torchaudio==2.5.1+cu121
```

Зачем: в smoke-логе DataSphere NVIDIA driver сообщал CUDA compatibility около `12020`, а pip мог поставить более новый PyTorch CUDA build. В DataSphere driver вы не обновляете вручную, поэтому безопаснее выбрать PyTorch wheel под более старую CUDA runtime.

### 0.2. Добавлен ранний GPU/BF16 preflight

Новый файл:

```text
experiments/vlm_finetuning/datasphere/bin/check_gpu_before_pipeline.py
```

Wrapper теперь вызывает его **до скачивания датасета**:

```bash
python experiments/vlm_finetuning/datasphere/bin/check_gpu_before_pipeline.py \
  --report-dir "$REPORT_DIR" \
  --require-bf16
```

Если PyTorch снова не видит CUDA или BF16, job упадёт сразу и запишет диагностику в:

```text
reports/<OUT_PREFIX>_datasphere/gpu_preflight_status.json
```

Это экономит время и деньги: раньше job сначала скачивал тысячи файлов датасета, и только потом падал на SFT.

### 0.3. Исправлен parser job id в managed launcher

Файл:

```text
experiments/vlm_finetuning/datasphere/run_full_pipeline.py
```

Теперь parser принимает только реальные DataSphere job id из строк вида:

```text
created job `bt1rrni5s0e0ug22jail`
/job/bt1rrni5s0e0ug22jail
job id: bt1rrni5s0e0ug22jail
```

и больше не принимает локальные пути вида:

```text
/tmp/datasphere/job_2026-06-14T19:34:37.865135
```

Также managed launcher больше не вызывает `cancel` после того, как `datasphere project job execute` уже завершился с non-zero кодом. Он сохраняет `job_id`, выставляет TTL и пытается скачать доступные outputs.

### 0.4. Python version в YAML упрощён до `3.10`

Во всех DataSphere job configs заменено:

```yaml
version: 3.10.13
```

на:

```yaml
version: "3.10"
```

Это убирает предупреждение DataSphere CLI:

```text
Python version will be reduced to major (3.10.13 -> 3.10)
```

## 1. Предварительные условия

Нужно иметь:

```text
[ ] Yandex Cloud account
[ ] DataSphere project id
[ ] доступ пользователя/группы к DataSphere project
[ ] DataSphere project secret HFTOKEN, если планируется Hugging Face upload
[ ] доступная GPU-конфигурация g2.2
[ ] локально установленный yc CLI или OAuth auth для DataSphere CLI
[ ] локально установленный Python 3.10-3.12
```

Проект теперь запускается без `community_id`. Основная переменная:

```bash
export DATASPHERE_PROJECT_ID='<project_id_from_datasphere_ui>'
```

## 2. Распакуйте исправленный архив

```bash
mkdir -p ~/Documents/top-papers-graph-fixed
cd ~/Documents/top-papers-graph-fixed
unzip /path/to/top-papers-graph-main-datasphere-smoke-fix.zip
cd top-papers-graph-main
```

Проверьте, что файлы исправлений на месте:

```bash
ls -lh experiments/vlm_finetuning/datasphere/bin/check_gpu_before_pipeline.py

grep -n "torch==2.5.1+cu121\|torchvision==0.20.1+cu121\|torchaudio==2.5.1+cu121" \
  experiments/vlm_finetuning/datasphere/requirements.txt

grep -n "check_gpu_before_pipeline" \
  experiments/vlm_finetuning/datasphere/bin/run_hf_top_papers_sft_grpo_full.sh
```

Ожидаемо:

```text
check_gpu_before_pipeline.py существует
torch==2.5.1+cu121 найден
runtime wrapper вызывает check_gpu_before_pipeline.py
```

## 3. Подготовьте локальное окружение

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U datasphere pyyaml
```

Проверьте CLI:

```bash
datasphere version
```

Если используете Yandex Cloud CLI profile:

```bash
yc init
```

Если используете OAuth token:

```bash
export YC_OAUTH_TOKEN='<yandex_oauth_token>'
```

## 4. Укажите DataSphere project id без community

Скопируйте project id из DataSphere UI и выполните:

```bash
export DATASPHERE_PROJECT_ID='<project_id_from_datasphere_ui>'

datasphere project get --id "$DATASPHERE_PROJECT_ID"
```

Ожидаемо: команда выводит информацию о проекте.

Не используйте старый путь:

```bash
datasphere project list -c <community_id>
```

Именно он раньше приводил к ошибке `Community ... was not found`.

## 5. Проверьте DataSphere secret `HFTOKEN`

Для smoke run upload обычно выключен, но для full run с Hugging Face upload нужен secret.

В DataSphere UI создайте project secret:

```text
Name: HFTOKEN
Value: hf_ваш_huggingface_write_token
```

В runtime wrapper уже добавлен bridge:

```bash
if [ -z "${HF_TOKEN:-}" ] && [ -n "${HFTOKEN:-}" ]; then
  export HF_TOKEN="$HFTOKEN"
fi
if [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ] && [ -n "${HF_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi
```

То есть DataSphere secret `HFTOKEN` автоматически становится переменной `HF_TOKEN`, которую использует Hugging Face Hub.

## 6. Локальные preflight-проверки

Эти команды не запускают обучение и не требуют GPU.

```bash
python -m py_compile \
  experiments/vlm_finetuning/datasphere/run_full_pipeline.py \
  experiments/vlm_finetuning/datasphere/bin/check_gpu_before_pipeline.py \
  experiments/vlm_finetuning/scripts/build_hf_graph_experts_dataset.py \
  experiments/vlm_finetuning/scripts/train_vlm_sft.py \
  experiments/vlm_finetuning/scripts/train_vlm_grpo.py \
  experiments/vlm_finetuning/scripts/train_vlm_dpo.py

bash -n experiments/vlm_finetuning/datasphere/bin/run_hf_top_papers_sft_grpo_full.sh
bash -n experiments/vlm_finetuning/datasphere/launch_examples.sh
```

Проверьте parser job id:

```bash
python - <<'PY'
from experiments.vlm_finetuning.datasphere.run_full_pipeline import parse_job_id

samples = [
    "2026-06-14 19:34:41,409 - [INFO] - created job `bt1rrni5s0e0ug22jail`",
    "https://datasphere.yandex.cloud/communities/x/projects/y/job/bt1rrni5s0e0ug22jail",
    "job id: bt1rrni5s0e0ug22jail",
    "logs file path: /tmp/datasphere/job_2026-06-14T19:34:37.865135",
]
for s in samples:
    print(s, "=>", parse_job_id(s))

assert parse_job_id(samples[0]) == "bt1rrni5s0e0ug22jail"
assert parse_job_id(samples[1]) == "bt1rrni5s0e0ug22jail"
assert parse_job_id(samples[2]) == "bt1rrni5s0e0ug22jail"
assert parse_job_id(samples[3]) is None
PY
```

Ожидаемый результат:

```text
created job ... => bt1rrni5s0e0ug22jail
/job/...        => bt1rrni5s0e0ug22jail
job id: ...     => bt1rrni5s0e0ug22jail
/tmp/...        => None
```

Проверьте YAML:

```bash
python - <<'PY'
from pathlib import Path
import yaml

for path in sorted(Path('experiments/vlm_finetuning/datasphere/job_configs').glob('*.yaml')):
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    py = cfg.get('env', {}).get('python', {})
    if py:
        assert str(py.get('version', '')).startswith('3.10'), (path, py.get('version'))
    storage = cfg.get('working-storage') or {}
    assert storage.get('type') == 'SSD', path
    assert str(storage.get('size', '')).endswith('Gb'), path
    print('[OK]', path)
PY
```

## 7. Dry-run managed launcher

```bash
python experiments/vlm_finetuning/datasphere/run_full_pipeline.py \
  --project-id "$DATASPHERE_PROJECT_ID" \
  --config experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_full_g2_2.yaml \
  --dry-run
```

Ожидаемо: будет напечатан JSON manifest с командой:

```text
datasphere project job execute -p <project_id> -c experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_full_g2_2.yaml
```

Job при `--dry-run` не запускается.

## 8. Smoke run

Для smoke run временно уменьшите шаги и выключите upload. Самый простой вариант — прямо отредактировать YAML:

```bash
python - <<'PY'
from pathlib import Path
import yaml

path = Path('experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_full_g2_2.yaml')
cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
vars_list = cfg['env']['vars']

def set_var(key, value):
    for item in vars_list:
        if isinstance(item, dict) and key in item:
            item[key] = value
            return
    vars_list.append({key: value})

set_var('MAX_SFT_STEPS', 20)
set_var('MAX_GRPO_STEPS', 5)
set_var('SFT_TIMEOUT_HOURS', 4)
set_var('GRPO_TIMEOUT_HOURS', 4)
set_var('GRPO_NUM_GENERATIONS', 1)
set_var('GRPO_NUM_GENERATIONS_EVAL', 1)
set_var('GRPO_MAX_COMPLETION_LENGTH', 32)
set_var('HF_UPLOAD_AFTER_TRAINING', '0')

path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding='utf-8')
print('Smoke config patched:', path)
PY
```

Запустите managed launcher:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh hf-full-managed
```

В начале remote logs теперь должен появиться блок:

```text
[gpu-check] status:
```

Успешный вариант выглядит примерно так:

```json
{
  "ok": true,
  "torch_version": "2.5.1+cu121",
  "torch_cuda_build": "12.1",
  "cuda_available": true,
  "device_count": 1,
  "bf16_supported": true
}
```

Если `ok: false`, откройте:

```text
reports/hf_top_papers_qwen3vl_8b_datasphere/gpu_preflight_status.json
```

или скачайте outputs через `download` по job id.

## 9. Мониторинг job

Список jobs:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh list
```

Информация о job:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh get <job_id>
```

Подключиться к логам:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh attach <job_id>
```

Скачать outputs вручную:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh download <job_id>
```

Выставить короткий TTL:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh ttl <job_id> 1
```

Остановить job вручную:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh cancel <job_id>
```

## 10. Проверка smoke outputs

```bash
ls -lh outputs/*hf_top_papers*qwen3vl*tar.gz 2>/dev/null || true
ls -lh reports/*hf_top_papers*qwen3vl*tar.gz 2>/dev/null || true

find reports -maxdepth 3 -type f \
  \( -name 'gpu_preflight_status.json' -o -name 'budget_plan.json' -o -name 'final_summary.json' \) \
  -print
```

Если smoke дошёл до SFT/GRPO, проверьте:

```bash
python -m json.tool reports/hf_top_papers_qwen3vl_8b_datasphere/final_summary.json | head -120
```

## 11. Верните full config перед настоящим запуском

После успешного smoke верните full параметры:

```bash
python - <<'PY'
from pathlib import Path
import yaml

path = Path('experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_full_g2_2.yaml')
cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
vars_list = cfg['env']['vars']

def set_var(key, value):
    for item in vars_list:
        if isinstance(item, dict) and key in item:
            item[key] = value
            return
    vars_list.append({key: value})

set_var('MAX_SFT_STEPS', 480)
set_var('MAX_GRPO_STEPS', 160)
set_var('SFT_TIMEOUT_HOURS', 48)
set_var('GRPO_TIMEOUT_HOURS', 35)
set_var('GRPO_NUM_GENERATIONS', 2)
set_var('GRPO_NUM_GENERATIONS_EVAL', 2)
set_var('GRPO_MAX_COMPLETION_LENGTH', 384)
set_var('HF_UPLOAD_AFTER_TRAINING', '1')

path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding='utf-8')
print('Full config restored:', path)
PY
```

Убедитесь, что `HFTOKEN` создан в DataSphere project, затем запустите full run:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh hf-full-managed
```

## 12. Если GPU check всё ещё падает

### Сценарий A: `cuda_available: false`

Проверьте, что DataSphere job действительно стартует на GPU:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh get <job_id>
```

Затем скачайте служебные логи:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh download <job_id>
```

Ищите:

```text
system.log
gpu_stats.tsv
docker_stats.tsv
stdout.txt
stderr.txt
reports/.../gpu_preflight_status.json
```

Если `nvidia-smi` тоже не видит GPU, проблема не в Python packages, а в DataSphere runtime/quota/instance type.

### Сценарий B: `cuda_available: true`, но `bf16_supported: false`

Для Qwen3-VL-8B pipeline с `--bf16` нужна BF16-capable GPU. Используйте A100/H100-class GPU или меняйте precision осознанно. Не переключайте Qwen3-VL-8B training на CPU.

### Сценарий C: PyTorch всё ещё ставится не тот

Проверьте remote log установки dependencies. В логах должно быть:

```text
torch==2.5.1+cu121
torchvision==0.20.1+cu121
torchaudio==2.5.1+cu121
```

Если DataSphere cache подтянул старые зависимости, запустите новый job после изменения requirements и проверьте, что CLI загрузил обновлённые файлы.

## 13. Источники для сверки

- PyTorch previous versions / CUDA wheel index: https://pytorch.org/get-started/previous-versions/
- PyTorch local install selector: https://pytorch.org/get-started/locally/
- Yandex DataSphere Jobs CLI: https://yandex.cloud/ru/docs/datasphere/concepts/jobs/cli
- Hugging Face Trainer / TrainingArguments: https://huggingface.co/docs/transformers/main_classes/trainer
- Hugging Face environment variable `HF_TOKEN`: https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables

## 14. Что было проверено при подготовке исправленного архива

Локально выполнено:

```text
[OK] py_compile для run_full_pipeline.py, check_gpu_before_pipeline.py и training scripts
[OK] bash -n для run_hf_top_papers_sft_grpo_full.sh и launch_examples.sh
[OK] parse_job_id больше не принимает /tmp/datasphere/job_... как job id
[OK] YAML configs читаются через PyYAML и используют Python 3.10
[OK] run_full_pipeline.py --dry-run строит корректную команду execute
```

Полный DataSphere GPU run здесь не выполнялся: для него нужен ваш DataSphere project, quota, GPU runtime и секреты.
