# Полный tutorial: VLM SFT + GRPO + auto-upload на Hugging Face через Yandex DataSphere Jobs

Дата ревизии: 2026-05-16.

Этот tutorial описывает полный цикл эксперимента: подготовка локального окружения и проекта DataSphere, запуск job, мониторинг, скачивание результатов, проверка артефактов и автоматическая публикация финальной модели и всех job-артефактов в Hugging Face model repository `top-papers/Qwen3-VL-8B-Instruct-scireason`.

## 0. Что запускается

Основной запуск находится здесь:

- job config: `experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_full_g2_2.yaml`;
- runtime wrapper внутри DataSphere: `experiments/vlm_finetuning/datasphere/bin/run_hf_top_papers_sft_grpo_full.sh`;
- Hugging Face uploader: `experiments/vlm_finetuning/scripts/upload_hf_finetuned_artifacts.py`;
- managed launcher с локальной стороны: `experiments/vlm_finetuning/datasphere/run_full_pipeline.py`;
- helper CLI: `experiments/vlm_finetuning/datasphere/launch_examples.sh`;
- сборка JSONL из HF dataset: `experiments/vlm_finetuning/scripts/build_hf_graph_experts_dataset.py`;
- SFT entrypoint: `experiments/vlm_finetuning/scripts/train_vlm_sft.py`;
- GRPO entrypoint: `experiments/vlm_finetuning/scripts/train_vlm_grpo.py`.

Pipeline делает четыре основные стадии:

1. скачивает HF dataset и сохраняет изображения в `data/derived/hf_top_papers_graph_experts/images/`;
2. строит `sft_train.jsonl`, `sft_eval.jsonl`, `grpo_train.jsonl`, `grpo_eval.jsonl`;
3. обучает SFT LoRA adapter, затем запускает GRPO поверх SFT adapter и упаковывает результаты;
4. после успешного GRPO автоматически собирает Hugging Face upload bundle и загружает финальную модель/adapter + все артефакты в `top-papers/Qwen3-VL-8B-Instruct-scireason`.

## 1. Предварительные требования

Нужно иметь:

- аккаунт Yandex Cloud с DataSphere;
- DataSphere community и проект;
- права на запуск Jobs в проекте;
- доступную GPU-конфигурацию `g2.2` в community;
- локальную машину с Linux/macOS или WSL, Python 3.10-3.12 и shell `bash`;
- достаточно места локально для скачанных outputs;
- интернет-доступ у DataSphere job для загрузки модели, датасета и последующей публикации в Hugging Face;
- Hugging Face token с write-доступом к model repo или организации `top-papers`, доступный внутри DataSphere job как `HF_TOKEN` или `HUGGING_FACE_HUB_TOKEN`.

Основной job использует `g2.2` и расширенную рабочую директорию SSD `1024Gb`. Это важно, потому что датасет занимает несколько гигабайт, а модельные кеши Hugging Face, промежуточные чекпойнты и upload bundle могут быть существенно больше исходного датасета.

## 2. Подготовьте локальное окружение

Все команды ниже выполняйте из корня репозитория.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U datasphere pyyaml huggingface_hub

datasphere version
```

`pyyaml` нужен для локальных preflight-проверок YAML. Внутри DataSphere job зависимости берутся из `experiments/vlm_finetuning/datasphere/requirements.txt`; там уже есть `huggingface_hub`, `hf-xet`, `transformers`, `trl`, `peft` и остальные runtime-зависимости.

## 3. Аутентификация в Yandex Cloud / DataSphere

Есть два типовых варианта.

### Вариант A: через профиль Yandex Cloud CLI

Настройте профиль `yc`, затем запускайте `datasphere` без явного токена:

```bash
yc init

datasphere project list -c <community_id>
```

### Вариант B: через OAuth token

DataSphere CLI поддерживает параметр `-t TOKEN`. Можно хранить токен в переменной окружения и передавать его явно:

```bash
export YC_OAUTH_TOKEN='<oauth_token>'
datasphere -t "$YC_OAUTH_TOKEN" project list -c <community_id>
```

Дальше в tutorial используется короткая форма без `-t`. Если вы используете token-вариант, добавляйте `-t "$YC_OAUTH_TOKEN"` к командам `datasphere` или настройте профиль `yc`.

## 4. Найдите project id и выставьте переменную

Если известен `community_id`, получите список проектов:

```bash
datasphere project list -c <community_id>
```

Затем сохраните id нужного проекта:

```bash
export DATASPHERE_PROJECT_ID='<project_id>'
```

Проверка:

```bash
datasphere project get --id "$DATASPHERE_PROJECT_ID"
```

## 5. Подготовьте Hugging Face upload token

Автозагрузка включена по умолчанию в job config:

```yaml
HF_UPLOAD_AFTER_TRAINING: "1"
HF_REPO_ID: top-papers/Qwen3-VL-8B-Instruct-scireason
HF_REPO_TYPE: model
HF_REVISION: main
HF_UPLOAD_PATH_PREFIX: ""
```

Токен нельзя коммитить в репозиторий. Рекомендуемый вариант — создать Hugging Face token с write-доступом и добавить его в DataSphere project secrets под именем `HF_TOKEN`, чтобы он был доступен job как переменная окружения.

Проверка локально, что token валиден:

```bash
export HF_TOKEN='<hf_write_token>'
python - <<'PY'
from huggingface_hub import HfApi
api = HfApi(token=None)
print(api.whoami())
PY
```

Если в вашем DataSphere setup секреты не пробрасываются в Jobs автоматически, используйте временный локальный override только для запуска и не коммитьте его:

```yaml
# experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_full_g2_2.yaml
env:
  vars:
    - HF_TOKEN: <hf_write_token>
```

Если автозагрузку нужно временно выключить, поставьте:

```yaml
- HF_UPLOAD_AFTER_TRAINING: "0"
```

## 6. Локальные preflight-проверки репозитория

Эти проверки не запускают обучение и не требуют GPU.

```bash
python -m py_compile \
  experiments/vlm_finetuning/scripts/build_hf_graph_experts_dataset.py \
  experiments/vlm_finetuning/scripts/train_vlm_sft.py \
  experiments/vlm_finetuning/scripts/train_vlm_grpo.py \
  experiments/vlm_finetuning/scripts/train_vlm_dpo.py \
  experiments/vlm_finetuning/scripts/upload_hf_finetuned_artifacts.py \
  experiments/vlm_finetuning/datasphere/run_full_pipeline.py

bash -n experiments/vlm_finetuning/datasphere/launch_examples.sh
for f in experiments/vlm_finetuning/datasphere/bin/*.sh; do bash -n "$f"; done
```

Проверьте, что DataSphere configs читаются и не содержат несовместимого сочетания `root-path` + `local-paths`:

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

Сделайте dry-run managed launcher:

```bash
python experiments/vlm_finetuning/datasphere/run_full_pipeline.py \
  --project-id "$DATASPHERE_PROJECT_ID" \
  --dry-run
```

Dry-run должен вывести manifest с командой `datasphere project job execute ...`; сам job при этом не стартует.

## 7. Запустите полный эксперимент

Рекомендуемый способ — managed launcher. Он запускает job, пишет локальный manifest, пытается распарсить job id, после завершения скачивает объявленные outputs и выставляет короткий TTL для данных job.

```bash
export DATASPHERE_PROJECT_ID='<project_id>'

bash experiments/vlm_finetuning/datasphere/launch_examples.sh hf-full-managed
```

Эквивалентный прямой запуск через DataSphere CLI:

```bash
datasphere project job execute \
  -p "$DATASPHERE_PROJECT_ID" \
  -c experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_full_g2_2.yaml
```

Managed-вариант удобнее для длинных запусков, потому что после окончания он дополнительно вызывает:

```bash
datasphere project job set-data-ttl --id <job_id> --days 1
datasphere project job download-files --id <job_id>
```

## 8. Что именно загружается на Hugging Face

После успешного GRPO wrapper вызывает:

```bash
python experiments/vlm_finetuning/scripts/upload_hf_finetuned_artifacts.py \
  --repo-id "$HF_REPO_ID" \
  --repo-type "$HF_REPO_TYPE" \
  --revision "$HF_REVISION" \
  --path-in-repo "$HF_UPLOAD_PATH_PREFIX" \
  --base-model "$BASE_MODEL" \
  --dataset-id "$DATASET_ID" \
  --out-prefix "$OUT_PREFIX" \
  --data-dir "$DATA_DIR" \
  --sft-dir "$SFT_DIR" \
  --grpo-dir "$GRPO_DIR" \
  --report-dir "$REPORT_DIR" \
  --bundle-dir "$HF_UPLOAD_BUNDLE_DIR"
```

Uploader делает следующее:

- готовит bundle в `reports/hf_top_papers_qwen3vl_8b_datasphere/hf_upload_bundle/`;
- кладет root-файлы финального GRPO adapter/processor в корень bundle, чтобы repo можно было использовать как финальный PEFT/LoRA adapter;
- сохраняет полный SFT adapter в `artifacts/sft_lora/`;
- сохраняет полный GRPO output в `artifacts/grpo_lora/`;
- сохраняет архивы `.tar.gz` в `artifacts/archives/`;
- сохраняет `summary.json` и train/eval JSONL в `artifacts/data/`;
- сохраняет budget/final/upload manifests в `artifacts/reports/`;
- генерирует `README.md` model card и `.gitattributes` для LFS-файлов;
- создает repo при необходимости через `HfApi.create_repo(..., exist_ok=True)`;
- загружает весь bundle через `HfApi.upload_folder(...)`.

Итоговый target по умолчанию:

```text
https://huggingface.co/top-papers/Qwen3-VL-8B-Instruct-scireason
```

Если нужно складывать результаты в подпапку repo, задайте, например:

```yaml
- HF_UPLOAD_PATH_PREFIX: runs/2026-05-16-full-sft-grpo
```

## 9. Мониторинг и управление job

Список jobs проекта:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh list
```

Информация по конкретному job:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh get <job_id>
```

Подключиться к логам, если локальная shell-сессия оборвалась:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh attach <job_id>
```

Остановить job:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh cancel <job_id>
```

Изменить TTL кеша, логов и результатов job:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh ttl <job_id> 1
```

Скачать outputs вручную:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh download <job_id>
```

## 10. Где искать результаты локально / в DataSphere outputs

После успешного запуска ожидаются файлы:

```text
data/derived/hf_top_papers_graph_experts/summary.json
data/derived/hf_top_papers_graph_experts/sft_train.jsonl
data/derived/hf_top_papers_graph_experts/sft_eval.jsonl
data/derived/hf_top_papers_graph_experts/grpo_train.jsonl
data/derived/hf_top_papers_graph_experts/grpo_eval.jsonl
outputs/hf_top_papers_qwen3vl_8b_sft_lora.tar.gz
outputs/hf_top_papers_qwen3vl_8b_grpo_lora.tar.gz
reports/hf_top_papers_qwen3vl_8b_datasphere_reports.tar.gz
reports/hf_top_papers_qwen3vl_8b_datasphere/budget_plan.json
reports/hf_top_papers_qwen3vl_8b_datasphere/final_summary.json
reports/hf_top_papers_qwen3vl_8b_datasphere/hf_upload_summary.json
reports/hf_top_papers_qwen3vl_8b_datasphere/hf_upload_bundle/artifacts/reports/hf_upload_manifest.json
reports/hf_top_papers_qwen3vl_8b_datasphere/artifact_manifest.txt
```

Managed launcher также пишет локальные файлы в:

```text
reports/datasphere_cli_runs/
```

Там находятся лог `*.log` и manifest `*.manifest.json` с командой запуска, project id, job id, статусом и TTL.

## 11. Как поменять модель, шаги, бюджет и Hugging Face target

Основные параметры находятся в `job_configs/hf_top_papers_sft_grpo_full_g2_2.yaml`, секция `env.vars`:

```yaml
BASE_MODEL: Qwen/Qwen3-VL-8B-Instruct
OUT_PREFIX: hf_top_papers_qwen3vl_8b
HF_UPLOAD_AFTER_TRAINING: "1"
HF_REPO_ID: top-papers/Qwen3-VL-8B-Instruct-scireason
HF_REPO_TYPE: model
HF_REVISION: main
HF_UPLOAD_PATH_PREFIX: ""
BUDGET_RUB: 100000
G2_2_RUB_PER_HOUR: 1085.76
MAX_SFT_STEPS: 180
MAX_GRPO_STEPS: 80
DATA_TIMEOUT_HOURS: 4
SFT_TIMEOUT_HOURS: 30
GRPO_TIMEOUT_HOURS: 45
```

Более быстрый smoke-вариант:

```yaml
BASE_MODEL: Qwen/Qwen2.5-VL-3B-Instruct
OUT_PREFIX: hf_top_papers_qwen25vl_3b_smoke
MAX_SFT_STEPS: 40
MAX_GRPO_STEPS: 10
SFT_TIMEOUT_HOURS: 8
GRPO_TIMEOUT_HOURS: 8
HF_UPLOAD_PATH_PREFIX: runs/smoke-qwen25vl-3b
```

Более длинный вариант:

```yaml
MAX_SFT_STEPS: 300
MAX_GRPO_STEPS: 120
SFT_TIMEOUT_HOURS: 45
GRPO_TIMEOUT_HOURS: 60
HF_UPLOAD_PATH_PREFIX: runs/full-300-120
```

Локальные `timeout` в shell wrapper защищают от бесконечного зависания процесса, но они не являются полноценным лимитом расходов аккаунта. Для строгого лимита используйте механизмы контроля расходов в Yandex Cloud/DataSphere и выставляйте TTL для данных job.

## 12. Проверка артефактов после скачивания

```bash
ls -lh outputs/*hf_top_papers*qwen3vl*tar.gz
ls -lh reports/*hf_top_papers*qwen3vl*tar.gz
python -m json.tool data/derived/hf_top_papers_graph_experts/summary.json | head -80
python -m json.tool reports/hf_top_papers_qwen3vl_8b_datasphere/final_summary.json | head -120
python -m json.tool reports/hf_top_papers_qwen3vl_8b_datasphere/hf_upload_summary.json | head -120
```

Проверка первых JSONL строк:

```bash
python - <<'PY'
import json
from pathlib import Path
for path in [
    Path('data/derived/hf_top_papers_graph_experts/sft_train.jsonl'),
    Path('data/derived/hf_top_papers_graph_experts/grpo_train.jsonl'),
]:
    print('\n---', path)
    with path.open(encoding='utf-8') as f:
        row = json.loads(next(f))
    print(json.dumps({k: row.get(k) for k in ['id', 'task_family', 'image', 'images', 'label_text', 'reference_label']}, ensure_ascii=False, indent=2))
PY
```

Проверка upload bundle manifest:

```bash
python -m json.tool \
  reports/hf_top_papers_qwen3vl_8b_datasphere/hf_upload_bundle/artifacts/reports/hf_upload_manifest.json \
  | head -120
```

## 13. Troubleshooting

### `HF upload is enabled, but HF_TOKEN/HUGGING_FACE_HUB_TOKEN is not set`

Автозагрузка включена, но внутри DataSphere job нет Hugging Face token. Добавьте project secret `HF_TOKEN` или временно пропишите `HF_TOKEN` в локальный job config перед запуском. Не коммитьте токен.

### `403 Forbidden` или `Repository Not Found` при upload

Токен не имеет write-доступа к `top-papers/Qwen3-VL-8B-Instruct-scireason` или к организации `top-papers`. Проверьте scope токена и права пользователя в организации.

### Нужно загрузить в подпапку, не перезаписывая корень repo

Задайте:

```yaml
- HF_UPLOAD_PATH_PREFIX: runs/<run_name>
```

### `Set DATASPHERE_PROJECT_ID before running this command`

Вы не выставили project id:

```bash
export DATASPHERE_PROJECT_ID='<project_id>'
```

### `DataSphere CLI is not available`

Активируйте виртуальное окружение и установите CLI:

```bash
source .venv/bin/activate
python -m pip install -U datasphere
datasphere version
```

### Job не видит файлы репозитория

Запускайте команды из корня репозитория. Внутри job shell wrappers сами переходят в корень через `common.sh`.

### Ошибка DataSphere config про `root-path` / `local-paths`

В исправленных configs используется только `local-paths`. Если вы создаете новый YAML, не задавайте одновременно `root-path` и `local-paths` в `env.python`.

### Недостаточно места

Проверьте `working-storage` в YAML. Для полного HF pipeline выставлено:

```yaml
working-storage:
  type: SSD
  size: 1024Gb
```

Для экспериментов с меньшей моделью можно уменьшить storage, но для 8B + кешей HF + чекпойнтов + upload bundle лучше оставить запас.

### CUDA OOM на SFT или GRPO

Сначала уменьшите:

```yaml
MAX_SFT_STEPS: 80
MAX_GRPO_STEPS: 20
```

Затем в `run_hf_top_papers_sft_grpo_full.sh` уменьшите эффективный batch:

```bash
SFT_PER_DEVICE_BATCH=1
SFT_GRAD_ACCUM=4
GRPO_PER_DEVICE_BATCH=1
GRPO_GRAD_ACCUM=4
GRPO_NUM_GENERATIONS=2
GRPO_MAX_COMPLETION_LENGTH=32
```

Если OOM сохраняется, используйте `Qwen/Qwen2.5-VL-3B-Instruct` или переходите на более крупную GPU-конфигурацию, если она разрешена в вашем community.

### `peft.PeftModel is required for --sft-adapter-path`

Установлен неполный или несовместимый `peft`. Внутри DataSphere job зависимости берутся из `datasphere/requirements.txt`; локально обновите окружение:

```bash
python -m pip install -U peft
```

### Локальная сессия оборвалась

Job продолжает выполняться в DataSphere. Найдите job id и подключитесь:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh list
bash experiments/vlm_finetuning/datasphere/launch_examples.sh attach <job_id>
```

### Outputs не скачались автоматически

Скачайте вручную:

```bash
bash experiments/vlm_finetuning/datasphere/launch_examples.sh download <job_id>
```

## 14. Источники, по которым сверялась DataSphere/Hugging Face-часть

- Official DataSphere Jobs docs: https://yandex.cloud/ru/docs/datasphere/concepts/jobs/
- Official DataSphere CLI docs: https://yandex.cloud/ru/docs/datasphere/concepts/jobs/cli
- Official DataSphere configurations: https://yandex.cloud/ru/docs/datasphere/concepts/configurations
- Official DataSphere pricing: https://yandex.cloud/ru/docs/datasphere/pricing
- Official DataSphere secrets docs: https://yandex.cloud/ru/docs/datasphere/operations/data/secrets
- Hugging Face Hub upload guide: https://huggingface.co/docs/huggingface_hub/guides/upload
- Hugging Face Hub CLI guide: https://huggingface.co/docs/huggingface_hub/guides/cli
- Hugging Face model upload docs: https://huggingface.co/docs/hub/models-uploading
- HF dataset card: https://huggingface.co/datasets/top-papers/top-papers-graph-experts-data
- TRL SFT Trainer docs: https://huggingface.co/docs/trl/sft_trainer
- TRL GRPO Trainer docs: https://huggingface.co/docs/trl/grpo_trainer

Notion guide URL from the original task was checked earlier but returned an unavailable page from this environment. Therefore the executable changes were aligned with official Yandex Cloud documentation, Hugging Face documentation and the current repository structure.
