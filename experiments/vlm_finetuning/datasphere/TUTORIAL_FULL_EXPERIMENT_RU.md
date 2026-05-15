# Полный tutorial: VLM SFT + GRPO на `top-papers/top-papers-graph-experts-data` через Yandex DataSphere Jobs

Дата ревизии: 2026-05-15.

Этот tutorial описывает полный цикл эксперимента: от подготовки локального окружения и проекта DataSphere до запуска job, мониторинга, скачивания результатов и проверки артефактов. Целевой сценарий — дообучение VLM на Hugging Face датасете `top-papers/top-papers-graph-experts-data` с помощью DataSphere Jobs.

## 0. Что запускается

Основной запуск находится здесь:

- job config: `experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_full_g2_2.yaml`;
- runtime wrapper внутри DataSphere: `experiments/vlm_finetuning/datasphere/bin/run_hf_top_papers_sft_grpo_full.sh`;
- managed launcher с локальной стороны: `experiments/vlm_finetuning/datasphere/run_full_pipeline.py`;
- helper CLI: `experiments/vlm_finetuning/datasphere/launch_examples.sh`;
- сборка JSONL из HF dataset: `experiments/vlm_finetuning/scripts/build_hf_graph_experts_dataset.py`;
- SFT entrypoint: `experiments/vlm_finetuning/scripts/train_vlm_sft.py`;
- GRPO entrypoint: `experiments/vlm_finetuning/scripts/train_vlm_grpo.py`.

Pipeline делает три основные стадии:

1. скачивает HF dataset и сохраняет изображения в `data/derived/hf_top_papers_graph_experts/images/`;
2. строит `sft_train.jsonl`, `sft_eval.jsonl`, `grpo_train.jsonl`, `grpo_eval.jsonl`;
3. обучает SFT LoRA адаптер, затем запускает GRPO поверх SFT-адаптера и упаковывает результаты.

## 1. Предварительные требования

Нужно иметь:

- аккаунт Yandex Cloud с DataSphere;
- DataSphere community и проект;
- права на запуск Jobs в проекте;
- доступную GPU-конфигурацию `g2.2` в community;
- локальную машину с Linux/macOS или WSL, Python 3.10-3.12 и shell `bash`;
- достаточно места локально для скачанных outputs;
- интернет-доступ у DataSphere job для загрузки модели и датасета с Hugging Face.

Основной job использует `g2.2` и расширенную рабочую директорию SSD `1024Gb`. Это важно, потому что датасет занимает несколько гигабайт, а модельные кеши Hugging Face и промежуточные чекпойнты могут быть существенно больше исходного датасета.

## 2. Подготовьте локальное окружение

Все команды ниже выполняйте из корня репозитория.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U datasphere pyyaml

datasphere version
```

`pyyaml` нужен только для локальных preflight-проверок YAML. Внутри DataSphere job зависимости берутся из `experiments/vlm_finetuning/datasphere/requirements.txt`.

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

## 5. Локальные preflight-проверки репозитория

Эти проверки не запускают обучение и не требуют GPU.

```bash
python -m py_compile \
  experiments/vlm_finetuning/scripts/build_hf_graph_experts_dataset.py \
  experiments/vlm_finetuning/scripts/train_vlm_sft.py \
  experiments/vlm_finetuning/scripts/train_vlm_grpo.py \
  experiments/vlm_finetuning/scripts/train_vlm_dpo.py \
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

## 6. Запустите полный эксперимент

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

## 7. Мониторинг и управление job

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

## 8. Где искать результаты

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
reports/hf_top_papers_qwen3vl_8b_datasphere/artifact_manifest.txt
```

Managed launcher также пишет локальные файлы в:

```text
reports/datasphere_cli_runs/
```

Там находятся лог `*.log` и manifest `*.manifest.json` с командой запуска, project id, job id, статусом и TTL.

## 9. Как поменять модель, шаги и бюджетные guardrails

Основные параметры находятся в `job_configs/hf_top_papers_sft_grpo_full_g2_2.yaml`, секция `env.vars`:

```yaml
BASE_MODEL: Qwen/Qwen3-VL-8B-Instruct
OUT_PREFIX: hf_top_papers_qwen3vl_8b
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
BASE_MODEL: Qwen/Qwen3-VL-8B-Instruct
OUT_PREFIX: hf_top_papers_qwen3vl_8b_smoke
MAX_SFT_STEPS: 40
MAX_GRPO_STEPS: 10
SFT_TIMEOUT_HOURS: 8
GRPO_TIMEOUT_HOURS: 8
```

Более длинный вариант:

```yaml
MAX_SFT_STEPS: 300
MAX_GRPO_STEPS: 120
SFT_TIMEOUT_HOURS: 45
GRPO_TIMEOUT_HOURS: 60
```

Локальные `timeout` в shell wrapper защищают от бесконечного зависания процесса, но они не являются полноценным лимитом расходов аккаунта. Для строгого лимита используйте механизмы контроля расходов в Yandex Cloud/DataSphere и выставляйте TTL для данных job.

## 10. Что именно исправлено для надежного запуска команд

В этой версии репозитория были устранены проблемы, которые обычно ломают запуск DataSphere Jobs:

- `datasphere/bin/common.sh` теперь корректно переходит в корень репозитория из `experiments/vlm_finetuning/datasphere/bin/`;
- YAML configs используют `local-paths` без несовместимого `root-path`;
- `working-storage` нормализован как `type: SSD` и размеры вида `100Gb`, `250Gb`, `500Gb`, `700Gb`, `1024Gb`;
- HF builder сохраняет изображения в `--out-dir/images` и пишет корректные пути в JSONL;
- SFT/GRPO normalizers понимают `image`, `images`, `messages`, `chat.messages`, `prompt_chat.messages` и image blocks;
- GRPO entrypoint корректно сообщает, если для `--sft-adapter-path` не установлен полноценный `peft.PeftModel`;
- managed launcher стримит длинные DataSphere logs, а не копит весь stdout в памяти;
- helper `launch_examples.sh` валидирует обязательные аргументы для `get`, `attach`, `cancel`, `ttl`, `download`.

## 11. Troubleshooting

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

Для экспериментов с меньшей моделью можно уменьшить storage, но для Qwen3-VL-8B + кешей HF + чекпойнтов лучше оставить запас.

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

Если OOM сохраняется, сначала уменьшите GRPO-параметры (`GRPO_NUM_GENERATIONS=1`, `GRPO_MAX_COMPLETION_LENGTH=32`), затем временно переключитесь на меньший Qwen3-VL вариант или переходите на более крупную GPU-конфигурацию, если она разрешена в вашем community.

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

## 12. Проверка артефактов после скачивания

```bash
ls -lh outputs/*hf_top_papers*qwen3vl*tar.gz
ls -lh reports/*hf_top_papers*qwen3vl*tar.gz
python -m json.tool data/derived/hf_top_papers_graph_experts/summary.json | head -80
python -m json.tool reports/hf_top_papers_qwen3vl_8b_datasphere/final_summary.json | head -120
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

## 13. Источники, по которым сверялась DataSphere-часть

- Official DataSphere Jobs docs: https://yandex.cloud/ru/docs/datasphere/concepts/jobs/
- Official DataSphere CLI docs: https://yandex.cloud/ru/docs/datasphere/concepts/jobs/cli
- Official DataSphere configurations: https://yandex.cloud/ru/docs/datasphere/concepts/configurations
- Official DataSphere pricing: https://yandex.cloud/ru/docs/datasphere/pricing
- HF dataset card: https://huggingface.co/datasets/top-papers/top-papers-graph-experts-data
- TRL SFT Trainer docs: https://huggingface.co/docs/trl/sft_trainer
- TRL GRPO Trainer docs: https://huggingface.co/docs/trl/grpo_trainer
- Qwen3-VL-8B-Instruct HF model card: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- Transformers Qwen3-VL docs: https://huggingface.co/docs/transformers/model_doc/qwen3_vl

Notion guide URL from the task was checked, but returned an unavailable page from this environment. Therefore the executable changes were aligned with official Yandex Cloud documentation and the current repository structure.
