<!-- SPDX-FileCopyrightText: 2026 top-papers-graph contributors -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# Следующие шаги: от очереди к публикационному A/B-эксперименту

Этот документ описывает исполнимый путь от текущей заблокированной ревизии benchmark до строгого
сравнения base и SciReason adapter. Команды ниже нужно запускать из корня `top-papers-graph`.

Текущую ревизию benchmark нельзя использовать для научного вывода: все 386 исходных строк имеют
блокирующие нарушения. Ближайшая работа является ручной курацией, а не inference.

## 1. Что уже зафиксировано

Неизменяемая очередь находится в:

```text
runs/vlm_ab/qwen3vl-scireason-remediation-audit-v2/curation_queue_v2_524cbc2d_hardened
```

Контрольные значения:

| Артефакт | Значение |
| --- | --- |
| Prepare manifest SHA256 | `c52fe431c63cb0e9d27f5d8133c9639942e8bfc66c24549beb07b2895f6e02d3` |
| Audit JSON SHA256 | `a409b86cf7e50611b923ff83b7c28211a40f55ff0e7946b57eef081689b1e05f` |
| Queue fingerprint | `bc15c288024c6c40a6a9274c13d26ea1e675994f7353bd98a6c35c072364cf25` |
| Queue manifest SHA256 | `37449b411cb729f0fb2f4afbbf95998249e68a0a7d51c1794d9eb4d504416caf` |
| Tasks SHA256 | `55445796b8deca312f2795721d6b45e0d4960afc506dc224572893901f596780` |
| Decision template SHA256 | `4a5c5a6468f3bd6b7461a771e22af61741d39ac18a3d0b18015722d38069528e` |
| Число задач | `386` |
| Минимум primary papers | `240` |

`tasks.jsonl`, `decision_template.jsonl` и `queue_manifest.json` нельзя редактировать, дополнять или
перемещать внутрь другого queue directory. Итоговые решения и новые изображения хранятся рядом с
очередью, но не внутри нее.

Эта queue связана с архивным audit contract v2. Текущий код создает новые audits v3, но при
`curate-assemble` воспроизводит version из сохраненного report. Не запускайте новый `prepare` в
старый `$Run`: при утрате bundle восстановите его из доверенного архива. Новый re-audit получает
новые experiment ID, output directory, queue и baseline hashes.

## 2. Разделение ответственности

| Роль | Ответственность |
| --- | --- |
| Curator | Проверяет статью, формулирует standalone prompt и готовит полную replacement row |
| Второй curator | Независимо проверяет retain/exclude decision и содержимое replacement row |
| Image verifier | Проверяет paper, page, locator, URL, license и конкретные image bytes |
| Release owner | Разрешает training overlap, публикует benchmark, adapter и lineage revisions |
| Experiment owner | Фиксирует config, preregistration, запускает arms и защищает blinding |
| Reviewer | Оценивает только ослепленные пары ответов после inference |

Нельзя автоматически выбирать строку из duplicate group, назначать изображение статье, сочинять
license, считать prompt корректным по имени файла или подменять второго проверяющего. Код проверяет
целостность решения, но не может доказать научную истинность curator judgment.

## 3. Подготовка рабочего окружения

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[vlm_ab,dev]"

$Config = "experiments/vlm_ab_evaluation/configs/qwen3vl_scireason_remediation_audit_v2.yaml"
$Run = "runs/vlm_ab/qwen3vl-scireason-remediation-audit-v2"
$Queue = "$Run/curation_queue_v2_524cbc2d_hardened"
$Forms = "$Run/curator_workspace_v2"
$Decisions = "$Run/completed_decisions.jsonl"
$Curated = "$Run/curated_dataset"
$Candidate = "$Run/release_candidate"
```

Для Hub-запросов token передается только через environment или project secret:

```powershell
$SecureToken = Read-Host "Hugging Face token" -AsSecureString
$env:HF_TOKEN = [System.Net.NetworkCredential]::new("", $SecureToken).Password
Remove-Variable SecureToken
```

Token нельзя помещать в config, JSONL, notebook, shell history общего сервера или Git.

## 4. Проверка неизменяемой очереди

```powershell
(Get-FileHash -Algorithm SHA256 "$Run/prepare_manifest.json").Hash.ToLowerInvariant()
(Get-FileHash -Algorithm SHA256 "$Queue/queue_manifest.json").Hash.ToLowerInvariant()
(Get-FileHash -Algorithm SHA256 "$Queue/tasks.jsonl").Hash.ToLowerInvariant()
(Get-FileHash -Algorithm SHA256 "$Queue/decision_template.jsonl").Hash.ToLowerInvariant()
Get-ChildItem -LiteralPath $Queue -Force
```

Должны получиться четыре SHA256 из таблицы выше. В queue directory должны быть ровно три файла:
`queue_manifest.json`, `tasks.jsonl`, `decision_template.jsonl`. Любое расхождение означает, что
курацию нужно остановить и восстановить byte-identical очередь из доверенного архива.

## 5. Создание offline curator workspace

Основной путь курации -- автономная HTML-форма. Команда повторно проверяет prepare bundle и все три
неизменяемых queue-файла, затем атомарно создает отдельный workspace с HTML и дедуплицированными
копиями доступных audited source images:

```powershell
python experiments/vlm_ab_evaluation/run_pipeline.py `
  --config $Config `
  curate-forms `
  --prepare-manifest "$Run/prepare_manifest.json" `
  --queue-manifest "$Queue/queue_manifest.json" `
  --output-dir $Forms

# Перед раздачей или каждым первым открытием повторить буквально ту же команду.
python experiments/vlm_ab_evaluation/run_pipeline.py `
  --config $Config `
  curate-forms `
  --prepare-manifest "$Run/prepare_manifest.json" `
  --queue-manifest "$Queue/queue_manifest.json" `
  --output-dir $Forms

(Get-FileHash -Algorithm SHA256 "$Forms/workspace_manifest.json").Hash.ToLowerInvariant()
(Get-FileHash -Algorithm SHA256 "$Forms/curator.html").Hash.ToLowerInvariant()
Get-Content -Raw "$Forms/workspace_manifest.json"

Start-Process "$Forms/curator.html"
New-Item -ItemType Directory -Path $Curated
```

Повторная генерация принимает только byte-identical workspace. Любое изменение HTML, manifest или
скопированного изображения приводит к ошибке; такой workspace нельзя раздавать или открывать до
восстановления в новом чистом каталоге. Release owner передает экспертам ожидаемые SHA256 manifest и
HTML, а также `queue_fingerprint` из manifest. Тот же fingerprint виден в header HTML для сверки.
Это внешняя сверка: browser сам криптографически workspace не проверяет.

HTML не обращается к сети. Он сохраняет draft в browser `localStorage`, связанный с queue fingerprint;
если storage недоступен, форма продолжает работать в текущей вкладке и требует регулярно нажимать
**Export draft**. Каждый эксперт заполняет только назначенное ему подмножество `task_id` и передает
owner свой draft envelope. Owner в одном master workspace последовательно выбирает **Merge draft**.
Пустые записи не стирают работу, одинаковые непустые записи являются no-op, а разные непустые записи
для одного `task_id` останавливают весь import без частичных изменений. Конфликт нужно разрешить
экспертами, а не перезаписью.

После merge всех subsets и заполнения всех 386 задач owner нажимает
**Export completed_decisions.jsonl** и помещает файл в `$Decisions`. Форма не импортирует final JSONL:
такое обратное преобразование потеряло бы произвольные допустимые benchmark/provenance поля.
Финальный export содержит строки в исходном порядке и все immutable bindings. Browser validation не
заменяет серверную проверку последующим `curate-assemble`.

В секции каждого изображения кнопка **Выбрать проверенный файл и вычислить SHA256** читает выбранный
локальный файл через Web Crypto, заполняет lowercase SHA256 и показывает локальный preview. Bytes и
сам файл не сохраняются в draft или JSONL, а имя файла никогда не подставляется в `image_path`.
Эксперт отдельно задаёт canonical `assets/images/...` path и отдельно копирует ровно выбранные bytes
в `$Curated` по этому пути. Если Web Crypto недоступен, SHA256 вводится вручную и затем всё равно
проверяется `curate-assemble` по файлу в `$Curated`.

Assignments следует вести по `task_id` во внешнем журнале. Нельзя разрешать нескольким людям
одновременно перезаписывать общий JSONL. Ручное копирование `decision_template.jsonl` и редактирование
JSONL остается только аварийным fallback; при нем итог также должен содержать ровно 386 задач без
дубликатов и пропусков, а queue directory не изменяется.

## 6. Как читать задачу curator

Каждая строка `tasks.jsonl` содержит:

| Поле | Назначение |
| --- | --- |
| `task_id` | Неизменяемый идентификатор решения |
| `source_row_index` | Индекс исходной benchmark row |
| `original_row` | Исходная запись только для расследования, не шаблон исправления |
| `legacy_provenance_rows` | Все legacy provenance rows с тем же исходным `sample_id` |
| `audited_image_hashes` | SHA256 реально прочитанных исходных image bytes |
| `critical_codes` | Найденные blockers |
| `warning_codes` | Найденные warnings |
| `training_overlap_paper_ids` | Canonical paper IDs, найденные в фактическом train export |

Поля binding в соответствующей строке `completed_decisions.jsonl` копируются из template и никогда
не меняются:

```text
artifact_version
queue_fingerprint
task_id
prepare_manifest_sha256
source_benchmark_sha256
source_provenance_sha256
source_audit_sha256
source_row_index
source_row_sha256
```

Curators изменяют только `status`, `disposition`, `exclusion_reason`, `benchmark_row`,
`provenance_row`, `reviewed_by` и `notes`.

## 7. Решение `exclude`

Для исключенной исходной строки обязательны:

```json
{
  "status": "complete",
  "disposition": "exclude",
  "exclusion_reason": "Конкретная проверяемая причина исключения",
  "benchmark_row": null,
  "provenance_row": null,
  "reviewed_by": ["curator-01", "curator-02"],
  "notes": "Ссылка на внутренний журнал проверки без секретов"
}
```

Это только фрагмент редактируемых полей, а не полная decision row. Binding fields должны остаться в
строке. `reviewed_by` содержит минимум два реальных, ASCII и normalized-distinct идентификатора.

Допустимые причины возникают после проверки источника: статья недоступна для верификации, license
не позволяет требуемое распространение, корректные image bytes не найдены, строка является лишним
членом duplicate group, paper пересекается с training и выбран exclusion path. Причина не может
быть `bad row`, `duplicate` или другой непроверяемой заглушкой.

Каждая строка duplicate group получает отдельное решение. Curators должны явно определить, какая
строка сохраняется, либо исключить все. Одинаковый `sample_id` не доказывает идентичность строк.

Если adapter не переобучается, все benchmark papers из `training_overlap_paper_ids` должны быть
исключены целиком. Нельзя оставить другой sample той же статьи под новым `sample_id`.

## 8. Решение `retain`

Для сохраненной строки обязательны:

```json
{
  "status": "complete",
  "disposition": "retain",
  "exclusion_reason": null,
  "benchmark_row": {},
  "provenance_row": {},
  "reviewed_by": ["curator-01", "curator-02"],
  "notes": "Краткая трассировка решения"
}
```

`benchmark_row` и `provenance_row` являются полными replacement objects. Patch, список измененных
полей или ссылка на legacy row не принимаются.

### 8.1 Benchmark row

Минимальная форма выглядит так. Значения ниже иллюстрируют структуру и не являются curator facts.

```json
{
  "sample_id": "study-v2-paper-0001-figure-01",
  "paper_id": "doi:10.5555/example.1",
  "stratum": "multimodal_hard",
  "primary_endpoint": true,
  "model_task_prompt": "Describe the trend in Figure 2 and identify the interval where it reverses.",
  "messages": [
    {
      "role": "system",
      "content": [{"type": "text", "text": "Use only the supplied scientific evidence."}]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Describe the trend in Figure 2 and identify the interval where it reverses."
        },
        {"type": "image"}
      ]
    }
  ],
  "images": ["assets/images/study-v2/paper-0001/figure-02.png"],
  "split_provenance": {
    "paper_holdout": true,
    "source_holdout": true,
    "creator_holdout": true,
    "training_overlap_checked": true,
    "source_document_id": "source:publisher-record-0001",
    "creator_group_id": "creator-group:independent-0001"
  }
}
```

Правила benchmark row:

- `sample_id` уникален во всем corrected release.
- `paper_id` записан канонически: `doi:...`, `arxiv:...` без version suffix или `paper:...`.
- Скрытые, nested или encoded aliases `paper_id` запрещены.
- `multimodal_hard` и `temporal_hard` требуют `primary_endpoint=true`.
- `easy_control` требует `primary_endpoint=false` при текущем preregistration policy.
- `model_task_prompt` совпадает с объединенным текстом user messages после нормализации; безопаснее
  использовать буквально тот же текст.
- `messages` содержит только exact text/image blocks, предусмотренные schema.
- Число path-free `{"type":"image"}` placeholders равно числу элементов `images`.
- Порядок placeholders соответствует порядку `images` и provenance entries.
- Prompt является задачей для одной модели и не упоминает A/B, варианты, model comparison или
  ожидаемый ответ.
- `split_provenance` содержит только подтвержденные curator/release-owner assertions.
- Нормализованный prompt уникален среди retained rows и не совпадает с training prompt.

`gold_answer` и `rubric` можно не добавлять, пока `benchmark.require_gold=false` и primary endpoint
остается blinded human review. При `require_gold=true` оба объекта обязательны, должны содержать
содержательные evidence/facts/criteria и минимум двух normalized-distinct adjudicators. Нельзя
помещать gold или rubric в model-facing messages.

### 8.2 Provenance row

На каждый retained `sample_id` создается ровно одна aggregated provenance row:

```json
{
  "sample_id": "study-v2-paper-0001-figure-01",
  "paper_id": "doi:10.5555/example.1",
  "images": [
    {
      "image_path": "assets/images/study-v2/paper-0001/figure-02.png",
      "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "page": 7,
      "locator": "Figure 2, panel b",
      "source_url": "https://publisher.example/article/figure-2",
      "license": "Verified license statement",
      "verified_by": ["verifier-01", "verifier-02"],
      "citation": "Author et al., title, year, DOI"
    }
  ]
}
```

Пример SHA256 является заглушкой и должен быть заменен хешем реального файла. Правила provenance:

- `sample_id` и canonical `paper_id` точно совпадают с benchmark row.
- `images[*].image_path` точно совпадают с `benchmark_row.images` и имеют тот же порядок.
- Каждый path начинается с `assets/images/`, является POSIX-normalized и не содержит `..`, drive,
  URI, backslash или Windows aliases.
- Файл находится в `$Curated` по этому относительному path.
- Автоматический SHA256 в форме не копирует файл: выбранные bytes нужно отдельно поместить в
  `$Curated` по введённому `image_path`.
- `sha256` вычислен по фактическим bytes и записан lowercase.
- `page` и `locator` позволяют независимо найти evidence в статье.
- `source_url` является проверенным HTTP(S) URL конкретного источника.
- `license` описывает реально проверенное право использования, а не предположение.
- `citation` содержит достаточную библиографическую ссылку; assembly и strict audit требуют ее,
  хотя frozen v1 JSON Schema сохраняет поле optional для совместимости с immutable queue.
- `verified_by` содержит минимум два независимых normalized-distinct verifier ID.
- Одинаковые bytes не используются под разными paper IDs.
- В одной row не должно быть двух image paths с одинаковыми bytes.

Проверить отдельный файл можно так:

```powershell
(Get-FileHash -Algorithm SHA256 "$Curated/assets/images/study-v2/paper-0001/figure-02.png").Hash.ToLowerInvariant()
```

## 9. Контроль прогресса и assembly

После browser export можно проверить сводку decisions JSONL:

```powershell
python -c "import collections,json,pathlib,sys; rows=[json.loads(x) for x in pathlib.Path(sys.argv[1]).read_text(encoding='utf-8').splitlines() if x.strip()]; print('rows',len(rows)); print('status',collections.Counter(r['status'] for r in rows)); print('disposition',collections.Counter(str(r['disposition']) for r in rows))" $Decisions
```

`curate-assemble` можно запускать как validator во время работы:

```powershell
python experiments/vlm_ab_evaluation/run_pipeline.py `
  --config $Config `
  curate-assemble `
  --prepare-manifest "$Run/prepare_manifest.json" `
  --queue-manifest "$Queue/queue_manifest.json" `
  --decisions-jsonl $Decisions `
  --curated-dataset-root $Curated `
  --output-dir $Candidate
```

Пока есть pending rows, ожидается ошибка вида `decision <task_id> is still pending`. Это корректный
hard stop. Команда последовательно проверяет bindings, complete decisions, JSON Schemas, canonical
paper identity, images, provenance, training overlap, duplicate prompts/bytes и minimum 240 unique
primary papers.

После успешного assembly `$Candidate` содержит:

```text
data/task3_vlm_generation.jsonl
article_image_sources.jsonl
assets/...
audit/benchmark_audit.json
audit/benchmark_audit.md
assembly_manifest.json
```

Acceptance для этой стадии:

- все 386 decisions имеют `status="complete"`;
- число retained плюс excluded равно 386;
- retained `sample_id` и provenance rows имеют взаимно однозначное соответствие;
- critical findings равны нулю;
- `duplicate_normalized_prompt` и `within_row_duplicate_image_bytes` равны нулю;
- unique primary paper IDs не меньше 240;
- `technical_audit_passed=true`;
- `scope="validated_benchmark_release_candidate"`;
- `publication_ready=false` остается ожидаемым до fresh strict `prepare`.

Повторно проверьте hashes queue после курации. Они не должны измениться.

## 10. Публикация corrected benchmark

Release owner проверяет candidate audit и публикует `$Candidate` в новом immutable dataset commit.
Нельзя изменять audited commit `33ccc5ed08e314c6457dcaa23e7f7508406cb4f8` или заменять старый
run directory.

`upload_folder` возвращает SHA созданного commit, поэтому не нужно отдельно читать mutable `main`:

```powershell
$BenchmarkRepo = "owner/corrected-benchmark"
$B = python -c "import sys; from huggingface_hub import HfApi; result=HfApi().upload_folder(repo_id=sys.argv[1],repo_type='dataset',folder_path=sys.argv[2],commit_message='Publish curator-validated VLM benchmark candidate v2'); print(result.oid)" $BenchmarkRepo $Candidate
if ($LASTEXITCODE -ne 0 -or $B -notmatch '^[0-9a-f]{40}$') {
    throw "Benchmark upload failed or returned no immutable commit SHA"
}
$B
```

Сохраните полученный 40-character commit SHA как `B`. Не используйте `main`, tag или branch name в
strict config. Сам факт upload не делает dataset publication-ready: это доказывает только новый
immutable source для последующего `prepare`.

## 11. Публикация adapter revision `R` и lineage revision `M`

Нужны две разные immutable revisions одного adapter repository:

| Revision | Содержимое | Использование |
| --- | --- | --- |
| `R` | Evaluated weights, processor assets и exact SFT/GRPO exports | Inference и training audit |
| `M` | Lineage manifest, объявляющий training inputs at `R` | Отдельная attestation для `prepare` |

Разделение `R` и `M` обязательно. Manifest внутри `R`, который содержит SHA самого `R`, потребовал
бы найти cryptographic fixed point Git commit и практически не может быть опубликован.

Release owner выбирает один из двух научно допустимых путей:

1. Исключить из benchmark все training-overlap papers и опубликовать новую adapter revision `R` с
   проверенными неизменными weights, processor metadata и exact training exports.
2. Переобучить adapter с заранее зафиксированным held-out split, затем опубликовать новые weights и
   exact exports как `R`.

В обоих случаях `R` должен быть новым проверенным commit. До upload задайте в
`adapter_config.json` exact `base_model_name_or_path` и 40-character `revision` base model. Strict
`prepare` скачивает этот файл из `R`, проверяет оба значения и связывает его SHA256 с prepare
manifest.

Публикация локально проверенного adapter folder возвращает exact `R`:

```powershell
$AdapterRepo = "top-papers/Qwen3-VL-8B-Instruct-scireason"
$AdapterRelease = "<LOCAL_ADAPTER_RELEASE_DIR>"
$R = python -c "import sys; from huggingface_hub import HfApi; result=HfApi().upload_folder(repo_id=sys.argv[1],repo_type='model',folder_path=sys.argv[2],commit_message='Publish evaluated adapter and exact training exports R'); print(result.oid)" $AdapterRepo $AdapterRelease
if ($LASTEXITCODE -ne 0 -or $R -notmatch '^[0-9a-f]{40}$') {
    throw "Adapter upload failed or returned no immutable commit SHA"
}
$R
```

После получения SHA `R` создается manifest по
`schemas/training_lineage_manifest.schema.json`. Структура:

```json
{
  "schema_version": 1,
  "training_sources": [
    {
      "repo_id": "top-papers/Qwen3-VL-8B-Instruct-scireason",
      "repo_type": "model",
      "revision": "<R>",
      "files": [
        {
          "path": "artifacts/data/sft_all.jsonl",
          "sha256": "<lowercase-sha256>",
          "row_count": 1
        },
        {
          "path": "artifacts/data/grpo_all.jsonl",
          "sha256": "<lowercase-sha256>",
          "row_count": 1
        }
      ]
    }
  ],
  "coverage": {
    "paper_ids": true,
    "source_documents": true,
    "creator_groups": true,
    "image_bytes": true,
    "prompts": true
  },
  "paper_ids": ["doi:10.5555/actual-training-paper"],
  "source_document_ids": ["source:actual-training-document"],
  "creator_group_ids": ["creator-group:actual-training-origin"],
  "image_sha256s": ["<lowercase-sha256>"],
  "prompt_sha256s": ["<lowercase-sha256>"]
}
```

`row_count` и все множества в примере являются placeholders. Manifest принимается только при
полных непустых множествах. `paper_ids` и `prompt_sha256s` дополнительно сверяются с реальными
training JSONL. Source, creator и image completeness остается явной release-owner attestation и не
может быть восстановлена из filenames.

Prompt hash вычисляется после Unicode NFKC, `casefold`, замены последовательностей whitespace одним
space и удаления крайних spaces, затем SHA256 от UTF-8. Audit v3 включает как отдельные text blocks,
так и объединенные варианты с whitespace и прямой конкатенацией, включая fragments из нескольких
user messages и prompt-like fields. Для защиты от смешанных fragment boundaries audit также
использует compact representation без whitespace. Manifest должен покрыть все эти представления для
training inputs, которые реально входили в обучение.

Локальная schema-проверка:

```powershell
python -c "import json,sys; from jsonschema import Draft202012Validator; schema=json.load(open(sys.argv[1],encoding='utf-8')); value=json.load(open(sys.argv[2],encoding='utf-8')); Draft202012Validator(schema,format_checker=Draft202012Validator.FORMAT_CHECKER).validate(value); print('lineage schema: OK')" "experiments/vlm_ab_evaluation/schemas/training_lineage_manifest.schema.json" "<LOCAL_LINEAGE_MANIFEST>"
```

Затем manifest публикуется в том же adapter repository отдельным commit. `parent_commit=R`
предотвращает race: upload завершится ошибкой, если mutable branch уже ушел от проверенного `R`.

```powershell
$LocalLineage = "<LOCAL_LINEAGE_MANIFEST>"
$M = python -c "import sys; from huggingface_hub import HfApi; result=HfApi().upload_file(repo_id=sys.argv[1],repo_type='model',path_or_fileobj=sys.argv[2],path_in_repo='artifacts/training_lineage_manifest.json',parent_commit=sys.argv[3],commit_message='Publish immutable training-lineage attestation for R'); print(result.oid)" $AdapterRepo $LocalLineage $R
if ($LASTEXITCODE -ne 0 -or $M -notmatch '^[0-9a-f]{40}$' -or $M -eq $R) {
    throw "Lineage upload failed or did not create a distinct immutable revision M"
}
$M
```

Сохраните новый SHA как `M`. Должны выполняться условия `M != R`, manifest находится в том же model
repository, `training_sources[].revision == R`, а inference остается pinned к `R`.

## 12. Создание нового strict config

Создайте новый config, не изменяя смысл старого audited run. Обновите как минимум:

```yaml
experiment:
  id: qwen3vl-8b-scireason-task3-ab-v2
  public_id: study-2026-07-v2
  require_clean_code: true
  require_preregistered_plan: true
  output_dir: runs/vlm_ab/qwen3vl-8b-scireason-task3-ab-v2

benchmark:
  repo_id: <BENCHMARK_REPO>
  revision: <B>
  data_file: data/task3_vlm_generation.jsonl
  provenance_file: article_image_sources.jsonl
  require_gold: false
  require_complete_provenance: true
  require_split_provenance: true

training_audit:
  require_lineage_manifest: true
  lineage_manifest:
    repo_id: top-papers/Qwen3-VL-8B-Instruct-scireason
    repo_type: model
    revision: <M>
    file: artifacts/training_lineage_manifest.json
  sources:
    - repo_id: top-papers/Qwen3-VL-8B-Instruct-scireason
      repo_type: model
      revision: <R>
      files:
        - artifacts/data/sft_all.jsonl
        - artifacts/data/grpo_all.jsonl

models:
  tuned:
    adapter:
      id: top-papers/Qwen3-VL-8B-Instruct-scireason
      revision: <R>
      adapter_name: default

processor:
  id: top-papers/Qwen3-VL-8B-Instruct-scireason
  revision: <R>
```

Скопируйте неизмененные base model, generation, conditions, statistics и power sections из
publication config. Замените reviewer placeholders на заранее назначенные pseudonymous IDs.

Если заявляются automatic semantic metrics, сначала задайте `benchmark.require_gold=true` и
повторите assembly с обязательными gold/rubric. Нельзя включать automatic claim после просмотра
model outputs.

Каждый новый protocol получает новые `experiment.id`, `public_id` и `output_dir`. Если config,
source tree, power assumptions или inputs меняются после `plan`, создается еще один run ID.

## 13. Freeze кода и preregistration

До strict run должны пройти tests, а Git tree должен быть clean:

```powershell
python -m pytest `
  tests/test_vlm_ab_curator.py `
  tests/test_vlm_ab_audit.py `
  tests/test_vlm_ab_blind.py `
  tests/test_vlm_ab_inference.py `
  tests/test_vlm_ab_pipeline.py `
  tests/test_vlm_ab_remediation.py `
  tests/test_vlm_ab_stats.py -q

python -m ruff check src/scireason/vlm_ab `
  tests/test_vlm_ab_curator.py `
  tests/test_vlm_ab_audit.py `
  tests/test_vlm_ab_blind.py `
  tests/test_vlm_ab_inference.py `
  tests/test_vlm_ab_pipeline.py `
  tests/test_vlm_ab_remediation.py `
  tests/test_vlm_ab_stats.py

git status --short
git rev-parse HEAD
```

`git status --short` должен вернуть пустой output. Зафиксируйте code/config commit обычным reviewed
commit. После этого запустите `plan`, используя новый strict config:

```powershell
$StrictConfig = "experiments/vlm_ab_evaluation/configs/<NEW_STRICT_CONFIG>.yaml"

python experiments/vlm_ab_evaluation/run_pipeline.py `
  --config $StrictConfig `
  plan
```

Сохраните `design/power_plan.json` во внешнем preregistration/archive до просмотра model outputs.
Текущий design ожидает 240 independent primary papers, evaluable fraction 0.90 и achieved power
около 0.836 для target effect 0.10.

После `plan` нельзя редактировать evaluation source, config или plan. Любое изменение требует нового
ID и нового plan.

## 14. Fresh strict `prepare`

```powershell
python experiments/vlm_ab_evaluation/run_pipeline.py `
  --config $StrictConfig `
  prepare
```

Не добавляйте `--exploratory`, `--benchmark-dir` или local `--training-file`. Strict prepare должен
скачать и проверить exact `B`, `R` и `M`. В strict mode lineage обязателен независимо от значения
`training_audit.require_lineage_manifest`; `false` не является waiver.

Acceptance gate:

- `prepare_manifest.publication_ready=true`;
- `prepare_manifest.result_scope="publication_ready"`;
- `prepare_manifest.code_gate_passed=true`;
- каждый configured revision равен resolved revision;
- audit status равен `pass`;
- critical findings равны нулю;
- каждый unique sample eligible;
- training overlap по paper/source/creator/image/prompt равен нулю;
- lineage issues отсутствуют;
- lineage manifest проходит canonical JSON Schema;
- bundled `adapter_config.json` фиксирует configured base model ID и revision;
- provenance image order и обязательные citations проходят проверку;
- duplicate normalized prompts и within-row duplicate bytes отсутствуют;
- primary papers не меньше preregistered minimum.

Если любой пункт не выполнен, inference не запускается. Исправление публикуется новой immutable
revision и новым experiment ID; failed run artifacts не редактируются вручную.

## 15. Inference в двух процессах

Base и tuned запускаются последовательно, каждый в отдельном process:

```powershell
python experiments/vlm_ab_evaluation/run_pipeline.py `
  --config $StrictConfig `
  infer --arm base --backend transformers

python experiments/vlm_ab_evaluation/run_pipeline.py `
  --config $StrictConfig `
  infer --arm tuned --backend transformers
```

Дождитесь полного завершения base process перед tuned process. В strict mode запрещены `--arm all`,
`--backend mock`, `--limit` и `--exploratory`.

Ожидаемые outputs:

```text
predictions/base.jsonl
predictions/base.jsonl.manifest.json
predictions/tuned.jsonl
predictions/tuned.jsonl.manifest.json
```

Не редактируйте prediction JSONL и sidecars. Resume допустим только при полном совпадении run,
protocol, input, config и code fingerprints.

## 16. Blinding и human review

После завершения обеих arms:

```powershell
python experiments/vlm_ab_evaluation/run_pipeline.py `
  --config $StrictConfig `
  blind
```

Каждому reviewer передается только его директория:

```text
blind_review/public/<opaque-reviewer-id>/
```

Нельзя передавать reviewer весь run, `blind_review/owner_only/`, owner mapping,
`blind_review_manifest.json`, prediction filenames или randomization secret. Reviewer открывает
`review.html`, заполняет все assignments и экспортирует JSON. Оригинальные exports складываются с
уникальными именами в новый `incoming_reviews/`.

Deblinding не проводится до получения всех назначенных reviews. Missing reviews блокируют strict
publication result; `--allow-incomplete` разрешен только для exploratory diagnostics.

## 17. Aggregation

```powershell
python experiments/vlm_ab_evaluation/run_pipeline.py `
  --config $StrictConfig `
  aggregate --reviews-dir "runs/vlm_ab/qwen3vl-8b-scireason-task3-ab-v2/incoming_reviews"
```

Ключевые outputs:

```text
analysis/results.json
analysis/report.md
analysis/tables/paper_scores.csv
analysis/figures/preference_effects.svg
analysis/deblinded_reviews.jsonl
```

Заявление о превосходстве разрешено только если `publication_artifacts_ready=true` и
`superiority_claim_supported=true`. Реализация дополнительно требует `theta > 0.5`, lower 95% CI
выше 0.5, two-sided `p < 0.05` и прохождение worst-case missingness gate. Статистически
незначимый результат не является доказательством эквивалентности.

## 18. Публикационный архив

Приватный immutable archive должен включать:

- exact config и config SHA256;
- code commit и environment/package inventory;
- power plan и preregistration timestamp;
- prepare manifest, audit JSON/Markdown и frozen inputs;
- prediction JSONL и sidecars обеих arms;
- public reviewer packages и отдельно owner-only mapping/secret;
- оригинальные review exports;
- results, report, paper table, figure и protocol deviations;
- benchmark revision `B`, adapter revision `R` и lineage attestation revision `M`.

В public supplement нельзя включать owner secret, arm mapping до завершения study или персональные
данные reviewers. `analysis/deblinded_reviews.jsonl` публикуется только после de-identification,
проверки consent и license.

## 19. Частые hard stops

| Ошибка | Что означает | Исправление |
| --- | --- | --- |
| `decision ... is still pending` | Не все human decisions завершены | Завершить указанную row, не обходить gate |
| `tampered ... binding` | Изменено поле связи с immutable queue | Восстановить binding из template |
| `requires two distinct ... identifiers` | Нет двух независимых normalized IDs | Провести вторую реальную проверку |
| `declared image SHA256 does not match bytes` | Provenance не соответствует файлу | Проверить правильный файл/path и пересчитать SHA256 |
| `corrected benchmark audit has critical findings` | Candidate все еще нарушает contract | Исправить curator record или исключить его |
| `fewer unique primary paper IDs` | Design меньше preregistered N | Добавить проверенные независимые papers или пересоздать protocol до inference |
| `training lineage sources differ` | Manifest не совпадает с config/files at `R` | Пересобрать manifest и опубликовать новый `M` |
| `distinct immutable attestation revision` | `M` ошибочно равен training source revision `R` | Опубликовать manifest отдельным commit |
| `publication inference requires a clean ... tree` | Git dirty или code snapshot изменен | Reviewed commit, clean tree, новый plan/run при изменении protocol |
| `power plan already exists for another protocol` | Output directory уже связан с другим config | Новый experiment ID и output directory |

## 20. Фактический следующий шаг

Сейчас нужно сгенерировать `curate-forms` workspace, открыть `curator.html`, распределить 386 задач
между curators и экспортировать полный `completed_decisions.jsonl`. До завершения human curation,
успешного `curate-assemble` и публикации `B`, `R`, `M` strict inference запускать нельзя.
