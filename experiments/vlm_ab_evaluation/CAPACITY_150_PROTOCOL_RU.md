<!-- SPDX-FileCopyrightText: 2026 top-papers-graph contributors -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# Capacity-Limited Protocol: N=150

Это отдельный protocol для capacity-limited confirmatory study. Он не делает текущий архивный
benchmark пригодным для публикации и не разрешает запускать inference до исправления данных.

Процесс намеренно разделен на две фазы. Phase 0 начинает работу с архивного blocked audit и создает
только candidate для human remediation. Phase 1 начинается после публикации исправленных immutable
releases и создает единственный strict publication run. Поэтому для начала Phase 0 не нужны заранее
исправленные `B`, `R` и `M`: их результатом не должен быть plan или model output.

## Power settings

Оба capacity config используют ровно следующие settings:

| Parameter | Value |
| --- | --- |
| Independent primary papers | `n_items=150` |
| Exact-N guard | `require_exact_n_items=true` |
| Reviews per paper | `reviews_per_item=2` |
| Expert roster | ровно два эксперта, каждый оценивает все papers |
| Evaluable fraction | `0.90` |
| ICC | `0.50` |
| Alpha | `0.05` |
| Target power | `0.80` |
| Score SD | `0.50` |
| Target effect | `0.121` |

При этих assumptions `power_mde_plan` дает effective sample size `135`, MDE около `0.12056` и
achieved power около `0.80284`. Effect `0.10` при `N=150` недостаточен. Два reviewer не создают
дополнительных независимых papers.

## Phase 0: remediation candidate

### 0.1 Create the working config

Запускайте команды из корня `top-papers-graph` в PowerShell. Новый generator принимает архивный
remediation audit config, проверяет его через `load_experiment_config`, сохраняет benchmark/model/training
references и все protocol settings. Он меняет только experiment identities, устанавливает
`require_clean_code=false`, `require_preregistered_plan=false` и exact-N capacity power settings.

```powershell
$ArchivedRemediationConfig = "experiments/vlm_ab_evaluation/configs/qwen3vl_scireason_remediation_audit_v2.yaml"
$CapacityRemediationConfig = "experiments/vlm_ab_evaluation/configs/qwen3vl-cap150-remediation-working-v1.yaml"
$CapacityRemediationRun = "runs/vlm_ab/qwen3vl-cap150-remediation-working-v1"

python experiments/vlm_ab_evaluation/make_capacity150_remediation_config.py `
  --input $ArchivedRemediationConfig `
  --output $CapacityRemediationConfig `
  --experiment-id "qwen3vl-cap150-remediation-working-v1" `
  --public-id "study-2026-cap150-remediation-working-v1" `
  --output-dir $CapacityRemediationRun
```

IDs должны содержать `cap150` и `remediation`. Output YAML пишется write-once. Output directory
должен быть новым безопасным путем внутри `runs/`; existing path и symlinked ancestor отклоняются.
Generator не вызывает `plan` или `prepare`, не создает run directory и печатает JSON preview с
`scope=capacity-remediation-working-config`. В preview нет power plan и нет заявления о publication
readiness.

Этот config допускает только fresh `prepare --exploratory`, `curate-queue`,
`curate-capacity-plan`, `curate-forms`, `curate-triage` и `curate-assemble`. CLI отклоняет `plan`,
`infer`, `blind`, `aggregate` и `run` для такого config.

### 0.2 Fresh exploratory prepare

```powershell
python experiments/vlm_ab_evaluation/run_pipeline.py `
  --config $CapacityRemediationConfig `
  prepare --exploratory
```

Используется новый `$CapacityRemediationRun`, а не historical run или queue. Архивные blockers могут
привести к exit после записи `prepare_manifest.json` и audit, когда не осталось technically runnable
rows. Это remediation evidence, а не strict publication pass; сохраненный manifest все равно является
входом для queue.

### 0.3 Queue, forms, triage, and evidence

```powershell
$CapacityPrepareManifest = Join-Path $CapacityRemediationRun "prepare_manifest.json"
$CapacityQueue = Join-Path $CapacityRemediationRun "curation_queue_v3"
$CapacityReviewPlan = Join-Path $CapacityRemediationRun "capacity_review_plan_v4"
$CapacityMachineAssist = Join-Path $CapacityRemediationRun "capacity_machine_assist_v4"
$CapacityMachineInput = Join-Path $CapacityRemediationRun "machine_enrichment_v2.jsonl"
$CapacityMachineOutput = Join-Path $CapacityRemediationRun "capacity_machine_enrichment_v2"
$CapacityForms = Join-Path $CapacityRemediationRun "curator_workspace_v4"
$CapacityTriage = Join-Path $CapacityRemediationRun "assisted_triage_v2"
$CapacityDecisions = Join-Path $CapacityRemediationRun "completed_decisions.jsonl"
$CapacityCuratedDataset = Join-Path $CapacityRemediationRun "curated_dataset"
$CapacityCandidate = Join-Path $CapacityRemediationRun "release_candidate"

python experiments/vlm_ab_evaluation/run_pipeline.py `
  --config $CapacityRemediationConfig `
  curate-queue `
  --prepare-manifest $CapacityPrepareManifest `
  --output-dir $CapacityQueue

python experiments/vlm_ab_evaluation/run_pipeline.py `
  --config $CapacityRemediationConfig `
  curate-forms `
  --prepare-manifest $CapacityPrepareManifest `
  --queue-manifest (Join-Path $CapacityQueue "queue_manifest.json") `
  --output-dir $CapacityForms

python experiments/vlm_ab_evaluation/run_pipeline.py `
  --config $CapacityRemediationConfig `
  curate-capacity-plan `
  --prepare-manifest $CapacityPrepareManifest `
  --queue-manifest (Join-Path $CapacityQueue "queue_manifest.json") `
  --output-dir $CapacityReviewPlan

python experiments/vlm_ab_evaluation/run_pipeline.py `
  --config $CapacityRemediationConfig `
  curate-capacity-assist `
  --prepare-manifest $CapacityPrepareManifest `
  --queue-manifest (Join-Path $CapacityQueue "queue_manifest.json") `
  --capacity-plan-manifest (Join-Path $CapacityReviewPlan "capacity_plan_manifest.json") `
  --output-dir $CapacityMachineAssist

python experiments/vlm_ab_evaluation/run_pipeline.py `
  --config $CapacityRemediationConfig `
  curate-triage `
  --prepare-manifest $CapacityPrepareManifest `
  --queue-manifest (Join-Path $CapacityQueue "queue_manifest.json") `
  --output-dir $CapacityTriage `
  --exclude-training-overlap

```

Forms and triage are bound to the fresh queue. Triage `exclude` values are only proposals: two
independent human reviewers must confirm every final retain/exclude decision, and artifact v3 requires
the exported field `independent_attestation=true`.

`curate-capacity-assist` требует ровно 150 non-overlap clean или identity-remediation groups и атомарно
создает отдельный
write-once пакет: 150 queue-bound dossiers, 150 пустых enrichment templates, human-review checklist и
manifest с SHA256. Dossiers содержат только immutable bindings, source hints и audit evidence. Legacy
prompt text не переносится, image hints остаются unverified, cross-paper reused hashes quarantined,
duplicate row не выбирается, а human
IDs и attestations остаются пустыми. Затем отдельный acquisition/enrichment process заполняет новые
benchmark/provenance proposals и прикладывает внешние evidence; до этого каждый template имеет
`release_eligibility="blocked"`.

Не редактируйте templates внутри `$CapacityMachineAssist`: они входят в hashed write-once package.
Acquisition process создает отдельный `$CapacityMachineInput` с теми же 150 group bindings. Для каждой
группы он выбирает task IDs, предлагает полные benchmark/provenance rows, прикладывает evidence records
с URL, content SHA256 и отдельным `asserted_license` для license evidence, затем связывает все восемь
claims с evidence IDs. Поля `verified_by`, curator IDs и attestation остаются пустыми, а
`release_eligibility` остается `blocked`.

Проверка и публикация заполненного machine package:

```powershell
python experiments/vlm_ab_evaluation/run_pipeline.py `
  --config $CapacityRemediationConfig `
  curate-capacity-enrichment `
  --prepare-manifest $CapacityPrepareManifest `
  --queue-manifest (Join-Path $CapacityQueue "queue_manifest.json") `
  --capacity-plan-manifest (Join-Path $CapacityReviewPlan "capacity_plan_manifest.json") `
  --capacity-assist-manifest (Join-Path $CapacityMachineAssist "capacity_assist_manifest.json") `
  --machine-enrichment-jsonl $CapacityMachineInput `
  --output-dir $CapacityMachineOutput
```

Команда byte-for-byte реконструирует queue-bound plan/assist, проверяет exact 150 groups, JSON schemas,
совместимость evidence kind с claim, URL без query/fragment/userinfo и cross-paper image quarantine.
Она не скачивает и не подтверждает научное содержание внешних evidence: это остается human gate.
Результат содержит новый manifest и `machine_assisted_draft.json`. Draft
заполняет научные поля, но не выбирает `retain`, не подставляет human IDs и не ставит attestation; его
можно импортировать через **Merge draft** только для последующей короткой проверки людьми.

### 0.4 Start artifact-v3 manual review

Open only `$CapacityForms/curator.html` from `curator_workspace_v4`. This workspace has the simplified
expert form with field-level instructions; `curator_workspace_v3` is superseded but its drafts remain
mergeable only when the form accepts their exact queue fingerprint. Old v1/v2 drafts have another queue
fingerprint and cannot be recovered into this run. Browser `file://` storage is not portable. Use
**Merge draft** only for queue-v3 machine, triage, or human subset drafts that pass the fingerprint and
exact-field checks.

Use `capacity_review_plan_v4/review_assignment_plan.csv` to distribute work. Experts verify all 150
proposals, including two identity-remediation groups, and manually confirm 74 exclusions with real IDs
and `independent_attestation=true`. **Сбросить неполные retain** only clears locally invalid edits in the
current workspace; no historical count is authoritative. Neither plan, merge, nor recovery creates a
final decision, reviewer identity, or attestation.

`curate-capacity-plan` создает отдельный read-only пакет планирования: `review_assignment_plan.csv`,
`capacity_paper_groups.jsonl`, `capacity_plan_summary.md` и `capacity_plan_manifest.json`. Он повторно
проверяет prepare/queue, не меняет queue, forms или drafts и использует только нейтральные
`reviewer-slot-1/2/3`, а не реальные reviewer ID. Он не создает decisions или attestations, не выбирает
строку из duplicate group, не remap-ит paper ID и не подтверждает факты о paper. Группа с любым
`training_paper_overlap` закрыта для capacity; group с unresolved identity не входит в clean count.
Primary membership вычисляется из configured `statistics.primary_strata`, а не legacy
`primary_endpoint`. Manifest отдельно фиксирует clean count и clean+identity-remediation pool. Exact pool
разрешает только blocked machine proposals; `capacity_exact_target_available` остается false, пока clean
count не достигнет `N=150` после human verification и corrected assembly.

### 0.5 Assemble and archive evidence

Только после публикации machine package и завершения independent human decisions запускайте assembly:

```powershell
python experiments/vlm_ab_evaluation/run_pipeline.py `
  --config $CapacityRemediationConfig `
  curate-assemble `
  --prepare-manifest $CapacityPrepareManifest `
  --queue-manifest (Join-Path $CapacityQueue "queue_manifest.json") `
  --decisions-jsonl $CapacityDecisions `
  --curated-dataset-root $CapacityCuratedDataset `
  --capacity-enrichment-manifest (Join-Path $CapacityMachineOutput "machine_enrichment_manifest.json") `
  --machine-enrichment-jsonl $CapacityMachineInput `
  --output-dir $CapacityCandidate
```

Capacity assembly требует оба evidence inputs, заново детерминированно реконструирует plan, assist и
machine package, но независимо валидирует human decisions. Исходный machine JSONL и полный package
архивируются byte-for-byte под `audit/machine_enrichment/`; их hashes входят в `assembly_manifest.json`
и `output_files`. Exact queue, blank decision template и completed decisions одновременно архивируются
под `audit/human_review/`; assembly manifest связывает их hashes, reviewer IDs и обязательный affirmative
attestation. Это делает review inputs воспроизводимыми, но по-прежнему не доказывает криптографически
реальную личность или независимость экспертов. Архивирование также не превращает declared external
content hashes в проверенные bytes:
`external_evidence_content_verified=false` сохраняется, а human gate остается обязательным. Assembly
также проверяет schemas, canonical identities, staged image bytes, contamination и frozen warnings и
отклоняет как меньше, так и больше 150 unique primary papers.

The result is only `validated_benchmark_release_candidate` with `publication_ready=false`. It cannot be
used for plan, inference, blind review, aggregation, or a publication claim.

## Phase 1: strict corrected study

### 1.1 Publish releases in order

After Phase 0 assembly succeeds, publish the corrected candidate as immutable benchmark revision `B`.
Then publish the evaluated adapter and exact training exports as immutable revision `R`. Publish a later
immutable lineage-attestation revision `M` in the same model repository; `M` declares the files and
hashes at `R`, so `M != R`. Create a corrected strict config that pins `B`, `R`, and `M`, includes the
lineage manifest, and contains the same exactly 150 verified unique nonoverlap primary papers. The
benchmark section must also pin Phase 0's published `assembly_manifest.json` by relative path and
lowercase SHA256 via `assembly_manifest_file` and `assembly_manifest_sha256`.

The strict config must require clean code, a preregistered plan, and exactly two pseudonymous reviewer
IDs with two reviews per item in both review and power sections. It is a new config, not a mutation of
the Phase 0 working config or its run directory. It must also contain the exact selected primary
runtime contract in both arms: unquantized `torch_dtype=float16`, `device_map=balanced`, SDPA,
`low_cpu_mem_usage=true`, `trust_remote_code=false`, plus top-level tuned
`adapter_kwargs.autocast_adapter_dtype=true` and `experiment.precision_mode=fp16-primary`. The final
output additionally requires `power.n_items=150`, `require_exact_n_items=true` and
`reviews_per_item=2`. The generator rejects rather than silently rewrites a corrected source with a
different precision contract.

Strict `prepare` downloads that exact manifest from `B`, verifies its archived review/queue bindings,
exact-N counts, machine-evidence policy, and every declared release file hash and size. It preserves the
complete declared inventory at the same relative paths in the relocatable prepare bundle and repeats the
assembly validation against those copied bytes. Omitting, re-sealing, or replacing the binding is a hard
publication-gate failure.

### 1.2 Generate, commit, plan, and prepare

```powershell
$CorrectedStrict = "experiments/vlm_ab_evaluation/configs/qwen3vl_corrected_strict.yaml"
$CapacityConfig = "experiments/vlm_ab_evaluation/configs/qwen3vl-cap150-capacity-v1.yaml"
$CapacityRun = "runs/vlm_ab/qwen3vl-cap150-capacity-v1"

python experiments/vlm_ab_evaluation/make_capacity150_config.py `
  --input $CorrectedStrict `
  --output $CapacityConfig `
  --experiment-id "qwen3vl-cap150-capacity-v1" `
  --public-id "study-2026-cap150-capacity-v1" `
  --output-dir $CapacityRun

python -m pytest `
  tests/test_vlm_ab_pipeline.py `
  tests/test_vlm_ab_remediation.py `
  tests/test_vlm_ab_capacity150.py `
  tests/test_vlm_ab_capacity150_remediation.py `
  tests/test_vlm_ab_capacity_plan.py `
  tests/test_vlm_ab_capacity_assist.py `
  tests/test_vlm_ab_capacity_enrichment.py `
  tests/test_vlm_ab_kaggle.py `
  tests/test_vlm_ab_stats.py -q

python -m ruff check src/scireason/vlm_ab `
  tests/test_vlm_ab_pipeline.py `
  tests/test_vlm_ab_remediation.py `
  tests/test_vlm_ab_capacity150.py `
  tests/test_vlm_ab_capacity150_remediation.py `
  tests/test_vlm_ab_capacity_plan.py `
  tests/test_vlm_ab_capacity_assist.py `
  tests/test_vlm_ab_capacity_enrichment.py `
  tests/test_vlm_ab_kaggle.py `
  tests/test_vlm_ab_stats.py

git add $CapacityConfig
git commit -m "Add capacity-limited N=150 protocol"
git status --short
git rev-parse HEAD

python experiments/vlm_ab_evaluation/run_pipeline.py `
  --config $CapacityConfig `
  plan

python experiments/vlm_ab_evaluation/run_pipeline.py `
  --config $CapacityConfig `
  prepare
```

`git status --short` must print no lines before `plan`. The clean plan gate also requires a valid
40-character Git `HEAD` before writing or reusing `design/power_plan.json`. That plan is the only
preregistration. Do not use `--exploratory`, local benchmark overrides, or local training overrides in
the final prepare.

Only after strict prepare succeeds may `$CapacityConfig` and `$CapacityRun` be used for separate strict
base/tuned inference, blinded review, and aggregation. Any change to B/R/M, config, source, or power
assumptions requires new IDs, a fresh output directory, a clean commit, a new plan, and a new prepare.
