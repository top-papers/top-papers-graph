<!-- SPDX-FileCopyrightText: 2026 top-papers-graph contributors -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# Publication A/B evaluation: Qwen3-VL base vs SciReason LoRA

This directory contains an end-to-end, resumable paired evaluation of:

- A: `Qwen/Qwen3-VL-8B-Instruct`;
- B: the same base model with
  `top-papers/Qwen3-VL-8B-Instruct-scireason` explicitly loaded as a PEFT adapter.

The pipeline freezes all inputs, varies only the adapter, produces blinded reviewer packages,
and reports paper-clustered inference. It never treats JSON validity as semantic quality.

## Important current status

The pinned public benchmark revision
`33ccc5ed08e314c6457dcaa23e7f7508406cb4f8` does **not** pass the publication gate. The authenticated
audit found 386 rows, 360 unique sample IDs, 1,544 image references but only 30 unique image
contents, 1,546 critical findings, 938 warnings, and zero eligible samples. It also found 22
training-overlap paper IDs affecting 70 unique samples (71 rows), plus no complete immutable
training-lineage manifest. Therefore the default `prepare` command is expected to stop before model
inference and write a diagnostic report.

Use `--exploratory` only to test infrastructure. Every downstream exploratory report is marked
`NOT FOR PUBLICATION`; this flag is not a way to waive the scientific validity gate.

The verified hashes, issue counts, correction ownership, and release acceptance criteria are in
`remote_audit_baseline_20260717.json` and `REMEDIATION.md`.

The detailed Russian operational tutorial from human curation through strict inference and
publication is `NEXT_STEPS_RU.md`.

The reviewer-facing handoff and scoring rules are in
[`EXPERT_REVIEW_RU.md`](EXPERT_REVIEW_RU.md). The same versioned rubric is embedded directly in every
offline `review.html` package.

The concise current readiness, immutable remediation hashes, blockers, and required owner decisions
are in [`PROJECT_STATUS_RU.md`](PROJECT_STATUS_RU.md).

The revised lower-capacity protocol is documented in
[`CAPACITY_150_PROTOCOL_RU.md`](CAPACITY_150_PROTOCOL_RU.md). Phase 0 creates a separate
exploratory remediation config from the archived failed audit so curators can assemble an exact-N=150
candidate. Phase 1 starts only after corrected immutable benchmark `B`, adapter/training release `R`,
and lineage attestation `M` exist; it creates the final strict capacity config and preregisters it.
The selected confirmatory endpoint is an unquantized FP16 base/compute runtime with the native FP32
PEFT LoRA, exact `N=150`, balanced placement on two T4 GPUs, and exactly two reviewers.
The config records `experiment.precision_mode=fp16-primary`; runtime verifies model placement and
dtypes and rejects incomplete or unexpected PEFT checkpoint keys.

The private Kaggle API workflow for two T4 GPUs is documented in
[`kaggle/README_RU.md`](kaggle/README_RU.md). It supports the FP16 confirmatory primary run and a
separate NF4 sensitivity run. Each mode has exact config/runtime checks, private mode-bound state, a
fresh plan, and a fresh strict prepare; neither may reuse the current blocked run or prepared bundle.

The final capacity-limited strict config is the FP16 primary config. The NF4 config is derived from it
with `kaggle/make_nf4_config.py` and needs its own IDs, output directory, clean commit, immutable plan,
and strict prepare. NF4 receives automatic diagnostics only, not a repeated human review.
Its inference artifacts use `result_scope=automatic_sensitivity_only`, and human-review commands are
rejected in code.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[vlm_ab,dev]"
```

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[vlm_ab,dev]"
```

For repeatable Hub downloads, authenticate before `prepare` (the public repositories do not
require special access, but anonymous IPs can be rate-limited):

```bash
read -rsp "Hugging Face token: " HF_TOKEN
export HF_TOKEN
printf '\n'
```

PowerShell without placing the token in command history:

```powershell
$SecureToken = Read-Host "Hugging Face token" -AsSecureString
$env:HF_TOKEN = [System.Net.NetworkCredential]::new("", $SecureToken).Password
Remove-Variable SecureToken
```

One A100/H100-class GPU is sufficient for sequential 8B inference. The supplied DataSphere job
uses `g2.2` and launches the arms in separate processes to guarantee model-memory release.

## Capacity-limited pipeline

The workflow has two non-interchangeable phases. The checked-in archived config is a blocked audit
input, not a publication run. Phase 0 turns that input into a fresh remediation workspace and never
plans or evaluates models. Phase 1 starts after publication of corrected data and creates the strict
capacity run.

### Phase 0. Generate a capacity remediation config

Run these commands from the repository root in PowerShell. The generator reads the archived remediation
audit config, preserves its benchmark/model/training references and protocol settings, changes only the
experiment identities and clean/preregistration flags, and installs the exact-N=150 capacity power
settings. It writes YAML once, refuses an existing or symlinked run path, and does not create a plan or a
run directory.

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

Its JSON preview has scope `capacity-remediation-working-config`, is explicitly not publication-ready,
and contains no power plan. This config may run only `prepare --exploratory` and the curation commands,
including `curate-capacity-plan`; `plan`, `infer`, `blind`, and `aggregate` are blocked.

### Phase 0. Fresh exploratory audit and freeze

Use the remediation variables from the preceding block. This is a new run and does not reuse the
archived run directory or queue.

```powershell
python experiments/vlm_ab_evaluation/run_pipeline.py `
  --config $CapacityRemediationConfig `
  prepare --exploratory
```

The command writes a fresh `prepare_manifest.json` and audit under `$CapacityRemediationRun`. The
archived faults can make it exit after writing those diagnostics because no technically runnable rows
remain; that is expected remediation evidence, not a publication gate pass. The manifest is the input to
the fresh queue below.

It pins and verifies the archived inputs, hashes images, and checks paper-ID and prompt overlap against
the configured training files. It writes:

- `audit/benchmark_audit.json` and `.md`;
- `inputs/frozen_benchmark.jsonl`;
- `inputs/dataset/` and `inputs/source/`, a relocatable copy of every frozen image and audited
  source JSONL;
- for a pinned assembly, the complete declared release inventory under `inputs/dataset/` at its
  original relative paths, including review and machine-evidence archives;
- `prepare_manifest.json` with file hashes, source-tree fingerprint, Git state, and resolved revisions.

### Phase 0. Curate an exact-N=150 candidate

Create every curation artifact under the fresh remediation run. The queue binds the generated working
config, its fresh prepare manifest, and the current audit contract. It does not repair rows or choose
duplicate/image mappings automatically. Every source row needs an explicit retain/exclude decision and
two independent reviewer IDs plus `independent_attestation=true`; retained rows need complete replacement
benchmark and provenance objects. Remediation artifact v3 exports and validates that attestation on the
server rather than treating the browser checkbox as sufficient.

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
```

The queue contains immutable `tasks.jsonl`, blank `decision_template.jsonl`, and a hash-bound
`queue_manifest.json`. Curator tooling must emit a separate completed decisions JSONL while leaving the
queue directory unchanged. Patches and inferred defaults are rejected. Every blank template contains
`independent_attestation=false`; final export and `curate-assemble` both require literal `true`.

### Phase 0. Plan manual capacity review

Create this standalone package from the same verified queue before distributing work. It is a routing and
capacity-counting aid, not a curation result:

```powershell
python experiments/vlm_ab_evaluation/run_pipeline.py `
  --config $CapacityRemediationConfig `
  curate-capacity-plan `
  --prepare-manifest $CapacityPrepareManifest `
  --queue-manifest (Join-Path $CapacityQueue "queue_manifest.json") `
  --output-dir $CapacityReviewPlan
```

The command atomically writes `review_assignment_plan.csv`, `capacity_paper_groups.jsonl`,
`capacity_plan_summary.md`, and `capacity_plan_manifest.json` outside the queue. CSV assignments rotate
two distinct neutral slots across `reviewer-slot-1/2/3`; they are not reviewer IDs and no identity,
attestation, or decision is prefilled. The package does not change queue, forms, or drafts; it does not
retain a row, choose a duplicate row, remap a paper, or claim factual paper verification.

`training_paper_overlap` in any task closes its canonical group. A group with unresolved identity is not
counted. Primary membership is derived from configured `statistics.primary_strata`, never from the
untrusted legacy `primary_endpoint` boolean. `capacity_plan_manifest.json` sets
`capacity_exact_target_available=true` only when its clean capacity candidate group count is exactly
`N=150`. It separately reports the clean+identity-remediation pool; an exact pool permits only blocked
machine proposals and does not increase the clean count before human verification and corrected assembly.

### Phase 0. Build machine-assistance dossiers

After the capacity plan reports an exact pool of 150 non-overlap clean or identity-remediation groups,
build a deterministic acquisition and enrichment handoff package:

```powershell
python experiments/vlm_ab_evaluation/run_pipeline.py `
  --config $CapacityRemediationConfig `
  curate-capacity-assist `
  --prepare-manifest $CapacityPrepareManifest `
  --queue-manifest (Join-Path $CapacityQueue "queue_manifest.json") `
  --capacity-plan-manifest (Join-Path $CapacityReviewPlan "capacity_plan_manifest.json") `
  --output-dir $CapacityMachineAssist
```

The command atomically publishes 150 queue-bound candidate dossiers, 150 blocked enrichment templates,
a short human-review checklist, and a SHA256 manifest. It includes immutable bindings, source hints, and
audit evidence, but omits legacy prompt text, marks every legacy image hint unverified, and permanently
quarantines hashes implicated in cross-paper reuse. It
does not select duplicate rows, create retain decisions, fill human IDs, or claim that paper identity,
provenance, licenses, or training holdout have been verified. External acquisition must supply new
evidence before curators can perform the final short verification.

Do not edit the templates inside `$CapacityMachineAssist`; they belong to the hashed write-once package.
The acquisition process writes a separate `$CapacityMachineInput` with all 150 bindings, selected task
IDs, benchmark/provenance proposals, and URL/content-hash evidence. Every `license` evidence record must
also carry the exact nonblank `asserted_license` used by the proposed provenance row; non-license
records must set it to `null`. Human verifier fields and attestations must remain empty. Validate and
publish that separate input with:

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

This command reconstructs the queue-bound plan and assist packages byte-for-byte, then validates all 150
proposals, claim-compatible evidence kinds, credential-free URL bindings, schemas, and the cross-paper
legacy-image quarantine. It does not retrieve or scientifically verify external evidence content; that
remains part of human verification. It emits a new hashed package containing
`machine_assisted_draft.json`. That draft prefills scientific fields but leaves disposition, reviewer IDs,
image verifiers, and attestation blank for short human verification.

Build the primary offline curator UI in a directory that is separate from, outside, and does not
contain the immutable queue:

```powershell
python experiments/vlm_ab_evaluation/run_pipeline.py `
  --config $CapacityRemediationConfig `
  curate-forms `
  --prepare-manifest $CapacityPrepareManifest `
  --queue-manifest (Join-Path $CapacityQueue "queue_manifest.json") `
  --output-dir $CapacityForms
```

Before distribution or first opening, run the exact `curate-forms` command above again. Idempotent
generation accepts a byte-identical workspace; edited HTML, manifest, or copied images cause an
error. Then record and distribute the expected integrity values:

```powershell
Get-FileHash (Join-Path $CapacityForms "workspace_manifest.json") -Algorithm SHA256
Get-FileHash (Join-Path $CapacityForms "curator.html") -Algorithm SHA256
```

Experts compare those expected hashes and fingerprint with their handoff; the fingerprint is visibly
printed in the HTML header. This is an external integrity procedure, not browser self-verification.
Open the `curator.html` file in `$CapacityForms` directly in a browser. The workspace is fully offline, uses
queue-fingerprint-bound local autosave, and previews copied audited source images without embedding
their bytes in HTML. If local storage fails, the form remains usable in that tab and warns the expert
to export drafts regularly.

Each replacement-image section can select a verified local image and calculate lowercase SHA256 via
Web Crypto while showing an in-tab preview. Selection never derives `image_path` from the filename
and does not persist file bytes in local storage, drafts, HTML, or final JSONL. The curator must enter
the canonical `assets/images/...` path and separately copy the exact selected bytes into
`curated_dataset` at that path. If Web Crypto is unavailable, enter SHA256 manually; in every case
`curate-assemble` remains responsible for matching the declaration to the staged file bytes.

### Phase 0. Start manual review from the artifact-v3 queue

Use only `$CapacityForms` at `curator_workspace_v4`. It is the simplified expert form with plain-language
labels, examples, and instructions for every field. `curator_workspace_v3` is superseded; import one of
its drafts only when the v4 form accepts the exact queue fingerprint. Drafts from v1/v2 forms have a
different queue fingerprint and must not be imported or recovered into this workspace. Browser `file://` storage is not
portable; begin from the blank v3 workspace, then use **Merge draft** only for the queue-v3
`machine_assisted_draft.json`, queue-v3 triage draft, or human subset drafts with the same fingerprint.

Use assignments from `capacity_review_plan_v4/review_assignment_plan.csv`. Experts verify all 150
machine proposals, including the two identity-remediation groups, and confirm the 74 exclusion proposals
with real reviewer IDs and `independent_attestation=true`. **Сбросить неполные retain** remains only a
local recovery action for incomplete edits in this workspace; no historical reset count is authoritative.
No plan, reset, or merge creates a final decision, reviewer identity, or attestation.

### Phase 0. Build assisted triage proposals for training overlap

After generating the separate curator workspace, the release owner can build a separate, offline
assisted-triage package bound to the same immutable queue. The flag is deliberate: it records the
chosen policy to propose exclusion for every task with the critical `training_paper_overlap` audit
code or nonempty `training_overlap_paper_ids`.

```powershell
python experiments/vlm_ab_evaluation/run_pipeline.py `
  --config $CapacityRemediationConfig `
  curate-triage `
  --prepare-manifest $CapacityPrepareManifest `
  --queue-manifest (Join-Path $CapacityQueue "queue_manifest.json") `
  --output-dir $CapacityTriage `
  --exclude-training-overlap
```

The package is atomically published outside the queue and contains `assisted_review_draft.json`,
`triage.jsonl`, `triage_summary.md`, `triage_manifest.json`, and
`external_review_log_template.csv`. It never creates completed decisions, never supplies reviewer IDs
or attestations, and never auto-retains a row. The CSV has all task IDs and blank fields for two
reviewer IDs, review timestamps, and a signed/dated external-record reference. Treat only the values
computed in the fresh `triage_manifest.json` for this queue as confirmed; the manifest, not a hardcoded
count, is the operational authority.

In the master curator workspace, select **Merge draft** and import
`assisted_review_draft.json` from `$CapacityTriage`. The imported `exclude` values are proposals pending
two independent human attestations, not final decisions. Humans must verify the overlap, enter two
real reviewer IDs, and set the independent-attestation field before export. All tasks without training
overlap remain manual review; no `retain` decision is prefilled.

The software verifies queue bindings and at least two normalized-distinct declared IDs, but cannot
cryptographically establish human identity or independence. The release owner must retain an external,
signed or dated review log linking two real reviewers to every `task_id`.

Assign disjoint `task_id` subsets to experts. Each expert completes only the assigned subset and uses
**Export draft**. In one master workspace, the owner sequentially uses **Merge draft** for every
received envelope. Blank incoming tasks do not erase work, equal nonblank tasks are no-ops, and a
different nonblank task stops the entire merge without partial changes. Resolve conflicts with the
experts before proceeding. The form intentionally does not import completed decisions JSONL because
reconstructing form fields could lose arbitrary valid benchmark/provenance properties.

After all subsets are merged, complete every task, confirm the independent-expert attestation for
each decision, click **Export completed_decisions.jsonl**, and save that export to
`$CapacityDecisions`. Manual editing of a copy of `decision_template.jsonl` is a fallback only. In both
workflows the queue itself remains
byte-identical, and `curate-assemble` below is the final authority for schemas, image bytes, and the
scientific release gates; browser validation does not replace it.

After curators stage verified image bytes under a separate dataset root:

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

Assembly parses and hashes each JSONL from one stable byte snapshot, executes the canonical JSON
Schemas, verifies canonical paper identity, exact image order and bytes, rejects Windows path
aliases, reruns contamination/provenance/available-lineage audits, rejects frozen warning codes, and
enforces exactly 150 unique primary papers. Capacity assembly also reconstructs the evidence package,
requires its exact original source JSONL, and archives both byte-for-byte under
`audit/machine_enrichment/`. The assembly manifest co-binds those hashes with the independently
validated human review archive under `audit/human_review/`, including the exact queue, blank template,
and completed decisions. This preserves declared reviewer IDs and attestations for reproduction but does
not cryptographically prove real-world identity or independence. The machine policy retains
`external_evidence_content_verified=false`. Its output and embedded audit are only a
`validated_benchmark_release_candidate`; both deliberately remain `publication_ready=false` until
new immutable benchmark/adapter revisions and a fresh strict `prepare` exist.

### Phase 1. Publish corrected releases and create the strict capacity run

The candidate is not a benchmark release and cannot be used for inference. After assembly, publish the
corrected candidate as immutable benchmark revision `B`. Then publish the evaluated adapter/training
release `R`, followed by a later immutable lineage-attestation revision `M` that declares `R`. Create a
corrected strict config that pins `B`, `R`, and `M`, includes the required lineage manifest, and has the
same 150 verified unique nonoverlap primary papers. Its benchmark section must set
`assembly_manifest_file` to the published Phase 0 `assembly_manifest.json` and
`assembly_manifest_sha256` to that file's lowercase SHA256.

Run the final generator only from that corrected strict config. It preserves corrected B/R/M and all
other protocol settings, changes the final experiment identities and capacity power settings, and emits a
local preview rather than a preregistration.
The generator rejects a source without that reviewed assembly binding. Strict `prepare` verifies the
manifest's archived review bindings, exact counts, machine-evidence policy, and complete declared
release-file inventory. It copies that inventory at the same relative paths and revalidates the copied
tree before accepting the frozen run bundle.

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
```

Before committing, run the capacity-aware preflight:

```powershell
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
```

Commit the generated final config and every reviewed VLM source change. `plan` checks both
`git_dirty=false` and a 40-character Git `HEAD`, so the status output must be empty before it can create
or reuse the immutable power plan.

```powershell
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

Do not add `--exploratory`, `--benchmark-dir`, or local `--training-file` to the final prepare. It
downloads and validates strict inputs, executes the lineage schema, verifies the adapter/base binding,
and requires exactly 150 primary papers. Changes after `plan` require fresh identities, a fresh output
directory, a new clean commit, and a new plan.

### Phase 1. Run both arms

Run each arm in a separate process. Both commands use the same frozen input, processor, image settings,
chat template, generation parameters, and seed.

```powershell
python experiments/vlm_ab_evaluation/run_pipeline.py `
  --config $CapacityConfig `
  infer --arm base --backend transformers

python experiments/vlm_ab_evaluation/run_pipeline.py `
  --config $CapacityConfig `
  infer --arm tuned --backend transformers
```

Each JSONL append is flushed and synced. A rerun resumes only when all input/protocol/config
fingerprints match. The tuned loader explicitly constructs the pinned base and then calls
`PeftModel.from_pretrained`; explicit runtime checks verify that the requested adapter is active.
Complete sidecars hash the prediction JSONL bytes and bind them to the prepare/config/code
fingerprints. An existing prediction JSONL without its sidecar is never adopted. Strict mode rejects
mock, limited, stale, swapped, or edited predictions.

For a dependency-light infrastructure smoke test:

```bash
python -m pytest \
  tests/test_vlm_ab_pipeline.py::test_mock_pipeline_reaches_blind_review_and_aggregation -q
```

### Phase 1. Build blinded packages

The final strict config must contain exactly two real pseudonymous reviewer IDs, with both
`review.reviews_per_item` and `power.reviews_per_item` equal to `2`. Replace placeholders before
freezing the study roster. Historical blocked audit configs that still describe a three-reviewer pool
are not valid confirmatory configs; `plan`, final config generators, and strict review commands reject
that design.

```powershell
python experiments/vlm_ab_evaluation/run_pipeline.py `
  --config $CapacityConfig `
  blind
```

Each reviewer receives only their one opaque hashed directory under
`blind_review/public/<opaque-reviewer-id>/`. Never send `blind_review/owner_only/`,
`blind_review_manifest.json`, or the whole run directory. Arm position and item order are
counterbalanced: the second expert sees the opposite side for every pair and the reverse item order.
Evidence images are copied with
opaque names; model IDs, creator rationales, expected errors, sample IDs, and arm truth are
excluded from public files.

Each browser export carries a content-derived study fingerprint, rubric hash, independent-review
attestation, and reviewer-specific package nonce. The private mapping and the file
inventory are HMAC-protected by `blind_review/owner_only/randomization_secret.txt`; aggregation
aborts if responses, images, assignments, or arm truth were changed.

Reviewers open `review.html`, complete every item, and export JSON. Collect exports under a new
directory, for example `incoming_reviews/`, with unique filenames.

### Phase 1. Aggregate after the human barrier

```powershell
$IncomingReviews = Join-Path $CapacityRun "incoming_reviews"
python experiments/vlm_ab_evaluation/run_pipeline.py `
  --config $CapacityConfig `
  aggregate --reviews-dir $IncomingReviews
```

The aggregator verifies prepare, prediction, blind-package, owner-HMAC, roster, and completeness
before deblinding. `--allow-incomplete` is exploratory-only. It
creates:

- `analysis/results.json` with primary, secondary, agreement, position-bias, error, and format
  metrics;
- `analysis/tables/paper_scores.csv` with the independent analysis units;
- `analysis/figures/preference_effects.svg`;
- `analysis/report.md` with manuscript-ready wording and an interpretation guard;
- `analysis/deblinded_reviews.jsonl`, which is owner-only and must not be shared with reviewers.

## Cloud run

```bash
export DATASPHERE_PROJECT_ID="your-project-id"
datasphere project job execute -p "$DATASPHERE_PROJECT_ID" \
  -c experiments/vlm_ab_evaluation/datasphere/job_config.yaml
```

The strict current job should terminate at the invalid benchmark gate. This is intentional and
prevents expensive inference on a benchmark that cannot support a publication claim.

Expose `HF_TOKEN` to the job through a DataSphere project secret; do not place a token in YAML or
the repository. DataSphere exports the complete `runs/` tree together with the exact config,
evaluation source, and `pyproject.toml`; preserve that directory layout for later aggregation. This
ensures a newly preregistered `experiment.output_dir` is transferred instead of being silently
omitted by a hardcoded output path.
`--repo-root` is accepted only when the executing module is that root's source (an editable install
or `run_pipeline.py`), preventing a wheel from hashing a different checkout.
The job stages `.git/` so strict preparation can prove a clean commit; do not remove it from the
job inputs. Later aggregation may use the exported exact source snapshot and the clean Git
attestation stored by prepare.

## Artifact contract

Never delete or hand-edit run artifacts. Keep the entire run archive private. Archive together:

- config and its SHA256;
- prepare/audit manifests;
- both prediction JSONLs and sidecar manifests;
- public reviewer packages and private owner mapping separately;
- original review exports;
- deblinded analysis, tables, figure, and software commit/environment.

Distribute only one reviewer-specific directory from `blind_review/public/`. The private archive
must retain `blind_review/owner_only/`; the public reviewer handoff must not contain it.

See `DESIGN_RU.md` for the statistical estimand, hypotheses, exclusions, controls, and reporting
rules. Corrected releases must satisfy `schemas/publication_benchmark_row.schema.json` and
`schemas/publication_provenance_row.schema.json`. Gold and rubrics are optional for the blinded-human
primary endpoint; a substantive `gold_answer`, typed non-empty evidence, rubric criteria, and two
normalized-distinct adjudicators become mandatory when `benchmark.require_gold=true`. The required
training release contract is `schemas/training_lineage_manifest.schema.json`; its containing
attestation revision must be distinct from every training-source revision in the same repository.
