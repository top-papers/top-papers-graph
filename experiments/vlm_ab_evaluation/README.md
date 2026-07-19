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

## Pipeline

The configuration is
`experiments/vlm_ab_evaluation/configs/qwen3vl_scireason_publication.yaml`.

### 1. Preregister power and protocol

```bash
python experiments/vlm_ab_evaluation/run_pipeline.py \
  --config experiments/vlm_ab_evaluation/configs/qwen3vl_scireason_publication.yaml \
  plan
```

This writes an immutable `design/power_plan.json` under the configured run directory, including
the config and source-tree fingerprints. The command refuses to overwrite a different plan in the
same run directory. Archive it before inspecting model outputs.

### 2. Download, audit, and freeze

```bash
python experiments/vlm_ab_evaluation/run_pipeline.py \
  --config experiments/vlm_ab_evaluation/configs/qwen3vl_scireason_publication.yaml \
  prepare
```

The command pins and verifies Hub commit SHAs, downloads only declared benchmark/training files,
hashes every image, checks exact paper-ID and prompt overlap against the adapter's actual
`sft_all.jsonl` and `grpo_all.jsonl`, and writes:

- `audit/benchmark_audit.json` and `.md`;
- `inputs/frozen_benchmark.jsonl`;
- `inputs/dataset/` and `inputs/source/`, a relocatable copy of every frozen image and audited
  source JSONL;
- `prepare_manifest.json` with file hashes, source-tree fingerprint, Git state, and resolved
  revisions.

Strict preparation also requires complete per-image provenance, explicit paper/source/creator
holdout declarations, a clean evaluation source tree, and a pinned training-lineage manifest. Use
two immutable commits in the adapter repository: revision `R` contains the evaluated adapter and
exact training exports, while a later attestation revision `M` contains the manifest that declares
`R`. The config evaluates and audits `R` but downloads the manifest from `M`. Requiring the manifest
inside `R` while making it declare `R` would create an impossible Git commit self-reference. The
manifest binds every audited training JSONL by path, SHA256, and row count and enumerates non-empty
training paper/source/creator IDs plus image/prompt SHA256 sets. Fuzzy or derivative contamination
is not inferred from filenames; it must be supported by that release lineage.

Fresh strict preparation executes the lineage JSON Schema, verifies `adapter_config.json` against
the pinned base ID and commit, combines fragmented training prompt blocks for overlap checks,
requires citation-complete ordered provenance, promotes duplicate prompt/image-byte warnings to
blockers, and enforces `power.n_items` as the minimum number of unique primary papers. Archived audit
v2 remains reproducible for the existing remediation queue; new runs use audit v3. Lineage is
mandatory in strict mode even if `training_audit.require_lineage_manifest=false`; that setting cannot
waive the publication gate.

After a corrected benchmark is published, update its immutable revision and use a new
`experiment.id`, `experiment.public_id`, and `experiment.output_dir`. Do not overwrite a prior run
or manually remove failed audit rows and call the remainder the same benchmark.

### 2a. Build the human remediation queue

The archived failed prepare bundle under
`runs/vlm_ab/qwen3vl-scireason-remediation-audit-v2/` is the source for the immutable row-level
curator tasks. It uses audit contract v2 and its hashes must match
`remediation_queue_v2_baseline_20260717.json`. Restore that exact bundle from the trusted archive if
it is absent; do not run a fresh `prepare` into the same output directory. Current code creates audit
v3, so a new re-audit requires a new experiment ID, output directory, queue, and baseline hashes.

Every source row in the archived run has a technical blocker. Its historical exploratory `prepare`
wrote the failed manifest and exited with `no technically runnable rows`; that failure was not an
acceptance signal. Replaying `curate-queue` below first recomputes the stored v2 audit contract and
therefore verifies the archive without silently upgrading it to v3. Queue creation does not repair
any record or propose which duplicate/image mapping is correct.

```bash
python experiments/vlm_ab_evaluation/run_pipeline.py \
  --config experiments/vlm_ab_evaluation/configs/qwen3vl_scireason_remediation_audit_v2.yaml \
  curate-queue \
  --prepare-manifest runs/vlm_ab/qwen3vl-scireason-remediation-audit-v2/prepare_manifest.json \
  --output-dir runs/vlm_ab/qwen3vl-scireason-remediation-audit-v2/curation_queue_v2_524cbc2d_hardened
```

The queue contains immutable `tasks.jsonl`, blank `decision_template.jsonl`, and a hash-bound
`queue_manifest.json`. Version 2 binds the original config fingerprint, a fresh recomputation of the
prepared audit, and byte-identical canonical schemas. Curator tooling must emit a separate completed
decisions JSONL while leaving the queue directory unchanged. Every source row requires an explicit
retain/exclude disposition and two independent reviewer IDs. Retained records require complete
replacement benchmark and provenance objects; patches and inferred defaults are rejected.

Build the primary offline curator UI in a directory that is separate from, outside, and does not
contain the immutable queue:

```bash
python experiments/vlm_ab_evaluation/run_pipeline.py \
  --config experiments/vlm_ab_evaluation/configs/qwen3vl_scireason_remediation_audit_v2.yaml \
  curate-forms \
  --prepare-manifest runs/vlm_ab/qwen3vl-scireason-remediation-audit-v2/prepare_manifest.json \
  --queue-manifest runs/vlm_ab/qwen3vl-scireason-remediation-audit-v2/curation_queue_v2_524cbc2d_hardened/queue_manifest.json \
  --output-dir runs/vlm_ab/qwen3vl-scireason-remediation-audit-v2/curator_workspace_v2
```

Before distribution or first opening, run the exact `curate-forms` command above again. Idempotent
generation accepts a byte-identical workspace; edited HTML, manifest, or copied images cause an
error. Then record and distribute the expected integrity values:

```bash
sha256sum \
  runs/vlm_ab/qwen3vl-scireason-remediation-audit-v2/curator_workspace_v2/workspace_manifest.json \
  runs/vlm_ab/qwen3vl-scireason-remediation-audit-v2/curator_workspace_v2/curator.html
python -c "import json; print(json.load(open('runs/vlm_ab/qwen3vl-scireason-remediation-audit-v2/curator_workspace_v2/workspace_manifest.json', encoding='utf-8'))['queue_fingerprint'])"
```

Experts compare those expected hashes and fingerprint with their handoff; the fingerprint is visibly
printed in the HTML header. This is an external integrity procedure, not browser self-verification.
Open `curator_workspace_v2/curator.html` directly in a browser. The workspace is fully offline, uses
queue-fingerprint-bound local autosave, and previews copied audited source images without embedding
their bytes in HTML. If local storage fails, the form remains usable in that tab and warns the expert
to export drafts regularly.

Each replacement-image section can select a verified local image and calculate lowercase SHA256 via
Web Crypto while showing an in-tab preview. Selection never derives `image_path` from the filename
and does not persist file bytes in local storage, drafts, HTML, or final JSONL. The curator must enter
the canonical `assets/images/...` path and separately copy the exact selected bytes into
`curated_dataset` at that path. If Web Crypto is unavailable, enter SHA256 manually; in every case
`curate-assemble` remains responsible for matching the declaration to the staged file bytes.

Assign disjoint `task_id` subsets to experts. Each expert completes only the assigned subset and uses
**Export draft**. In one master workspace, the owner sequentially uses **Merge draft** for every
received envelope. Blank incoming tasks do not erase work, equal nonblank tasks are no-ops, and a
different nonblank task stops the entire merge without partial changes. Resolve conflicts with the
experts before proceeding. The form intentionally does not import completed decisions JSONL because
reconstructing form fields could lose arbitrary valid benchmark/provenance properties.

After all subsets are merged, complete every task, confirm the independent-expert attestation for
each decision, click **Export completed_decisions.jsonl**, and save that export as
`runs/vlm_ab/qwen3vl-scireason-remediation-audit-v2/completed_decisions.jsonl`. Manual editing of a
copy of `decision_template.jsonl` is a fallback only. In both workflows the queue itself remains
byte-identical, and `curate-assemble` below is the final authority for schemas, image bytes, and the
scientific release gates; browser validation does not replace it.

After curators stage verified image bytes under a separate dataset root:

```bash
python experiments/vlm_ab_evaluation/run_pipeline.py \
  --config experiments/vlm_ab_evaluation/configs/qwen3vl_scireason_remediation_audit_v2.yaml \
  curate-assemble \
  --prepare-manifest runs/vlm_ab/qwen3vl-scireason-remediation-audit-v2/prepare_manifest.json \
  --queue-manifest runs/vlm_ab/qwen3vl-scireason-remediation-audit-v2/curation_queue_v2_524cbc2d_hardened/queue_manifest.json \
  --decisions-jsonl runs/vlm_ab/qwen3vl-scireason-remediation-audit-v2/completed_decisions.jsonl \
  --curated-dataset-root runs/vlm_ab/qwen3vl-scireason-remediation-audit-v2/curated_dataset \
  --output-dir runs/vlm_ab/qwen3vl-scireason-remediation-audit-v2/release_candidate
```

Assembly parses and hashes each JSONL from one stable byte snapshot, executes the canonical JSON
Schemas, verifies canonical paper identity, exact image order and bytes, rejects Windows path
aliases, reruns contamination/provenance/available-lineage audits, rejects frozen warning codes, and
enforces 240 unique primary papers. Its output and embedded audit are only a
`validated_benchmark_release_candidate`; both deliberately remain `publication_ready=false` until
new immutable benchmark/adapter revisions and a fresh strict `prepare` exist.

### 3. Run both arms

Run each arm in a separate process. Both commands use the same frozen input, processor, image
settings, chat template, generation parameters, and seed.

```bash
python experiments/vlm_ab_evaluation/run_pipeline.py \
  --config experiments/vlm_ab_evaluation/configs/qwen3vl_scireason_publication.yaml \
  infer --arm base --backend transformers

python experiments/vlm_ab_evaluation/run_pipeline.py \
  --config experiments/vlm_ab_evaluation/configs/qwen3vl_scireason_publication.yaml \
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

### 4. Build blinded packages

Replace placeholder reviewer IDs in the config before freezing the study roster.

```bash
python experiments/vlm_ab_evaluation/run_pipeline.py \
  --config experiments/vlm_ab_evaluation/configs/qwen3vl_scireason_publication.yaml \
  blind
```

Each reviewer receives only their one opaque hashed directory under
`blind_review/public/<opaque-reviewer-id>/`. Never send `blind_review/owner_only/`,
`blind_review_manifest.json`, or the whole run directory. Arm position and item order are
independently randomized and near-balanced for each reviewer. Evidence images are copied with
opaque names; model IDs, creator rationales, expected errors, sample IDs, and arm truth are
excluded from public files.

Each browser export carries a content-derived study fingerprint. The private mapping and the file
inventory are HMAC-protected by `blind_review/owner_only/randomization_secret.txt`; aggregation
aborts if responses, images, assignments, or arm truth were changed.

Reviewers open `review.html`, complete every item, and export JSON. Collect exports under a new
directory, for example `incoming_reviews/`, with unique filenames.

### 5. Aggregate after the human barrier

```bash
python experiments/vlm_ab_evaluation/run_pipeline.py \
  --config experiments/vlm_ab_evaluation/configs/qwen3vl_scireason_publication.yaml \
  aggregate --reviews-dir runs/vlm_ab/qwen3vl-8b-scireason-task3-ab-v1/incoming_reviews
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
