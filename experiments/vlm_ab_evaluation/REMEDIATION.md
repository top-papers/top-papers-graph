<!-- SPDX-FileCopyrightText: 2026 top-papers-graph contributors -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# Benchmark remediation contract

## Scope

This contract records the authenticated audit of the pinned benchmark and adapter releases. The
machine-readable source of truth is `remote_audit_baseline_20260717.json`. Its source hashes bind the
counts below to the exact downloaded bytes; it is not a corrected dataset.

`remediation_queue_v2_baseline_20260717.json` separately records the retained-config re-audit and
version 2 queue hashes. The 2026-07-18 hardened re-audit preserves the original population and
finding-count invariants but intentionally has a new audit SHA256. It does not replace or mutate the
original source baseline.

The audited release cannot be repaired by filtering rows. All 360 unique samples are blocked by
cross-paper image-byte reuse, incomplete provenance, residual comparison wording, and schema
errors. The correction must be a new curated release with a new immutable revision.

## Verified baseline

| Metric | Value |
| --- | ---: |
| Benchmark rows | 386 |
| Unique sample IDs | 360 |
| Eligible samples | 0 |
| Image references | 1,544 |
| Unique image paths | 1,488 |
| Unique image contents | 30 |
| Critical findings | 1,546 |
| Warnings | 938 |
| Duplicate-ID groups / extra rows | 20 / 26 |
| Training-overlap paper IDs / unique samples / rows | 22 / 70 / 71 |

The 46 rows in duplicate-ID groups are all distinct as full JSON records. A script must not choose
which record to keep. Likewise, equal image hashes establish byte reuse but cannot identify the
correct paper assignment.

## Correction ownership

### Deterministic checks

- Recompute SHA256 values, row counts, prompt normalization, and image-byte groups.
- Synchronize each finalized `model_task_prompt` with the canonical user message.
- Set `primary_endpoint` from the frozen `primary_strata` policy.
- Validate every row against `schemas/publication_benchmark_row.schema.json`.
- Validate one aggregated provenance row per sample against
  `schemas/publication_provenance_row.schema.json`.

These operations verify curator output. They must not invent paper mappings, prompts, licenses,
gold answers, or duplicate dispositions.

### Curator and adjudicator decisions

- Resolve every duplicate-ID group and record the disposition of every source row.
- Rewrite prompts as standalone single-model tasks without A/B wording or embedded expected facts.
- Retrieve each image from the canonical paper and verify paper, page, locator, source URL, license,
  and bytes with two independent verifier identifiers.
- Replace or remove every cross-paper and within-row duplicate image after source verification.
- Add independently adjudicated gold answers and rubrics if automatic semantic metrics will be
  claimed. Gold is optional for the preregistered blinded-human primary endpoint.

### Release-owner decisions

- Exclude all training-overlap papers from evaluation or retrain and republish the adapter.
- Publish the corrected adapter and exact training exports as immutable revision `R`.
- Publish `schemas/training_lineage_manifest.schema.json`-compatible lineage at a distinct later
  immutable attestation revision `M` in the same adapter repository; the manifest declares `R`,
  while inference continues to evaluate `R`.
- Attest complete paper, source-document, creator-group, image-byte, and prompt coverage.
- Publish new benchmark and adapter revisions. Never rewrite the audited revisions in place.

## Canonical release layout

The benchmark JSONL contains one row per unique `sample_id`. The provenance JSONL contains exactly
one row for the same `sample_id`, with all evidence entries nested under `images`. Array order and
paths must match the benchmark row exactly. Cross-file uniqueness and equality are enforced by
`prepare`, because JSON Schema cannot express those constraints.

For compatibility with the immutable queue, the frozen provenance v1 schema leaves `citation`
optional. Current `curate-assemble` and fresh strict audit v3 nevertheless require a substantive
citation for every retained image. Audit v3 also joins fragmented training text blocks, promotes
duplicate prompt/image-byte warnings to strict blockers, and enforces the preregistered primary-paper
minimum during `prepare`.

The benchmark contract keeps `gold_answer` and `rubric` optional. Set `benchmark.require_gold=true`
for a preregistration that claims automatic semantic metrics; that setting requires both a
substantive adjudicated gold record and rubric with normalized-distinct adjudicators.

## Implemented workflow

`curate-queue` verifies the complete failed prepare bundle, including its config fingerprint, a fresh
audit recomputation, and non-frozen image bytes, and creates one deterministic task per source
benchmark row. Task IDs bind the prepare-manifest hash, source row index, and canonical row hash.
The generated decision template contains no proposed correction. Queue artifact version 2 accepts
only byte-identical canonical publication schemas.

`curate-assemble` accepts a separate completed decisions JSONL. Every decision must use
`status="complete"`, explicitly retain or exclude the row, and identify two independent reviewers.
Retained decisions provide full replacement benchmark/provenance objects. The command rejects
patches, missing decisions, changed bindings, schema errors, conflicting paper identifiers, unsafe
or Windows-aliased paths, image hash/order changes, training overlap, duplicate prompts/bytes, and
fewer than 240 unique primary papers. Every decisions JSONL is parsed and hashed from one stable byte
snapshot.

The command stages outputs atomically and labels both the assembly manifest and embedded audit
`validated_benchmark_release_candidate`, not publication-ready. Existing bundled training lineage is
included in the candidate audit; absent lineage domains remain explicitly unresolved. The command
performs no Hub upload and does not create a training-lineage attestation.

## Acceptance

A corrected release is acceptable only when a fresh strict `prepare` reports:

- zero critical findings and every unique sample eligible;
- zero duplicate IDs, cross-paper image groups, provenance mismatches, residual comparison prompts,
  missing/unsafe images, placeholder mismatches, and training paper/prompt overlaps;
- zero duplicate normalized prompts and within-row duplicate image-byte warnings;
- a valid complete training-lineage manifest with no overlap in any declared domain;
- at least 240 independent primary papers for the preregistered design.

`missing_gold` may remain only when `benchmark.require_gold=false` and the study makes no automatic
semantic-quality claim. After acceptance, create a new `experiment.id`, `experiment.public_id`,
`experiment.output_dir`, and immutable power plan before inference.
