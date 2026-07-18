# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scireason.vlm_ab.audit import (
    BenchmarkAuditError,
    audit_benchmark,
    benchmark_paper_ids,
    canonical_paper_id,
    load_jsonl,
    write_audit_report,
)


def _row(
    sample_id: str,
    paper_id: str,
    image: str,
    prompt: str,
    *,
    reference_answer: str = "The plotted value is 42 units.",
) -> dict:
    return {
        "sample_id": sample_id,
        "benchmark_version": "task3_hf_benchmark_v1",
        "task_family": "task3_vlm_ab_generation",
        "stratum": "multimodal_hard",
        "primary_endpoint": True,
        "paper_id": paper_id,
        "model_task_prompt": prompt,
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Use only the supplied scientific evidence."}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, {"type": "image"}],
            },
        ],
        "images": [image],
        "reference_answer": reference_answer,
        "gold_answer": {
            "answer": reference_answer,
            "evidence_used": ["supplied figure"],
            "visual_facts": ["The plotted value is visible."],
            "temporal_facts": ["The figure reports the stated period."],
            "uncertainty": None,
            "missing_evidence": None,
        },
        "rubric": {
            "criteria": ["The response reports the plotted value."],
            "adjudicators": ["adjudicator-1", "adjudicator-2"],
        },
    }


def _provenance(row: dict, image_bytes: bytes) -> dict:
    return {
        "sample_id": row["sample_id"],
        "paper_id": row["paper_id"],
        "image": row["images"][0],
        "sha256": hashlib.sha256(image_bytes).hexdigest(),
    }


def _codes(report: dict, field: str) -> set[str]:
    return {finding["code"] for finding in report[field]}


def test_tracked_remote_audit_contract_is_internally_consistent() -> None:
    root = Path(__file__).resolve().parents[1] / "experiments" / "vlm_ab_evaluation"
    baseline = json.loads((root / "remote_audit_baseline_20260717.json").read_text("utf-8"))
    remediation = json.loads(
        (root / "remediation_queue_v2_baseline_20260717.json").read_text("utf-8")
    )
    benchmark_schema = json.loads(
        (root / "schemas" / "publication_benchmark_row.schema.json").read_text("utf-8")
    )
    provenance_schema = json.loads(
        (root / "schemas" / "publication_provenance_row.schema.json").read_text("utf-8")
    )

    assert sum(item["findings"] for item in baseline["critical_findings"].values()) == 1546
    assert sum(item["findings"] for item in baseline["warning_findings"].values()) == 938
    assert baseline["gate"]["eligible_unique_samples"] == 0
    assert baseline["diagnostics"]["duplicate_ids"]["extra_rows"] == 386 - 360
    assert baseline["release_acceptance"]["benchmark_contract"] == (
        "schemas/publication_benchmark_row.schema.json"
    )
    assert "gold_answer" not in benchmark_schema["required"]
    assert "rubric" not in benchmark_schema["required"]
    provenance_item = provenance_schema["properties"]["images"]["items"]
    assert "verified_by" in provenance_item["required"]
    assert provenance_item["properties"]["verified_by"]["minItems"] == 2
    assert (
        remediation["source_reaudit"]["original_baseline_audit_json_sha256"]
        == baseline["sources"]["audit_report_sha256"]
    )
    assert remediation["source_reaudit"]["audit_matches_original_baseline"] is False
    assert remediation["source_reaudit"]["audit_invariants_match_original_baseline"] is True
    assert remediation["queue"]["artifact_version"] == 2
    assert remediation["invariants"]["tasks"] == baseline["population"]["benchmark_rows"]
    assert remediation["invariants"]["pending_decisions"] == remediation["invariants"]["tasks"]
    assert remediation["publication_ready"] is False
    tracked_hashes = {
        "config_file_sha256": hashlib.sha256(
            (root / remediation["source_reaudit"]["config_file"]).read_bytes()
        ).hexdigest(),
        "benchmark_schema_sha256": hashlib.sha256(
            (root / "schemas" / "publication_benchmark_row.schema.json").read_bytes()
        ).hexdigest(),
        "provenance_schema_sha256": hashlib.sha256(
            (root / "schemas" / "publication_provenance_row.schema.json").read_bytes()
        ).hexdigest(),
    }
    assert (
        tracked_hashes["config_file_sha256"] == remediation["source_reaudit"]["config_file_sha256"]
    )
    assert (
        tracked_hashes["benchmark_schema_sha256"] == remediation["queue"]["benchmark_schema_sha256"]
    )
    assert (
        tracked_hashes["provenance_schema_sha256"]
        == remediation["queue"]["provenance_schema_sha256"]
    )
    attributes = (root.parents[1] / ".gitattributes").read_text(encoding="utf-8")
    assert "experiments/vlm_ab_evaluation/schemas/*.json text eol=lf" in attributes
    assert "experiments/vlm_ab_evaluation/configs/*.yaml text eol=lf" in attributes


def test_clean_fixture_passes_and_is_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    image_dir = root / "assets"
    image_dir.mkdir(parents=True)
    first_bytes = b"first-scientific-figure"
    second_bytes = b"second-scientific-figure"
    (image_dir / "first.png").write_bytes(first_bytes)
    (image_dir / "second.png").write_bytes(second_bytes)
    rows = [
        _row(
            "sample-b",
            "https://doi.org/10.5555/CLEAN.B",
            "assets/second.png",
            "Extract the trend shown in Figure 2.",
        ),
        _row(
            "sample-a",
            "arXiv:2401.00001v2",
            "assets/first.png",
            "State the measured value shown in Figure 1.",
        ),
    ]
    provenance = [
        _provenance(rows[0], second_bytes),
        _provenance(rows[1], first_bytes),
    ]
    training = [
        {
            "metadata": {"paper": {"doi": "10.5555/unrelated"}},
            "messages": [{"role": "user", "content": "Summarize an unrelated table."}],
        }
    ]

    report = audit_benchmark(
        rows,
        root,
        provenance_rows=provenance,
        training_rows=training,
        require_gold=True,
    )
    repeated = audit_benchmark(
        rows,
        root,
        provenance_rows=provenance,
        training_rows=training,
        require_gold=True,
    )

    assert report == repeated
    assert report["status"] == "pass"
    assert report["publication_ready"] is True
    assert report["eligible_sample_ids"] == ["sample-a", "sample-b"]
    assert report["summary"]["hashed_images"] == 2
    assert report["critical_findings"] == []
    assert report["warnings"] == []


def test_corruption_relabelled_bytes_training_doi_and_prompt_leakage_fail(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    image_dir = root / "assets"
    image_dir.mkdir(parents=True)
    duplicated_bytes = b"same-page-bytes-relabelled-as-two-papers"
    (image_dir / "paper-a.png").write_bytes(duplicated_bytes)
    (image_dir / "paper-b.png").write_bytes(duplicated_bytes)
    leaked = "The expected answer is 42 units. Compare model A and model B."
    rows = [
        _row("sample-a", "doi:10.7777/CONTAMINATED", "assets/paper-a.png", leaked),
        _row(
            "sample-b",
            "doi:10.8888/relabelled",
            "assets/paper-b.png",
            "Extract the value from the supplied figure.",
        ),
    ]
    provenance = [
        _provenance(rows[0], duplicated_bytes),
        {
            **_provenance(rows[1], duplicated_bytes),
            "paper_id": "doi:10.7777/contaminated",
        },
    ]
    training = [
        {
            "payload": {
                "prompt": "  THE EXPECTED ANSWER IS 42 UNITS. COMPARE MODEL A AND MODEL B. ",
                "source": {"publication": {"doi": "https://doi.org/10.7777/contaminated"}},
            }
        }
    ]

    report = audit_benchmark(
        rows,
        root,
        provenance_rows=provenance,
        training_rows=training,
    )

    assert report["status"] == "fail"
    assert report["publication_ready"] is False
    assert {
        "cross_paper_image_reuse",
        "provenance_mismatch",
        "residual_comparison_prompt",
        "likely_answer_leakage",
        "training_paper_overlap",
        "training_prompt_overlap",
    } <= _codes(report, "critical_findings")
    reuse = report["contamination"]["cross_paper_image_reuse"][0]
    assert "does not prove" in reuse["interpretation"]


def test_duplicate_ids_placeholder_mismatch_and_traversal_are_critical(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    row = _row(
        "duplicate",
        "arxiv:2402.00002",
        "../outside.png",
        "Read the supplied figure.",
    )
    row["messages"][1]["content"] = [{"type": "text", "text": row["model_task_prompt"]}]

    report = audit_benchmark([row, dict(row)], root)

    assert report["status"] == "fail"
    assert {
        "duplicate_sample_id",
        "image_placeholder_mismatch",
        "unsafe_image_path",
    } <= _codes(report, "critical_findings")
    assert report["eligible_sample_ids"] == []


def test_missing_gold_only_blocks_when_required(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "page.png").write_bytes(b"page")
    row = _row("sample", "arxiv:2403.00003", "page.png", "Read the figure.")
    reference_only = dict(row)
    del reference_only["gold_answer"]
    del reference_only["rubric"]
    del row["reference_answer"]
    del row["gold_answer"]
    del row["rubric"]

    warning_report = audit_benchmark([row], root)
    required_report = audit_benchmark([reference_only], root, require_gold=True)

    assert warning_report["status"] == "pass"
    assert "missing_gold" in _codes(warning_report, "warnings")
    assert required_report["status"] == "fail"
    assert "missing_gold" in _codes(required_report, "critical_findings")

    fake_nested = dict(reference_only)
    fake_nested["metadata"] = {"gold": "x", "rubric": "y"}
    fake_report = audit_benchmark([fake_nested], root, require_gold=True)
    assert "missing_gold" in _codes(fake_report, "critical_findings")

    nested_only = dict(row)
    nested_only["metadata"] = {"gold": "x", "rubric": "y"}
    nested_report = audit_benchmark([nested_only], root)
    assert "missing_gold" in _codes(nested_report, "warnings")
    assert nested_report["summary"]["samples_with_gold"] == 0
    assert nested_report["summary"]["samples_with_rubric"] == 0


def test_conflicting_paper_aliases_and_unknown_message_blocks_fail(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "page.png").write_bytes(b"page")
    row = _row("sample", "doi:10.5555/canonical", "page.png", "Read the figure.")
    row = {"paper-id": "doi:10.5555/conflict", **row}
    row["doi"] = "10.5555/unsafe\u200balias"
    row["paper\u200b_id"] = "doi:10.5555/hidden-conflict"
    row["paper%5Fidentifier"] = "doi:10.5555/encoded-conflict"
    row["paper%2\u200b55fid"] = "doi:10.5555/encoded-after-filter"
    row["payload"] = {"paper_id": "doi:10.5555/nested-conflict"}
    row["messages"][0]["model_identity"] = "must-not-reach-inference"
    row["messages"][1]["content"].append({"type": "video", "url": "https://example.invalid/video"})

    report = audit_benchmark([row], root)

    assert "schema_error" in _codes(report, "critical_findings")
    schema_errors = next(
        finding["details"]["errors"]
        for finding in report["critical_findings"]
        if finding["code"] == "schema_error"
    )
    assert "paper identifier aliases conflict with top-level paper_id" in schema_errors
    assert any("alias 'doi' must contain safe canonical" in error for error in schema_errors)
    assert any("alias key 'paper\\u200b_id' contains unsafe" in error for error in schema_errors)
    assert any("alias key 'paper%5Fidentifier' contains unsafe" in error for error in schema_errors)
    assert any("paper%2\\u200b55fid" in error and "unsafe" in error for error in schema_errors)
    assert any("must contain exactly role and content" in error for error in schema_errors)
    assert any("type is unsupported" in error for error in schema_errors)


def test_training_text_extracts_encoded_and_opaque_identifiers(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "doi.png").write_bytes(b"doi")
    (root / "opaque.png").write_bytes(b"opaque")
    rows = [
        _row("doi-sample", "doi:10.5555/encoded", "doi.png", "Read the DOI figure."),
        _row("opaque-sample", "paper:local-7", "opaque.png", "Read the local figure."),
    ]
    training = [
        {
            "payload": (
                "malformed %FF text\nencoded doi%253A10.5555%252Fencoded and opaque paper:local-7"
            )
        }
    ]

    report = audit_benchmark(rows, root, training_rows=training)

    overlaps = report["contamination"]["training_paper_overlaps"]
    assert {item["paper_id"] for item in overlaps} == {
        "doi:10.5555/encoded",
        "paper:local-7",
    }


def test_audit_v1_preserves_legacy_paper_id_semantics(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "page.png").write_bytes(b"page")
    row = _row("sample", "unsafe\u034f-paper", "page.png", "Read the figure.")
    row["doi"] = "10.5555/legacy-paper"

    legacy = audit_benchmark([row], root, audit_version=1)
    current = audit_benchmark([row], root)

    assert legacy["audit_version"] == 1
    assert legacy["status"] == "pass"
    assert current["audit_version"] == 3
    assert current["status"] == "fail"
    assert "schema_error" in _codes(current, "critical_findings")
    with pytest.raises(BenchmarkAuditError, match="audit_version"):
        audit_benchmark([row], root, audit_version=0)


def test_within_row_duplicate_bytes_are_a_warning(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "one.png").write_bytes(b"duplicate-page")
    (root / "two.png").write_bytes(b"duplicate-page")
    row = _row("sample", "doi:10.9999/one-paper", "one.png", "Read both pages.")
    row["images"].append("two.png")
    row["messages"][1]["content"].append({"type": "image"})

    report = audit_benchmark([row], root)

    assert report["status"] == "pass"
    assert "within_row_duplicate_image_bytes" in _codes(report, "warnings")
    assert report["contamination"]["cross_paper_image_reuse"] == []


def test_empty_images_and_missing_canonical_prompt_fail_schema(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    row = _row("sample", "arxiv:2404.00004", "unused.png", "Read the figure.")
    row["images"] = []
    row["messages"][1]["content"] = [{"type": "text", "text": "Read the figure."}]
    del row["model_task_prompt"]

    report = audit_benchmark([row], root)

    assert report["status"] == "fail"
    assert {"missing_image", "schema_error"} <= _codes(report, "critical_findings")


def test_publication_provenance_and_split_contract_is_enforced(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    image_bytes = b"verified-publication-image"
    (root / "page.png").write_bytes(image_bytes)
    row = _row(
        "publication-sample",
        "doi:10.5555/publication",
        "page.png",
        "Read the verified figure.",
    )
    row["split_provenance"] = {
        "paper_holdout": True,
        "source_holdout": True,
        "creator_holdout": True,
        "training_overlap_checked": True,
        "source_document_id": "source-document-1",
        "creator_group_id": "creator-group-1",
    }
    provenance = {
        **_provenance(row, image_bytes),
        "page": 3,
        "locator": "Figure 1",
        "source_url": "https://doi.org/10.5555/publication",
        "license": "CC-BY-4.0",
        "verified_by": ["curator-1", "curator-2"],
    }

    passing = audit_benchmark(
        [row],
        root,
        provenance_rows=[provenance],
        require_complete_provenance=True,
        require_split_provenance=True,
    )
    assert passing["publication_ready"] is True

    del provenance["license"]
    failing = audit_benchmark(
        [row],
        root,
        provenance_rows=[provenance],
        require_complete_provenance=True,
        require_split_provenance=True,
    )
    assert failing["publication_ready"] is False
    assert "provenance_mismatch" in _codes(failing, "critical_findings")


def test_training_lineage_manifest_is_required_and_intersected(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    image_bytes = b"lineage-image"
    (root / "page.png").write_bytes(image_bytes)
    row = _row("lineage-sample", "doi:10.5555/heldout", "page.png", "Read the plot.")
    row["split_provenance"] = {
        "paper_holdout": True,
        "source_holdout": True,
        "creator_holdout": True,
        "training_overlap_checked": True,
        "source_document_id": "heldout-source",
        "creator_group_id": "heldout-creator",
    }
    base_lineage = {
        "schema_version": 1,
        "training_sources": [
            {
                "repo_id": "example/adapter",
                "repo_type": "model",
                "revision": "a" * 40,
                "files": [
                    {
                        "path": "training.jsonl",
                        "sha256": "0" * 64,
                        "row_count": 0,
                    }
                ],
            }
        ],
        "coverage": {
            "paper_ids": True,
            "source_documents": True,
            "creator_groups": True,
            "image_bytes": True,
            "prompts": True,
        },
        "paper_ids": ["doi:10.5555/training"],
        "source_document_ids": ["training-source"],
        "creator_group_ids": ["training-creator"],
        "image_sha256s": ["0" * 64],
        "prompt_sha256s": ["1" * 64],
    }
    expected_sources = base_lineage["training_sources"]

    missing = audit_benchmark([row], root, require_training_lineage=True)
    assert "missing_training_lineage" in _codes(missing, "critical_findings")

    passing = audit_benchmark(
        [row],
        root,
        training_lineage=base_lineage,
        expected_training_sources=expected_sources,
        require_training_lineage=True,
    )
    assert passing["publication_ready"] is True

    overlapping = {
        **base_lineage,
        "image_sha256s": [hashlib.sha256(image_bytes).hexdigest()],
    }
    failing = audit_benchmark(
        [row],
        root,
        training_lineage=overlapping,
        expected_training_sources=expected_sources,
        require_training_lineage=True,
    )
    assert "training_lineage_overlap" in _codes(failing, "critical_findings")

    normalized_overlap = {
        **base_lineage,
        "source_document_ids": ["HELDOUT-SOURCE"],
    }
    failing = audit_benchmark(
        [row],
        root,
        training_lineage=normalized_overlap,
        expected_training_sources=expected_sources,
        require_training_lineage=True,
    )
    assert failing["contamination"]["training_lineage"]["overlaps"]["source_document_ids"] == [
        "heldout-source"
    ]


@pytest.mark.parametrize(
    "training_row",
    [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Read the"},
                        {"type": "text", "text": "plot."},
                    ],
                }
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Read the"},
                {"role": "assistant", "content": "Continue."},
                {"role": "user", "content": "plot."},
            ]
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Read the pl"},
                        {"type": "text", "text": "ot."},
                    ],
                }
            ]
        },
        {"instruction": "Read", "input": ["the pl", "ot."]},
    ],
)
def test_audit_v3_detects_fragmented_training_prompts(tmp_path: Path, training_row: dict) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "page.png").write_bytes(b"split-prompt-image")
    row = _row("split-prompt", "doi:10.5555/heldout", "page.png", "Read the plot.")
    training = [training_row]

    legacy = audit_benchmark([row], root, training_rows=training, audit_version=2)
    hardened = audit_benchmark([row], root, training_rows=training)

    assert "training_prompt_overlap" not in _codes(legacy, "critical_findings")
    assert "training_prompt_overlap" in _codes(hardened, "critical_findings")


def test_strict_audit_promotes_frozen_warnings_and_requires_primary_n(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "first.png").write_bytes(b"first-strict-image")
    (root / "second.png").write_bytes(b"second-strict-image")
    prompt = "Read the shared scientific plot."
    rows = [
        _row("strict-a", "doi:10.5555/strict.a", "first.png", prompt),
        _row("strict-b", "doi:10.5555/strict.b", "second.png", prompt),
    ]

    report = audit_benchmark(
        rows,
        root,
        blocked_warning_codes=("duplicate_normalized_prompt",),
        minimum_primary_papers=3,
    )

    assert "duplicate_normalized_prompt" in _codes(report, "critical_findings")
    assert "insufficient_primary_papers" in _codes(report, "critical_findings")
    assert report["publication_ready"] is False


def test_strict_audit_requires_provenance_order_and_citation(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    image_bytes = {"first.png": b"first-order-image", "second.png": b"second-order-image"}
    for name, payload in image_bytes.items():
        (root / name).write_bytes(payload)
    row = _row("ordered", "doi:10.5555/ordered", "first.png", "Compare the plotted intervals.")
    row["images"] = ["first.png", "second.png"]
    row["messages"][1]["content"].append({"type": "image"})

    def entry(path: str) -> dict:
        return {
            "image_path": path,
            "sha256": hashlib.sha256(image_bytes[path]).hexdigest(),
            "page": 1,
            "locator": path,
            "source_url": "https://example.org/paper",
            "license": "CC-BY-4.0",
            "verified_by": ["verifier-1", "verifier-2"],
        }

    provenance = {
        "sample_id": row["sample_id"],
        "paper_id": row["paper_id"],
        "images": [entry("second.png"), entry("first.png")],
    }
    citation_only = audit_benchmark(
        [row],
        root,
        provenance_rows=[provenance],
        require_citation=True,
    )
    assert "provenance_mismatch" in _codes(citation_only, "critical_findings")

    failing = audit_benchmark(
        [row],
        root,
        provenance_rows=[provenance],
        require_complete_provenance=True,
        require_provenance_order=True,
        require_citation=True,
    )
    assert "provenance_mismatch" in _codes(failing, "critical_findings")
    issues = failing["provenance"]["mismatches"][0]["issues"]
    assert any("image order differs" in issue for issue in issues)
    assert any("requires citation" in issue for issue in issues)

    provenance["images"] = [entry("first.png"), entry("second.png")]
    for image in provenance["images"]:
        image["citation"] = "Example et al., Figure 1"
    passing = audit_benchmark(
        [row],
        root,
        provenance_rows=[provenance],
        require_complete_provenance=True,
        require_provenance_order=True,
        require_citation=True,
    )
    assert "provenance_mismatch" not in _codes(passing, "critical_findings")


def test_canonical_jsonl_and_report_writers(tmp_path: Path) -> None:
    assert canonical_paper_id("https://doi.org/10.1000/ABC.") == "doi:10.1000/abc"
    assert canonical_paper_id("https://arxiv.org/pdf/2401.01234v3.pdf") == ("arxiv:2401.01234")
    assert canonical_paper_id(" Local Paper  7 ") == "paper:local paper 7"
    assert (
        canonical_paper_id(
            "https%3A%2F%2Fdoi.org%2F%EF%BC%91%EF%BC%90.%EF%BC%95%EF%BC%95%EF%BC%95%EF%BC%95%2FABC"
        )
        == "doi:10.5555/abc"
    )
    assert canonical_paper_id("doi%253A10.5555%252FABC") == "doi:10.5555/abc"
    assert benchmark_paper_ids(
        {
            "paper_id": "doi:10.5555/one",
            "metadata": {
                "paper_ids": [
                    "bad%FF value\ndoi:10.5555/three",
                    "doi%253A10.5555%252Ftwo and arXiv:2401.01234v2",
                ]
            },
        }
    ) == [
        "doi:10.5555/one",
        "doi:10.5555/three",
        "doi:10.5555/two",
        "arxiv:2401.01234",
    ]
    assert canonical_paper_id("local\u034f-paper") == ""
    assert canonical_paper_id("local\u180b-paper") == ""
    assert canonical_paper_id("local\u115f-paper") == ""

    source = tmp_path / "rows.jsonl"
    source.write_text('\n{"sample_id": "one"}\n', encoding="utf-8")
    assert load_jsonl(source) == [{"sample_id": "one"}]
    source.write_text('{"sample_id": "one", "sample_id": "two"}\n', encoding="utf-8")
    with pytest.raises(BenchmarkAuditError, match="duplicate JSON key"):
        load_jsonl(source)

    report = {
        "status": "fail",
        "publication_ready": False,
        "summary": {"critical_findings": 1},
        "critical_findings": [{"code": "example", "message": "Example failure", "sample_ids": []}],
        "warnings": [],
        "contamination": {},
        "eligible_sample_ids": [],
    }
    json_path = tmp_path / "reports" / "audit.json"
    markdown_path = tmp_path / "reports" / "audit.md"
    write_audit_report(report, json_path, markdown_path)

    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    assert b"\r\n" not in json_path.read_bytes()
    assert b"\r\n" not in markdown_path.read_bytes()
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "blocks publication claims" in markdown
    assert "do not prove" in markdown
