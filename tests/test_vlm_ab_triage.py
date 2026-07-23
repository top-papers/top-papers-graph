# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import hashlib
import json
from csv import DictReader
from io import StringIO
from pathlib import Path

import pytest

from scireason.vlm_ab import cli
from scireason.vlm_ab.curator import _blank_edit, _merge_draft_edits
from scireason.vlm_ab.remediation import RemediationError, _queue_material
from scireason.vlm_ab.triage import (
    ASSISTED_REVIEW_DRAFT_FILENAME,
    TRIAGE_FILENAME,
    TRIAGE_MANIFEST_FILENAME,
    TRIAGE_SUMMARY_FILENAME,
    REVIEW_LOG_TEMPLATE_FILENAME,
    _triage_row,
    _required_actions,
    generate_triage_package,
)
from test_vlm_ab_remediation import (
    BENCHMARK_SCHEMA,
    PROVENANCE_SCHEMA,
    _generate_queue,
    _prepare_bundle,
)


def _generate(
    bundle: dict,
    queue: Path,
    output: Path,
    *,
    exclude_training_overlap: bool = True,
) -> dict:
    return generate_triage_package(
        bundle["config"],
        bundle["prepare_manifest"],
        queue / "queue_manifest.json",
        output,
        benchmark_schema=BENCHMARK_SCHEMA,
        provenance_schema=PROVENANCE_SCHEMA,
        exclude_training_overlap=exclude_training_overlap,
    )


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_triage_package_is_queue_bound_deterministic_and_minimal(tmp_path: Path) -> None:
    bundle = _prepare_bundle(tmp_path / "bundle", row_count=2, contaminated_training=True)
    queue = tmp_path / "queue"
    queue_manifest = _generate_queue(bundle, queue)
    queue_before = _files(queue)
    output = tmp_path / "triage"

    manifest = _generate(bundle, queue, output)
    first = _files(output)
    repeated = _generate(bundle, queue, output)
    records = _read_jsonl(output / TRIAGE_FILENAME)
    material = _queue_material(
        bundle["config"], bundle["prepare_manifest"], BENCHMARK_SCHEMA, PROVENANCE_SCHEMA
    )

    assert repeated == manifest
    assert _files(output) == first
    assert _files(queue) == queue_before
    assert set(first) == {
        ASSISTED_REVIEW_DRAFT_FILENAME,
        TRIAGE_FILENAME,
        TRIAGE_SUMMARY_FILENAME,
        REVIEW_LOG_TEMPLATE_FILENAME,
        TRIAGE_MANIFEST_FILENAME,
    }
    assert "completed_decisions.jsonl" not in first
    assert json.loads((output / TRIAGE_MANIFEST_FILENAME).read_text(encoding="utf-8")) == manifest
    assert manifest["queue_fingerprint"] == queue_manifest["queue_fingerprint"]
    assert manifest["artifact_version"] == 1
    assert manifest["task_count"] == len(material["tasks"])
    assert manifest["training_overlap_task_count"] == 1
    assert manifest["training_overlap_missing_identifier_task_count"] == 0
    assert manifest["overlap_paper_ids"] == ["doi:10.5555/remediation.0"]
    assert manifest["policy"] == {
        "exclude_training_overlap": True,
        "overlap_proposal": "exclude_pending_human_attestation",
        "nonoverlap_proposal": "manual_review_required",
        "automatic_retain": False,
    }
    assert [entry["path"] for entry in manifest["files"]] == [
        ASSISTED_REVIEW_DRAFT_FILENAME,
        REVIEW_LOG_TEMPLATE_FILENAME,
        TRIAGE_FILENAME,
        TRIAGE_SUMMARY_FILENAME,
    ]
    for entry in manifest["files"]:
        payload = first[entry["path"]]
        assert len(payload) == entry["size_bytes"]
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]

    expected_fields = {
        "artifact_version",
        "queue_fingerprint",
        "task_id",
        "source_row_index",
        "proposed_disposition",
        "training_overlap_detected",
        "training_overlap_identifier_status",
        "training_overlap_paper_ids",
        "critical_codes",
        "warning_codes",
        "required_actions",
        "explanation",
    }
    assert len(records) == len(material["tasks"])
    assert all(set(record) == expected_fields for record in records)
    assert [record["task_id"] for record in records] == [
        task["task_id"] for task in material["tasks"]
    ]
    assert [record["source_row_index"] for record in records] == [
        task["source_row_index"] for task in material["tasks"]
    ]
    assert all(
        record["required_actions"] == sorted(set(record["required_actions"])) for record in records
    )
    for record in records:
        assert record["required_actions"] == _required_actions(
            record["critical_codes"],
            record["warning_codes"],
            has_training_overlap=record["training_overlap_detected"],
        )
        expected_disposition = (
            "exclude_pending_human_attestation"
            if record["training_overlap_detected"]
            else "manual_review_required"
        )
        assert record["proposed_disposition"] == expected_disposition

    triage_text = (output / TRIAGE_FILENAME).read_text(encoding="utf-8")
    assert "original_row" not in triage_text
    assert "legacy_provenance_rows" not in triage_text
    assert bundle["rows"][0]["model_task_prompt"] not in triage_text
    assert bundle["rows"][0]["reference_answer"] not in triage_text
    summary = (output / TRIAGE_SUMMARY_FILENAME).read_text(encoding="utf-8")
    assert "В пакете нет завершенных решений" in summary
    assert "Ни один `retain` не создается автоматически" in summary
    assert "Группы по required actions" in summary
    review_rows = list(
        DictReader(StringIO((output / REVIEW_LOG_TEMPLATE_FILENAME).read_text(encoding="utf-8")))
    )
    assert len(review_rows) == len(material["tasks"])
    assert all(not row["reviewer_1_id"] and not row["reviewer_2_id"] for row in review_rows)
    assert (
        sum(row["machine_proposal"] == "exclude_pending_human_attestation" for row in review_rows)
        == 1
    )


def test_idempotent_triage_regeneration_rejects_workspace_tampering(tmp_path: Path) -> None:
    bundle = _prepare_bundle(tmp_path / "bundle", contaminated_training=True)
    queue = tmp_path / "queue"
    _generate_queue(bundle, queue)
    output = tmp_path / "triage"
    _generate(bundle, queue, output)
    draft_path = output / ASSISTED_REVIEW_DRAFT_FILENAME
    draft_path.write_bytes(draft_path.read_bytes() + b"tampered")

    with pytest.raises(RemediationError, match="different triage workspace"):
        _generate(bundle, queue, output)


def test_assisted_draft_prefills_only_overlap_and_merges_safely(tmp_path: Path) -> None:
    bundle = _prepare_bundle(tmp_path / "bundle", row_count=2, contaminated_training=True)
    queue = tmp_path / "queue"
    queue_manifest = _generate_queue(bundle, queue)
    output = tmp_path / "triage"
    _generate(bundle, queue, output)
    draft = json.loads((output / ASSISTED_REVIEW_DRAFT_FILENAME).read_text(encoding="utf-8"))
    material = _queue_material(
        bundle["config"], bundle["prepare_manifest"], BENCHMARK_SCHEMA, PROVENANCE_SCHEMA
    )

    assert draft["artifact_version"] == 1
    assert draft["queue_fingerprint"] == queue_manifest["queue_fingerprint"]
    assert set(draft["edits"]) == {task["task_id"] for task in material["tasks"]}
    overlap_count = 0
    for task in material["tasks"]:
        edit = draft["edits"][task["task_id"]]
        blank = _blank_edit()
        if task["training_overlap_paper_ids"] or "training_paper_overlap" in task["critical_codes"]:
            overlap_count += 1
            assert edit["disposition"] == "exclude"
            assert edit["exclusion_reason"]
            assert "заранее выбранной политике" in edit["exclusion_reason"]
            assert all(
                paper_id in edit["exclusion_reason"]
                for paper_id in task["training_overlap_paper_ids"]
            )
            assert edit["reviewed_by"] == ""
            assert edit["adjudicators"] == ""
            assert edit["independent_attestation"] is False
            assert all(
                edit[field] == blank[field]
                for field in blank
                if field not in {"disposition", "exclusion_reason"}
            )
        else:
            assert edit == blank
    assert overlap_count == 1

    current = {task["task_id"]: _blank_edit() for task in material["tasks"]}
    merged, summary = _merge_draft_edits(
        current,
        draft,
        set(current),
        queue_manifest["queue_fingerprint"],
    )

    assert summary == {"merged": 1, "equal": 0, "blank_ignored": len(current) - 1}
    assert current == {task_id: _blank_edit() for task_id in current}
    assert sum(edit["disposition"] == "exclude" for edit in merged.values()) == 1


def test_training_overlap_audit_code_without_extracted_id_is_still_excluded() -> None:
    row = _triage_row(
        {
            "task_id": "task-code-only",
            "source_row_index": 7,
            "training_overlap_paper_ids": [],
            "critical_codes": ["training_paper_overlap"],
            "warning_codes": [],
        },
        "f" * 64,
    )

    assert row["training_overlap_detected"] is True
    assert row["training_overlap_identifier_status"] == "audit_code_without_extracted_canonical_id"
    assert row["proposed_disposition"] == "exclude_pending_human_attestation"
    assert "confirm_training_overlap_exclusion" in row["required_actions"]


def test_triage_requires_explicit_exclusion_policy(tmp_path: Path) -> None:
    bundle = _prepare_bundle(tmp_path / "bundle", contaminated_training=True)
    queue = tmp_path / "queue"
    _generate_queue(bundle, queue)
    output = tmp_path / "triage"

    with pytest.raises(RemediationError, match="explicitly true"):
        _generate(bundle, queue, output, exclude_training_overlap=False)

    assert not output.exists()


@pytest.mark.parametrize("location", ["same", "inside", "contains"])
def test_triage_rejects_queue_related_output(tmp_path: Path, location: str) -> None:
    bundle = _prepare_bundle(tmp_path / "bundle")
    queue = tmp_path / "container" / "queue"
    _generate_queue(bundle, queue)
    output = {
        "same": queue,
        "inside": queue / "triage",
        "contains": queue.parent,
    }[location]
    before = _files(queue)

    with pytest.raises(RemediationError, match="separate from"):
        _generate(bundle, queue, output)

    assert _files(queue) == before


def test_triage_rejects_tampered_queue(tmp_path: Path) -> None:
    bundle = _prepare_bundle(tmp_path / "bundle", contaminated_training=True)
    queue = tmp_path / "queue"
    _generate_queue(bundle, queue)
    (queue / "tasks.jsonl").write_bytes((queue / "tasks.jsonl").read_bytes() + b"\n")

    with pytest.raises(RemediationError, match="tampered"):
        _generate(bundle, queue, tmp_path / "triage")

    assert not (tmp_path / "triage").exists()


def test_triage_rejects_output_inside_prepare_inputs(tmp_path: Path) -> None:
    bundle = _prepare_bundle(tmp_path / "bundle", contaminated_training=True)
    queue = tmp_path / "queue"
    _generate_queue(bundle, queue)
    dataset_root = Path(bundle["prepared"]["dataset_root"])
    output = dataset_root / "nested" / "triage"

    with pytest.raises(RemediationError, match="immutable input workspaces"):
        _generate(bundle, queue, output)

    assert not (dataset_root / "nested").exists()


def test_required_action_mapping_is_deterministic_and_preserves_unknown_codes() -> None:
    assert _required_actions(
        ["duplicate_sample_id", "provenance_mismatch", "unknown_code"],
        ["likely_answer_leakage"],
        has_training_overlap=True,
    ) == [
        "confirm_training_overlap_exclusion",
        "create_complete_provenance",
        "manual_investigation:unknown_code",
        "remove_answer_leakage",
        "resolve_duplicate_id",
    ]
    assert _required_actions([], [], has_training_overlap=False) == ["human_review_other"]


def test_curate_triage_cli_parser_and_routing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("unused: true\n", encoding="utf-8")
    captured: dict = {}

    monkeypatch.setattr(cli, "load_experiment_config", lambda _path: {"loaded": True})
    monkeypatch.setattr(cli, "_repo_root", lambda _path, _root: tmp_path)

    def fake_generate(*args, **kwargs) -> dict:
        captured.update(args=args, kwargs=kwargs)
        return {"task_count": 2}

    monkeypatch.setattr(cli, "generate_triage_package", fake_generate)
    code = cli.main(
        [
            "--config",
            str(config_path),
            "curate-triage",
            "--prepare-manifest",
            "prepare_manifest.json",
            "--queue-manifest",
            "queue/queue_manifest.json",
            "--output-dir",
            "triage-workspace",
            "--exclude-training-overlap",
        ]
    )

    assert code == 0
    assert captured["args"][0] == {"loaded": True}
    assert captured["kwargs"]["exclude_training_overlap"] is True
    assert captured["kwargs"]["benchmark_schema"].name == "publication_benchmark_row.schema.json"
    with pytest.raises(SystemExit):
        cli._parser().parse_args(
            [
                "--config",
                str(config_path),
                "curate-triage",
                "--prepare-manifest",
                "prepare_manifest.json",
                "--queue-manifest",
                "queue/queue_manifest.json",
                "--output-dir",
                "triage-workspace",
            ]
        )
