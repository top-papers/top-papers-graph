# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments/vlm_ab_evaluation/make_capacity_enrichment_handoff.py"
AUDIT = REPO_ROOT / "experiments/vlm_ab_evaluation/capacity_source_probe_audit_v2.json"
PRODUCTION_DOSSIERS = (
    REPO_ROOT
    / "runs/vlm_ab/qwen3vl-cap150-remediation-working-v1/capacity_machine_assist_v4"
    / "capacity_candidate_dossiers.jsonl"
)
PRODUCTION_ASSIST_MANIFEST = PRODUCTION_DOSSIERS.with_name("capacity_assist_manifest.json")


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("capacity_enrichment_handoff", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load capacity handoff generator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


handoff = _load_module()


def _audit() -> dict[str, Any]:
    return json.loads(AUDIT.read_text(encoding="ascii"))


def _audit_action_rows(audit: dict[str, Any] | None = None) -> list[int]:
    value = audit or _audit()
    rows = {exception["row"] for exception in value["identity_exceptions"]}
    rows.update(
        probe["row"]
        for probe in value["source_probes"]
        if probe["dossier_canonical_paper_id"] != probe["verified_canonical_paper_id"]
    )
    return sorted(rows)


def _expected_identity_action_rows(
    dossiers: list[dict[str, Any]], audit: dict[str, Any] | None = None
) -> list[int]:
    rows = set(_audit_action_rows(audit))
    canonical_id_rows: dict[str, list[int]] = {}
    for row_number, dossier in enumerate(dossiers, start=1):
        canonical_id = dossier["canonical_paper_id"]
        if not handoff._is_canonical_paper_id(canonical_id):
            rows.add(row_number)
        else:
            canonical_id_rows.setdefault(canonical_id, []).append(row_number)
    for duplicate_rows in canonical_id_rows.values():
        if len(duplicate_rows) > 1:
            rows.update(duplicate_rows)
    return sorted(rows)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="ascii",
    )


def _form_data(path: Path) -> dict[str, Any]:
    html = path.read_text(encoding="utf-8")
    match = re.search(
        r'<script nonce="[^"]+" type="application/json" id="form-data">(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert match is not None
    return json.loads(match.group(1))


def _inline_javascript(path: Path) -> str:
    html = path.read_text(encoding="utf-8")
    scripts = []
    for match in re.finditer(r"<script(?P<attrs>[^>]*)>(?P<body>.*?)</script>", html, re.DOTALL):
        if 'type="application/json"' not in match.group("attrs"):
            scripts.append(match.group("body"))
    assert scripts
    return "\n".join(scripts)


def _node_prelude(data: dict[str, Any]) -> str:
    form_data = json.dumps(
        json.dumps(data, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    )
    return "\n".join(
        [
            '"use strict";',
            'const assert=require("node:assert/strict");',
            "class FakeElement {",
            "  constructor(tag){",
            "    this.tag=tag;this.listeners={};this.files=[];this.value='';this.textContent='';",
            "    this.children=[];this.scrollIntoViewCalls=[];",
            "    this.classList={add(){},remove(){}};",
            "  }",
            "  append(...children){this.children.push(...children)}",
            "  replaceChildren(...children){this.children=[...children]}",
            "  addEventListener(name,listener){this.listeners[name]=listener}",
            "  removeAttribute(){}",
            "  remove(){}",
            "  click(){}",
            "  focus(){}",
            "  scrollIntoView(options){this.scrollIntoViewCalls.push(options)}",
            "}",
            f"const __FORM_DATA={form_data};",
            "const __ELEMENTS=new Map();",
            "global.document={",
            "  body:new FakeElement('body'),",
            "  createElement:tag=>new FakeElement(tag),",
            "  getElementById:id=>{",
            "    if(id==='form-data')return {textContent:__FORM_DATA};",
            "    if(!__ELEMENTS.has(id))__ELEMENTS.set(id,new FakeElement(id));",
            "    return __ELEMENTS.get(id);",
            "  },",
            "};",
        ]
    )


def _run_node_script(tmp_path: Path, name: str, source: str) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = tmp_path / name
    script.write_text(source, encoding="utf-8")
    checked = subprocess.run(
        [node, str(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert checked.returncode == 0, checked.stderr


def _synthetic_dossiers(tmp_path: Path) -> Path:
    audit = _audit()
    probes = {row["row"]: row for row in audit["source_probes"]}
    exceptions = {row["row"]: row for row in audit["identity_exceptions"]}
    rows = []
    for row_number in range(1, 151):
        probe = probes.get(row_number)
        exception = exceptions.get(row_number)
        group_id = (
            probe["group_id"]
            if probe is not None
            else exception["group_id"]
            if exception is not None
            else hashlib.sha256(f"group-{row_number}".encode()).hexdigest()
        )
        canonical_id = (
            probe["dossier_canonical_paper_id"]
            if probe is not None
            else exception["current_canonical_paper_id"]
            if exception is not None
            else f"doi:10.5555/synthetic.{row_number}"
        )
        task_id = f"task_{hashlib.sha256(f'task-{row_number}'.encode()).hexdigest()}"
        legacy_sha256 = hashlib.sha256(f"legacy-image-{row_number}".encode()).hexdigest()
        rows.append(
            {
                "artifact_version": 2,
                "queue_fingerprint": audit["queue_fingerprint"],
                "group_id": group_id,
                "canonical_paper_id": canonical_id,
                "capacity_role": "required_candidate_for_exact_150",
                "identity_status": "clean",
                "task_ids": [task_id],
                "primary_task_ids": [task_id],
                "tasks": [
                    {
                        "binding": {
                            "task_id": task_id,
                            "source_row_index": row_number - 1,
                        },
                        "machine_retrieval_url": (f"https://example.test/source/{row_number}"),
                        "source_row_hint": {
                            "stratum": "multimodal_hard",
                            "primary_endpoint": True,
                            "legacy_prompt_text_included": False,
                        },
                        "audited_image_hints": [
                            {
                                "sha256": legacy_sha256,
                                "release_usable": False,
                                "evidence_status": "quarantined_cross_paper_reuse",
                            }
                        ],
                        "critical_codes": ["cross_paper_image_reuse"],
                        "warning_codes": [],
                        "required_actions": ["replace_quarantined_image_bytes"],
                    }
                ],
                "machine_selected_task_ids": [],
                "candidate_status": "blocked_pending_enrichment_and_human_attestation",
                "human_review": {"curator_ids": [], "independent_attestation": False},
            }
        )
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    path = inputs / "dossiers.jsonl"
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="ascii",
    )
    return path


def _write_audit(tmp_path: Path, value: dict[str, Any]) -> Path:
    inputs = tmp_path / "inputs"
    inputs.mkdir(exist_ok=True)
    path = inputs / "audit.json"
    path.write_text(
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="ascii",
    )
    return path


def test_production_audit_is_ascii_and_matches_current_dossiers(tmp_path: Path) -> None:
    assert all(byte < 128 for byte in AUDIT.read_bytes())
    if not PRODUCTION_DOSSIERS.is_file() or not PRODUCTION_ASSIST_MANIFEST.is_file():
        pytest.skip("production capacity-assist workspace is not present")

    output = tmp_path / "handoff"
    manifest = handoff.generate_handoff(PRODUCTION_DOSSIERS, AUDIT, output)
    assist_manifest = json.loads(PRODUCTION_ASSIST_MANIFEST.read_text(encoding="utf-8"))

    assert manifest["queue_fingerprint"] == assist_manifest["queue_fingerprint"]
    assert (
        manifest["source_inputs"]["dossiers"]["sha256"]
        == hashlib.sha256(PRODUCTION_DOSSIERS.read_bytes()).hexdigest()
    )
    assert (
        manifest["source_inputs"]["source_audit"]["sha256"]
        == hashlib.sha256(AUDIT.read_bytes()).hexdigest()
    )

    source = _form_data(output / "source_submission_form.html")
    identity = _form_data(output / "identity_resolution_form.html")
    requirements = _jsonl(output / "enrichment_requirements.jsonl")
    with (output / "identity_resolution_template.csv").open(encoding="utf-8", newline="") as handle:
        identity_csv = list(csv.DictReader(handle))

    actions = source["identity_actions"]
    action_rows = [action["row"] for action in actions]
    assert action_rows == sorted(action_rows)
    assert {28, 59}.issubset(action_rows)
    assert len(actions) == len(identity["actions"]) == len(identity_csv) == 15
    assert identity["actions"] == actions
    assert manifest["counts"]["identity_action_count"] == len(actions)
    assert manifest["counts"]["unresolved_empty_dossier_id_action_count"] == 2
    assert manifest["counts"]["noncanonical_dossier_id_action_count"] == 0
    assert manifest["counts"]["duplicate_dossier_id_action_count"] == 0
    assert {
        action["row"]
        for action in actions
        if action["action_type"] == "unresolved_empty_dossier_id"
    } == {28, 59}
    assert all(
        action["machine_candidate_canonical_paper_id"] is None
        and "no machine candidate is available" in action["machine_reason"]
        for action in actions
        if action["action_type"] == "unresolved_empty_dossier_id"
    )
    for action, csv_row in zip(actions, identity_csv, strict=True):
        assert csv_row == {
            "row": str(action["row"]),
            "group_id": action["group_id"],
            "action_type": action["action_type"],
            "current_canonical_paper_id": action["current_canonical_paper_id"],
            "machine_candidate_canonical_paper_id": action["machine_candidate_canonical_paper_id"]
            or "",
            "partition_category": action["partition_category"],
            "status": action["machine_status"],
            "machine_reason": action["machine_reason"],
            "resolution_type": "",
            "resolved_canonical_paper_id": "",
            "resolution_evidence_url": "",
            "resolution_evidence_sha256": "",
            "resolution_evidence_retrieved_at": "",
            "capacity_plan_regenerated": "",
            "resolution_notes": "",
        }
    action_row_set = set(action_rows)
    assert all(
        source["initial_submissions"][row - 1]["canonical_paper_id"] == "" for row in action_rows
    )
    assert all(
        source["groups"][row - 1]["requires_identity_resolution"] is True for row in action_rows
    )
    assert all(
        requirement["requires_identity_resolution"]
        and "identity_resolution" in requirement["missing_inputs"]
        for requirement in requirements
        if requirement["row"] in action_row_set
    )
    assert len(identity["initial_resolutions"]) == len(actions)
    assert all(
        not resolution["resolved_canonical_paper_id"]
        for resolution in identity["initial_resolutions"]
    )
    readme = (output / "README_RU.md").read_text(encoding="utf-8")
    assert f"все {len(actions)} identity actions" in readme

    if shutil.which("node") is not None:
        source_prefix, source_marker, _ = handoff._SOURCE_JS.partition(
            '$("lineageUrl").addEventListener'
        )
        assert source_marker
        _run_node_script(
            tmp_path,
            "production-source-identity-equality.js",
            "\n".join(
                [
                    _node_prelude(source),
                    handoff._FORM_SHARED_JS,
                    source_prefix,
                    r"""
const importedIdentity={
  kind:"capacity_identity_resolution_submission",
  artifact_version:1,
  queue_fingerprint:BOOT.queue_fingerprint,
  actions:BOOT.identity_actions.map(action=>({
    row:action.row,
    group_id:action.group_id,
    resolution_type:"replace_with_other",
    resolved_canonical_paper_id:`paper:production-resolution-${action.row}`,
    evidence:{
      source_url:`https://example.test/identity/${action.row}`,
      sha256:"d".repeat(64),
      retrieved_at:"2026-07-23T12:00:00Z",
      notes:"Проверено владельцем данных",
    },
    capacity_plan_regenerated:true,
  })),
  policy:clone(BOOT.safe_submission_policy),
  publication_ready:false,
};
assert.deepEqual(identityErrors(importedIdentity),[]);
bindings.identity={value:importedIdentity,sha256:"e".repeat(64)};
const importedByRow=new Map(importedIdentity.actions.map(action=>[action.row,action]));
for(const submission of state.submissions){
  const resolution=importedByRow.get(submission.row);
  if(resolution)submission.canonical_paper_id=resolution.resolved_canonical_paper_id;
}
const dynamicRows=new Set(BOOT.identity_actions.map(action=>action.row));
assert.deepEqual(validateSource().filter(error=>
  dynamicRows.has(error.row)&&error.message.includes("canonical_paper_id")
),[]);
""",
                ]
            ),
        )


def test_handoff_is_deterministic_idempotent_blocked_and_exact(tmp_path: Path) -> None:
    dossiers = _synthetic_dossiers(tmp_path)
    output = tmp_path / "handoff"

    manifest = handoff.generate_handoff(dossiers, AUDIT, output)
    first = _files(output)
    repeated = handoff.generate_handoff(dossiers, AUDIT, output)
    second_output = tmp_path / "handoff-copy"
    handoff.generate_handoff(dossiers, AUDIT, second_output)
    requirements = _jsonl(output / "enrichment_requirements.jsonl")
    identity_actions = _form_data(output / "identity_resolution_form.html")["actions"]

    assert repeated == manifest
    assert _files(output) == first == _files(second_output)
    assert set(first) == {
        "README_RU.md",
        "enrichment_requirements.jsonl",
        "source_submission_template.csv",
        "identity_resolution_template.csv",
        "source_submission_form.html",
        "identity_resolution_form.html",
        "training_lineage_form.html",
        "handoff_manifest.json",
    }
    assert "machine_enrichment_v2.jsonl" not in first
    assert "machine_assisted_draft.json" not in first
    assert manifest["publication_ready"] is False
    assert manifest["file_count"] == 7
    assert manifest["policy"]["network_retrieval_performed_by_generator"] is False
    assert manifest["policy"]["source_probe_network_audit_present"] is True
    assert manifest["policy"]["direct_image_bytes_sha256_machine_checked"] is True
    expected_action_rows = _expected_identity_action_rows(_jsonl(dossiers))
    assert manifest["counts"] == {
        "total_rows": 150,
        "source_probe_count": 48,
        "matching_current_source_probe_count": 44,
        "identity_correction_source_probe_count": 4,
        "metadata_only_or_unlicensed_count": 91,
        "unresolved_identity_count": 5,
        "duplicate_candidate_count": 1,
        "no_retrieval_hint_count": 5,
        "identity_exception_count": 9,
        "unresolved_empty_dossier_id_action_count": sum(
            action["action_type"] == "unresolved_empty_dossier_id" for action in identity_actions
        ),
        "noncanonical_dossier_id_action_count": sum(
            action["action_type"] == "noncanonical_dossier_id" for action in identity_actions
        ),
        "duplicate_dossier_id_action_count": sum(
            action["action_type"] == "duplicate_dossier_id" for action in identity_actions
        ),
        "identity_action_count": len(expected_action_rows),
    }
    assert [action["row"] for action in identity_actions] == expected_action_rows
    assert len(requirements) == 150
    audit = _audit()
    assert {
        row["row"]
        for row in requirements
        if row["status"] == "identity_correction_requires_regeneration"
    } == {
        probe["row"]
        for probe in audit["source_probes"]
        if probe["dossier_canonical_paper_id"] != probe["verified_canonical_paper_id"]
    }
    assert {
        row["row"] for row in requirements if row["status"] == "blocked_identity_exception"
    } == {exception["row"] for exception in audit["identity_exceptions"]}
    assert all(
        row["publication_ready"] is False
        and row["human_verified"] is False
        and row["automatic_retain"] is False
        and row["missing_inputs"]
        and "full_enrichment_validation" in row["missing_inputs"]
        for row in requirements
    )
    assert not any(
        row["status"] in {"complete", "release_ready", "publication_ready"} for row in requirements
    )
    for record in manifest["files"]:
        payload = first[record["path"]]
        assert record["size_bytes"] == len(payload)
        assert record["sha256"] == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("identity_case", ["noncanonical", "duplicate"])
def test_invalid_dossier_ids_become_dynamic_identity_actions(
    tmp_path: Path, identity_case: str
) -> None:
    dossiers = _synthetic_dossiers(tmp_path)
    rows = _jsonl(dossiers)
    audit = _audit()
    audited_rows = {
        record["row"] for key in ("source_probes", "identity_exceptions") for record in audit[key]
    }
    available_rows = [row for row in range(1, 151) if row not in audited_rows]
    if identity_case == "noncanonical":
        target_rows = available_rows[:1]
        dossier_id = "DOI:10.5555/noncanonical"
        action_type = "noncanonical_dossier_id"
    else:
        target_rows = available_rows[:2]
        dossier_id = "paper:synthetic-duplicate"
        action_type = "duplicate_dossier_id"
    for row in target_rows:
        rows[row - 1]["canonical_paper_id"] = dossier_id
    _write_jsonl(dossiers, rows)

    output = tmp_path / "handoff"
    manifest = handoff.generate_handoff(dossiers, AUDIT, output)
    source = _form_data(output / "source_submission_form.html")
    identity = _form_data(output / "identity_resolution_form.html")
    requirements = _jsonl(output / "enrichment_requirements.jsonl")
    actions_by_row = {action["row"]: action for action in identity["actions"]}

    assert [action["row"] for action in identity["actions"]] == (
        _expected_identity_action_rows(rows, audit)
    )
    assert all(actions_by_row[row]["action_type"] == action_type for row in target_rows)
    assert all(
        actions_by_row[row]["machine_candidate_canonical_paper_id"] is None for row in target_rows
    )
    assert all(
        source["initial_submissions"][row - 1]["canonical_paper_id"] == ""
        and source["groups"][row - 1]["requires_identity_resolution"] is True
        and "canonical_paper_id" not in source["groups"][row - 1]["machine_prefilled_fields"]
        and requirements[row - 1]["requires_identity_resolution"] is True
        for row in target_rows
    )
    assert dossier_id not in identity["occupied_non_action_paper_ids"]
    assert manifest["counts"][f"{action_type}_action_count"] == len(target_rows)


def test_templates_do_not_prefill_human_state_or_decisions(tmp_path: Path) -> None:
    dossiers = _synthetic_dossiers(tmp_path)
    output = tmp_path / "handoff"
    handoff.generate_handoff(dossiers, AUDIT, output)

    with (output / "source_submission_template.csv").open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    with (output / "identity_resolution_template.csv").open(encoding="utf-8", newline="") as handle:
        identity_rows = list(csv.DictReader(handle))
    expected_action_rows = _expected_identity_action_rows(_jsonl(dossiers))

    forbidden = ("curator", "attestation", "retain", "decision")
    assert len(source_rows) == 150
    assert len(identity_rows) == len(expected_action_rows)
    assert not any(part in field.lower() for field in source_rows[0] for part in forbidden)
    assert not any(part in field.lower() for field in identity_rows[0] for part in forbidden)
    assert all(not row["selected_task_ids"] for row in source_rows)
    assert all(not row["resolved_canonical_paper_id"] for row in identity_rows)
    assert all(not row["resolution_evidence_url"] for row in identity_rows)
    assert all(not row["resolution_evidence_sha256"] for row in identity_rows)
    assert all(not row["capacity_plan_regenerated"] for row in identity_rows)
    assert [int(row["row"]) for row in identity_rows] == expected_action_rows
    assert sum(row["action_type"] == "identity_correction_probe" for row in identity_rows) == 4
    assert sum(row["action_type"] == "identity_exception" for row in identity_rows) == 9
    readme = (output / "README_RU.md").read_text(encoding="utf-8")
    assert "НЕ импортируется в curator" in readme
    assert "machine_assisted_draft.json" in readme
    assert "полного\nenrichment validation" in readme


def test_offline_forms_embed_exact_safe_inventories(tmp_path: Path) -> None:
    dossiers = _synthetic_dossiers(tmp_path)
    output = tmp_path / "handoff"
    handoff.generate_handoff(dossiers, AUDIT, output)

    source = _form_data(output / "source_submission_form.html")
    identity = _form_data(output / "identity_resolution_form.html")
    lineage = _form_data(output / "training_lineage_form.html")
    expected_action_rows = _expected_identity_action_rows(_jsonl(dossiers))

    assert source["form_kind"] == "capacity_source_submission_form"
    assert len(source["groups"]) == len(source["initial_submissions"]) == 150
    assert all(not row["selected_task_ids"] for row in source["initial_submissions"])
    assert all(group["task_summaries"] for group in source["groups"])
    assert all(
        [task["task_id"] for task in group["task_summaries"]] == group["primary_task_ids"]
        for group in source["groups"]
    )
    assert all(
        set(task)
        == {
            "task_id",
            "source_row_index",
            "stratum",
            "primary_endpoint",
            "machine_retrieval_url",
            "critical_codes",
            "warning_codes",
            "required_action_codes",
            "is_primary",
            "machine_only",
        }
        for group in source["groups"]
        for task in group["task_summaries"]
    )
    assert len(source["quarantined_legacy_sha256s"]) == 150
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", digest) for digest in source["quarantined_legacy_sha256s"]
    )
    assert "legacy_prompt" not in json.dumps(source, sort_keys=True)

    audit = _audit()
    probes = {probe["row"]: probe for probe in audit["source_probes"]}
    assert {group["row"] for group in source["groups"] if group["machine_prefilled_fields"]} == set(
        probes
    )
    probe = probes[min(probes)]
    group = source["groups"][probe["row"] - 1]
    submission = source["initial_submissions"][probe["row"] - 1]
    assert group["machine_prefilled_fields"] == [
        "canonical_paper_id",
        "images[0].sha256",
        "images[0].locator",
        "images[0].source_url",
        "images[0].license",
        "images[0].license_url",
        "article_evidence.source_url",
    ]
    assert submission["canonical_paper_id"] == probe["verified_canonical_paper_id"]
    assert submission["images"] == [
        {
            "image_path": "",
            "sha256": probe["direct_image_sha256"],
            "page": "",
            "locator": probe["citation_locator"],
            "source_url": probe["direct_image_url"],
            "license": probe["license"],
            "license_url": probe["license_url"],
            "citation": "",
            "retrieved_at": "",
        }
    ]
    assert submission["article_evidence"] == {
        "source_url": probe["metadata_url"],
        "sha256": "",
        "retrieved_at": "",
    }
    assert submission["metadata_evidence"] == {
        "source_url": "",
        "sha256": "",
        "retrieved_at": "",
    }

    action_rows = {action["row"] for action in source["identity_actions"]}
    correction_rows = handoff.CORRECTION_ROWS
    for row in action_rows:
        action_group = source["groups"][row - 1]
        action_submission = source["initial_submissions"][row - 1]
        assert action_submission["canonical_paper_id"] == ""
        assert "canonical_paper_id" not in action_group["machine_prefilled_fields"]
    for row in correction_rows:
        correction_probe = probes[row]
        correction_group = source["groups"][row - 1]
        correction_submission = source["initial_submissions"][row - 1]
        assert correction_group["machine_prefilled_fields"] == [
            "images[0].sha256",
            "images[0].locator",
            "images[0].source_url",
            "images[0].license",
            "images[0].license_url",
            "article_evidence.source_url",
        ]
        assert (
            correction_submission["images"][0]["sha256"] == correction_probe["direct_image_sha256"]
        )
        assert (
            correction_submission["images"][0]["source_url"] == correction_probe["direct_image_url"]
        )
        assert (
            correction_submission["article_evidence"]["source_url"]
            == correction_probe["metadata_url"]
        )

    assert identity["form_kind"] == "capacity_identity_resolution_form"
    assert source["identity_actions"] == identity["actions"]
    assert len(identity["actions"]) == len(identity["initial_resolutions"])
    assert len(identity["actions"]) == len(expected_action_rows)
    assert all(not row["resolution_type"] for row in identity["initial_resolutions"])
    assert {
        action["row"]
        for action in identity["actions"]
        if action["action_type"] == "identity_correction_probe"
    } == {26, 52, 78, 114}
    assert {
        action["row"]
        for action in identity["actions"]
        if action["action_type"] == "identity_exception"
    } == {exception["row"] for exception in _audit()["identity_exceptions"]}
    assert all(action["machine_only"] is True for action in identity["actions"])
    action_rows = {action["row"] for action in identity["actions"]}
    assert identity["occupied_non_action_paper_ids"] == sorted(
        group["dossier_canonical_paper_id"]
        for group in source["groups"]
        if group["row"] not in action_rows and group["dossier_canonical_paper_id"]
    )

    expected_policy = {
        "data_owner_proposal_only": True,
        "curator_importable": False,
        "automatic_selection": False,
        "external_verification_claimed": False,
        "human_review_included": False,
        "independent_attestation_included": False,
        "retain_exclude_decisions_included": False,
        "full_downstream_validation_required": True,
        "publication_ready": False,
    }
    assert source["safe_submission_policy"] == expected_policy
    assert identity["safe_submission_policy"] == expected_policy
    assert lineage["export_contract"]["filename"] == "training_lineage_manifest.json"
    assert lineage["initial_state"]["coverage"] == {
        "paper_ids": False,
        "source_documents": False,
        "creator_groups": False,
        "image_bytes": False,
        "prompts": False,
    }


def test_source_form_exposes_and_accepts_only_primary_tasks(tmp_path: Path) -> None:
    dossiers = _synthetic_dossiers(tmp_path)
    rows = _jsonl(dossiers)
    secondary_id = f"task_{hashlib.sha256(b'secondary-task').hexdigest()}"
    secondary = json.loads(json.dumps(rows[0]["tasks"][0]))
    secondary["binding"] = {"task_id": secondary_id, "source_row_index": 150}
    secondary["machine_retrieval_url"] = "https://example.test/source/secondary"
    secondary["source_row_hint"]["primary_endpoint"] = False
    secondary["audited_image_hints"] = []
    secondary["critical_codes"] = []
    secondary["warning_codes"] = []
    secondary["required_actions"] = []
    rows[0]["task_ids"].append(secondary_id)
    rows[0]["tasks"].append(secondary)
    _write_jsonl(dossiers, rows)

    output = tmp_path / "handoff"
    handoff.generate_handoff(dossiers, AUDIT, output)
    source = _form_data(output / "source_submission_form.html")
    group = source["groups"][0]
    script = _inline_javascript(output / "source_submission_form.html")

    assert group["task_ids"] == [group["primary_task_ids"][0], secondary_id]
    assert [task["task_id"] for task in group["task_summaries"]] == group["primary_task_ids"]
    assert secondary_id not in json.dumps(group["task_summaries"], sort_keys=True)
    assert "expected.primary_task_ids.includes" in script
    assert "task=>task.is_primary&&requirement.primary_task_ids.includes" in script


def test_all_release_unusable_hashes_are_forbidden_and_reuse_needs_hints(
    tmp_path: Path,
) -> None:
    dossiers = _synthetic_dossiers(tmp_path)
    rows = _jsonl(dossiers)
    image_hint = rows[0]["tasks"][0]["audited_image_hints"][0]
    legacy_sha256 = image_hint["sha256"]
    image_hint["evidence_status"] = "pending_external_source_verification"
    rows[0]["tasks"][0]["critical_codes"] = []
    _write_jsonl(dossiers, rows)

    output = tmp_path / "handoff"
    handoff.generate_handoff(dossiers, AUDIT, output)
    source = _form_data(output / "source_submission_form.html")
    assert legacy_sha256 in source["quarantined_legacy_sha256s"]

    rows[0]["tasks"][0]["critical_codes"] = ["cross_paper_image_reuse"]
    rows[0]["tasks"][0]["audited_image_hints"] = []
    _write_jsonl(dossiers, rows)
    with pytest.raises(handoff.HandoffError, match="cross-paper reuse has no image hints"):
        handoff.generate_handoff(dossiers, AUDIT, tmp_path / "invalid-handoff")


def test_forms_are_strictly_offline_and_have_required_browser_support(tmp_path: Path) -> None:
    dossiers = _synthetic_dossiers(tmp_path)
    output = tmp_path / "handoff"
    handoff.generate_handoff(dossiers, AUDIT, output)

    for name in (
        "source_submission_form.html",
        "identity_resolution_form.html",
        "training_lineage_form.html",
    ):
        html = (output / name).read_text(encoding="utf-8")
        assert '<html lang="ru">' in html
        for directive in (
            "default-src 'none'",
            "connect-src 'none'",
            "object-src 'none'",
            "base-uri 'none'",
            "form-action 'none'",
            "img-src blob:",
            "style-src 'nonce-capacity-handoff-v1'",
            "script-src 'nonce-capacity-handoff-v1'",
        ):
            assert directive in html
        assert '<style nonce="capacity-handoff-v1">' in html
        assert '<script nonce="capacity-handoff-v1">' in html
        assert "localStorage" in html
        assert "crypto.subtle.digest" in html
        assert "const MAX_JSON_BYTES=32*1024*1024" in html
        assert "strictJsonParse" in html
        assert "Повторяющийся ключ JSON" in html
        assert "размер JSON превышает 32 МиБ" in html
        assert "storageGet" in html and "storageSet" in html and "storageRemove" in html
        assert "pendingHashCount" in html
        assert "runtime.token!==token" in html
        assert "draftImportToken" in html and "stateRevision" in html
        assert "draftImportCurrent(input,file,token,revision)" in html
        assert "const persisted=save()" in html
        assert "const removed=storageRemove(KEY)" in html
        assert "только в памяти" in html
        assert "Финальный экспорт заблокирован" in html
        assert "draftImport" in html and "draftExport" in html and "finalExport" in html
        assert "URL.createObjectURL" in html and "URL.revokeObjectURL" in html
        assert "fetch(" not in html
        assert "XMLHttpRequest" not in html
        assert "innerHTML" not in html
        assert "<script src=" not in html
        assert "<link " not in html
        assert "Final export blocked" not in html
        assert "Draft autosaved locally" not in html
        if name == "source_submission_form.html":
            assert "BOOT.identity_actions.length" in html
        if name == "identity_resolution_form.html":
            assert "BOOT.actions.length" in html


def test_forms_share_one_responsive_stylesheet(tmp_path: Path) -> None:
    dossiers = _synthetic_dossiers(tmp_path)
    output = tmp_path / "handoff"
    handoff.generate_handoff(dossiers, AUDIT, output)

    css = re.sub(r"\s+", "", handoff._FORM_CSS)
    required_rules = (
        "html,body{min-width:0;max-width:100%}",
        ".shell{width:100%;min-width:0}",
        ".layout,main,aside,.notice,.card,.errors,.grid,label,.toolbar,.row,.actions,.task,.image,.source{min-width:0;max-width:100%}",
        ".layout>*,.grid>*,.toolbar>*,.row>*,.actions>*,.task>*,.image>*,.source>*{min-width:0;max-width:100%}",
        "input,select,textarea,button,progress{min-width:0;max-width:100%}",
        'input[type="file"]{width:100%}',
    )
    for rule in required_rules:
        assert css.count(rule) == 1
    assert "overflow-wrap:anywhere" in css
    assert "word-break:break-word" in css
    assert "overflow-x:hidden" not in css

    mobile = css.rsplit("@media(max-width:800px)", maxsplit=1)[1]
    for rule in (
        "header{padding:.85rem.75rem}",
        ".shell{padding:.65rem}",
        ".notice,.card,.errors{padding:.75rem;margin-bottom:.75rem}",
        ".layout,.grid{grid-template-columns:minmax(0,1fr)}",
        ".sidebar{position:static;top:auto;max-height:min(55vh,28rem);overflow-y:auto}",
        ".nav-list{display:grid;grid-template-columns:minmax(0,1fr)}",
        ".actions,.row{flex-direction:column}",
        ".actions>*,.toolbar>*,.row>*{width:100%}",
        "label.check{display:grid;grid-template-columns:autominmax(0,1fr);align-items:start}",
    ):
        assert mobile.count(rule) == 1
    assert "display:none" not in mobile

    for name, editor_id in (
        ("source_submission_form.html", "groupForm"),
        ("identity_resolution_form.html", "actionForm"),
    ):
        html = (output / name).read_text(encoding="utf-8")
        assert html.index('class="card sidebar"') < html.index(f'id="{editor_id}"')

    assert {name for name in vars(handoff) if name.endswith("_CSS")} == {"_FORM_CSS"}
    style_pattern = re.compile(r'<style nonce="([^"]+)">(.*?)</style>', re.DOTALL)
    for name in (
        "source_submission_form.html",
        "identity_resolution_form.html",
        "training_lineage_form.html",
    ):
        html = (output / name).read_text(encoding="utf-8")
        assert len(re.findall(r"<style\b", html)) == 1
        assert style_pattern.findall(html) == [(handoff.HTML_NONCE, handoff._FORM_CSS)]


def test_source_and_identity_export_contracts_have_no_curator_decision_fields(
    tmp_path: Path,
) -> None:
    dossiers = _synthetic_dossiers(tmp_path)
    output = tmp_path / "handoff"
    handoff.generate_handoff(dossiers, AUDIT, output)

    forbidden = (
        "curator",
        "verified_by",
        "attestation",
        "retain",
        "exclude",
        "review",
        "decision",
    )
    for name in ("source_submission_form.html", "identity_resolution_form.html"):
        contract = _form_data(output / name)["export_contract"]
        encoded = json.dumps(contract, sort_keys=True).lower()
        assert not any(value in encoded for value in forbidden)


def test_generated_inline_javascript_parses_with_node(
    tmp_path: Path,
) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    dossiers = _synthetic_dossiers(tmp_path)
    output = tmp_path / "handoff"
    handoff.generate_handoff(dossiers, AUDIT, output)

    for name in (
        "source_submission_form.html",
        "identity_resolution_form.html",
        "training_lineage_form.html",
    ):
        script = tmp_path / f"{name}.js"
        script.write_text(_inline_javascript(output / name), encoding="utf-8")
        checked = subprocess.run(
            [node, "--check", str(script)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert checked.returncode == 0, checked.stderr


def test_browser_guards_execute_with_node(tmp_path: Path) -> None:
    dossiers = _synthetic_dossiers(tmp_path)
    output = tmp_path / "handoff"
    handoff.generate_handoff(dossiers, AUDIT, output)
    source_data = _form_data(output / "source_submission_form.html")
    identity_data = _form_data(output / "identity_resolution_form.html")
    lineage_data = _form_data(output / "training_lineage_form.html")

    shared_checks = "\n".join(
        [
            _node_prelude(source_data),
            handoff._FORM_SHARED_JS,
            r"""
assert.throws(()=>strictJsonParse('{"a":1,"a":2}'),/Повторяющийся ключ JSON/);
assert.throws(()=>strictJsonParse('{"a":NaN}'),/Ожидалось JSON-значение/);
const protoValue=strictJsonParse('{"__proto__":{"polluted":true}}');
assert.equal(Object.prototype.hasOwnProperty.call(protoValue,"__proto__"),true);
assert.equal({}.polluted,undefined);
assert.equal(canonicalPaperId("doi:10.5555/example.1"),true);
assert.equal(canonicalPaperId("arxiv:2401.01234"),true);
assert.equal(canonicalPaperId("paper:openalex:w123"),true);
assert.equal(canonicalPaperId("DOI:10.5555/example.1"),false);
assert.equal(canonicalPaperId("10.5555/example.1"),false);
assert.equal(safeAssetPath("assets/images/Figure.PNG"),true);
assert.equal(safeAssetPath("assets/images/рисунок.png"),false);
assert.equal(windowsPathKey("assets/images/Figure.PNG"),"assets/images/figure.png");
assert.equal(windowsPathKey("assets/images/Ｆigure.PNG"),"assets/images/figure.png");
async function importChecks(){
  const oversized={size:MAX_JSON_BYTES+1,arrayBuffer:async()=>new ArrayBuffer(0)};
  await assert.rejects(readJson(oversized),/32 МиБ/);
}
importChecks().catch(error=>{console.error(error.stack||error);process.exitCode=1});
""",
        ]
    )
    _run_node_script(tmp_path, "shared-browser-guards.js", shared_checks)

    source_prefix, source_marker, _ = handoff._SOURCE_JS.partition(
        '$("draftExport").addEventListener'
    )
    assert source_marker
    source_checks = "\n".join(
        [
            _node_prelude(source_data),
            handoff._FORM_SHARED_JS,
            source_prefix,
            r"""
const initialDraft=draftEnvelope();
assert.deepEqual(Object.keys(initialDraft.draft).sort(),["lineage_reference","submissions"]);
assert.equal(validDraft(initialDraft),true);
const sourceNav=$("groupNav"),groupForm=$("groupForm");
renderProgress();
assert.equal(sourceNav.children.length,150);
sourceNav.children.at(-1).listeners.click();
assert.equal(currentRow,150);
assert.deepEqual(groupForm.scrollIntoViewCalls.at(-1),{block:"start"});
let sourceScrollCount=groupForm.scrollIntoViewCalls.length;
renderErrorSummary([{row:149,message:"source navigation check"}],go);
$("errors").children.at(-1).children[0].children[0].listeners.click();
assert.equal(currentRow,149);
assert.equal(groupForm.scrollIntoViewCalls.length,sourceScrollCount+1);
sourceScrollCount=groupForm.scrollIntoViewCalls.length;
$("firstIncomplete").listeners.click();
assert.equal(currentRow,1);
assert.equal(groupForm.scrollIntoViewCalls.length,sourceScrollCount+1);
cleanupFileHashers();
bindings={
  identity:{value:{runtime_secret:"identity"},sha256:"1".repeat(64)},
  lineage:{value:{runtime_secret:"lineage"},sha256:"2".repeat(64)},
};
const serializedDraft=JSON.stringify(draftEnvelope());
assert.equal(serializedDraft.includes("runtime_secret"),false);
assert.equal(serializedDraft.includes('"identity_import"'),false);
assert.equal(serializedDraft.includes('"lineage_import"'),false);
const persistedBinding=clone(initialDraft);
persistedBinding.draft.identity_import={};
assert.equal(validDraft(persistedBinding),false);

function completeIdentity(){
  return {
    kind:"capacity_identity_resolution_submission",
    artifact_version:1,
    queue_fingerprint:BOOT.queue_fingerprint,
    actions:BOOT.identity_actions.map(action=>({
      row:action.row,
      group_id:action.group_id,
      resolution_type:"replace_with_other",
      resolved_canonical_paper_id:`paper:runtime-resolution-${action.row}`,
      evidence:{
        source_url:`https://example.test/evidence/${action.row}`,
        sha256:"a".repeat(64),
        retrieved_at:"2026-07-23T12:00:00Z",
        notes:"Проверено владельцем данных",
      },
      capacity_plan_regenerated:true,
    })),
    policy:clone(BOOT.safe_submission_policy),
    publication_ready:false,
  };
}
const validIdentity=completeIdentity();
assert.deepEqual(identityErrors(validIdentity),[]);
const duplicateIdentity=clone(validIdentity);
duplicateIdentity.actions[1].resolved_canonical_paper_id=
  duplicateIdentity.actions[0].resolved_canonical_paper_id;
assert.ok(identityErrors(duplicateIdentity).some(message=>message.includes("повторяет")));
const occupiedIdentity=clone(validIdentity);
occupiedIdentity.actions[0].resolved_canonical_paper_id=BOOT.groups.find(
  group=>!actionByRow.has(group.row)&&canonicalPaperId(group.dossier_canonical_paper_id)
).dossier_canonical_paper_id;
assert.ok(identityErrors(occupiedIdentity).some(message=>message.includes("уже занят")));
const stalePlan=clone(validIdentity);
stalePlan.actions[0].capacity_plan_regenerated=false;
assert.ok(identityErrors(stalePlan).some(message=>message.includes("capacity_plan_regenerated")));
const noncanonicalIdentity=clone(validIdentity);
noncanonicalIdentity.actions[0].resolved_canonical_paper_id="DOI:10.5555/example";
assert.ok(identityErrors(noncanonicalIdentity).some(message=>message.includes("каноническим")));

bindings={identity:null,lineage:null};
state=clone(initialDraft.draft);
state.submissions[0].selected_task_ids=["task_"+"f".repeat(64)];
assert.ok(validateSource().some(
  error=>error.row===1&&error.message.includes("primary task")
));
const knownTask=BOOT.groups[0].task_summaries[0];
const selectedSubmission=clone(initialDraft.draft.submissions[0]);
selectedSubmission.stratum=STRATA.find(value=>value!==knownTask.stratum);
selectTask(selectedSubmission,knownTask);
assert.deepEqual(selectedSubmission.selected_task_ids,[knownTask.task_id]);
assert.equal(selectedSubmission.stratum,knownTask.stratum);
const unknownStratumTask={...knownTask,stratum:""};
selectedSubmission.stratum="easy_control";
selectTask(selectedSubmission,unknownStratumTask);
assert.equal(selectedSubmission.stratum,"easy_control");
state=clone(initialDraft.draft);
state.submissions[0].selected_task_ids=[knownTask.task_id];
state.submissions[0].stratum=STRATA.find(value=>value!==knownTask.stratum);
assert.ok(validateSource().some(
  error=>error.row===1&&error.message.includes("явно выбранной primary task")
));
state=clone(initialDraft.draft);
state.submissions[0].canonical_paper_id="paper:runtime-duplicate";
state.submissions[1].canonical_paper_id="paper:runtime-duplicate";
assert.ok(validateSource().some(
  error=>error.row===2&&error.message.includes("canonical_paper_id уже используется")
));
state=clone(initialDraft.draft);
state.submissions[0].images[0].image_path="assets/images/рисунок.png";
assert.ok(validateSource().some(
  error=>error.row===1&&error.message.includes("ASCII Windows-safe")
));
state=clone(initialDraft.draft);
state.submissions[0].images[0].image_path="assets/images/Figure.PNG";
state.submissions[1].images[0].image_path="assets/images/figure.png";
assert.ok(validateSource().some(
  error=>error.row===2&&error.message.includes("image_path уже используется")
));
assert.equal(
  draftEnvelope().draft.submissions[0].images[0].image_path,
  "assets/images/Figure.PNG"
);
for(const [firstPath,secondPath] of [
  ["assets/images/a","assets/images/a/b.png"],
  ["assets/images/a/b.png","assets/images/a"],
]){
  state=clone(initialDraft.draft);
  state.submissions[0].images[0].image_path=firstPath;
  state.submissions[1].images[0].image_path=secondPath;
  assert.ok(validateSource().some(
    error=>error.row===2&&error.message.includes("file/directory")
  ));
}
state=clone(initialDraft.draft);
state.submissions[0].images[0].sha256=BOOT.quarantined_legacy_sha256s[0];
assert.ok(validateSource().some(
  error=>error.row===1&&error.message.includes("release_usable=false")
));
pendingHashCount=1;
assert.ok(validateSource().some(error=>error.row===null&&error.message.includes("SHA256")));
pendingHashCount=0;

const draftFile={name:"draft.json"},draftInput={files:[draftFile]};
let capturedToken=++draftImportToken,capturedRevision=stateRevision;
assert.equal(draftImportCurrent(draftInput,draftFile,capturedToken,capturedRevision),true);
stateChanged();
assert.equal(draftImportCurrent(draftInput,draftFile,capturedToken,capturedRevision),false);
capturedToken=++draftImportToken;capturedRevision=stateRevision;
assert.equal(draftImportCurrent(draftInput,draftFile,capturedToken,capturedRevision),true);
draftImportToken+=1;
assert.equal(draftImportCurrent(draftInput,draftFile,capturedToken,capturedRevision),false);
capturedToken=++draftImportToken;capturedRevision=stateRevision;
draftInput.files=[{name:"newer.json"}];
assert.equal(draftImportCurrent(draftInput,draftFile,capturedToken,capturedRevision),false);
const storageMessages=[];
setMessage=(text,bad=false)=>storageMessages.push({text,bad});
storageSet=()=>{setMessage("storage failure",true);return false};
renderProgress=()=>{};
assert.equal(save(),false);
assert.deepEqual(storageMessages.at(-1),{text:"storage failure",bad:true});

async function bindingDraftRaceChecks(){
  const staleDraftFile={name:"slow-draft.json"},staleDraftInput={files:[staleDraftFile]};
  const staleToken=++draftImportToken,staleRevision=stateRevision;
  let releaseDraft;
  const staleDraft=new Promise(resolve=>{releaseDraft=resolve}).then(()=>{
    if(draftImportCurrent(staleDraftInput,staleDraftFile,staleToken,staleRevision)){
      bindings={identity:null,lineage:null};
    }
  });
  const selectedIdentityFile={name:"identity.json"},identityInput=$("identityFile");
  identityInput.files=[selectedIdentityFile];
  readJsonWithHash=async file=>{
    assert.equal(file,selectedIdentityFile);
    return {value:validIdentity,sha256:"3".repeat(64)};
  };
  await identityInput.listeners.change({target:identityInput});
  assert.equal(draftImportCurrent(
    staleDraftInput,staleDraftFile,staleToken,staleRevision
  ),false);
  releaseDraft();
  await staleDraft;
  assert.equal(bindings.identity.value,validIdentity);

  const validLineage={
    schema_version:1,
    training_sources:[{
      repo_id:"example/training",repo_type:"dataset",revision:"4".repeat(40),
      files:[{path:"data.jsonl",sha256:"5".repeat(64),row_count:1}],
    }],
    coverage:{paper_ids:true,source_documents:true,creator_groups:true,image_bytes:true,prompts:true},
    paper_ids:["paper:training-lineage"],
    source_document_ids:["training-source"],
    creator_group_ids:["training-creator"],
    image_sha256s:["6".repeat(64)],
    prompt_sha256s:["7".repeat(64)],
  };
  const priorLineageDraftFile={name:"slow-lineage-draft.json"};
  const priorLineageDraftInput={files:[priorLineageDraftFile]};
  const priorLineageToken=++draftImportToken,priorLineageRevision=stateRevision;
  const selectedLineageFile={name:"lineage.json"},lineageInput=$("lineageFile");
  lineageInput.files=[selectedLineageFile];
  readJsonWithHash=async file=>{
    assert.equal(file,selectedLineageFile);
    return {value:validLineage,sha256:"8".repeat(64)};
  };
  await lineageInput.listeners.change({target:lineageInput});
  assert.equal(draftImportCurrent(
    priorLineageDraftInput,priorLineageDraftFile,priorLineageToken,priorLineageRevision
  ),false);
  assert.equal(bindings.lineage.value,validLineage);
}
bindingDraftRaceChecks().catch(error=>{console.error(error.stack||error);process.exitCode=1});

async function hashRaceChecks(){
  save=()=>{};
  const completions=new Map();
  fileSha=file=>new Promise((resolve,reject)=>completions.set(file,{resolve,reject}));
  const target={sha256:"previous"},parent=new FakeElement("parent");
  const linkedInput=new FakeElement("input");linkedInput.value="previous";
  const control=addFileHasher(parent,target,"sha256","SHA256","",linkedInput);
  const firstFile={name:"first.bin",size:1,type:"application/octet-stream"};
  control.input.files=[firstFile];
  const first=control.input.listeners.change();
  assert.equal(target.sha256,"");
  assert.equal(linkedInput.value,"");
  assert.equal(pendingHashCount,1);
  const secondFile={name:"second.bin",size:2,type:"application/octet-stream"};
  control.input.files=[secondFile];
  const second=control.input.listeners.change();
  assert.equal(pendingHashCount,1);
  completions.get(secondFile).resolve("2".repeat(64));
  await second;
  assert.equal(target.sha256,"2".repeat(64));
  assert.equal(linkedInput.value,"2".repeat(64));
  assert.equal(pendingHashCount,0);
  completions.get(firstFile).resolve("1".repeat(64));
  await first;
  assert.equal(target.sha256,"2".repeat(64));
  assert.equal(linkedInput.value,"2".repeat(64));
  assert.equal(pendingHashCount,0);
  const thirdFile={name:"third.bin",size:3,type:"application/octet-stream"};
  control.input.files=[thirdFile];
  const third=control.input.listeners.change();
  assert.equal(target.sha256,"");
  assert.equal(linkedInput.value,"");
  cancelFieldHashers(target,"sha256");
  target.sha256="3".repeat(64);
  linkedInput.value="3".repeat(64);
  completions.get(thirdFile).resolve("4".repeat(64));
  await third;
  assert.equal(target.sha256,"3".repeat(64));
  assert.equal(linkedInput.value,"3".repeat(64));
  assert.equal(pendingHashCount,0);
  const failedFile={name:"failed.bin",size:4,type:"application/octet-stream"};
  control.input.files=[failedFile];
  const failed=control.input.listeners.change();
  assert.equal(target.sha256,"");
  assert.equal(linkedInput.value,"");
  completions.get(failedFile).reject(new Error("hash failed"));
  await failed;
  assert.equal(target.sha256,"");
  assert.equal(linkedInput.value,"");
  assert.equal(pendingHashCount,0);
  cleanupFileHashers(target);
  assert.equal(fileHasherRuntimes.size,0);
}
hashRaceChecks().catch(error=>{console.error(error.stack||error);process.exitCode=1});
""",
        ]
    )
    _run_node_script(tmp_path, "source-browser-guards.js", source_checks)

    identity_prefix, identity_marker, _ = handoff._IDENTITY_JS.partition(
        '$("draftExport").addEventListener'
    )
    assert identity_marker
    identity_checks = "\n".join(
        [
            _node_prelude(identity_data),
            handoff._FORM_SHARED_JS,
            identity_prefix,
            r"""
const actionNav=$("actionNav"),actionForm=$("actionForm");
renderNav();
assert.equal(actionNav.children.length,BOOT.actions.length);
actionNav.children.at(-1).listeners.click();
assert.equal(currentIndex,BOOT.actions.length-1);
assert.deepEqual(actionForm.scrollIntoViewCalls.at(-1),{block:"start"});
let actionScrollCount=actionForm.scrollIntoViewCalls.length;
renderErrorSummary([{row:BOOT.actions[0].row,message:"action navigation check"}],goRow);
$("errors").children.at(-1).children[0].children[0].listeners.click();
assert.equal(currentIndex,0);
assert.equal(actionForm.scrollIntoViewCalls.length,actionScrollCount+1);
actionScrollCount=actionForm.scrollIntoViewCalls.length;
$("firstIncomplete").listeners.click();
assert.equal(currentIndex,0);
assert.equal(actionForm.scrollIntoViewCalls.length,actionScrollCount+1);
cleanupFileHashers();
function completeResolutions(){
  return BOOT.actions.map(action=>({
    row:action.row,
    group_id:action.group_id,
    resolution_type:"replace_with_other",
    resolved_canonical_paper_id:`paper:identity-resolution-${action.row}`,
    evidence:{
      source_url:`https://example.test/evidence/${action.row}`,
      sha256:"b".repeat(64),
      retrieved_at:"2026-07-23T12:00:00Z",
      notes:"Проверено владельцем данных",
    },
    capacity_plan_regenerated:true,
  }));
}
state.resolutions=completeResolutions();
assert.deepEqual(allErrors(),[]);
state.resolutions=completeResolutions();
state.resolutions[0].resolved_canonical_paper_id=BOOT.occupied_non_action_paper_ids[0];
assert.ok(allErrors().some(error=>error.message.includes("уже занят")));
state.resolutions=completeResolutions();
state.resolutions[1].resolved_canonical_paper_id=
  state.resolutions[0].resolved_canonical_paper_id;
assert.ok(allErrors().some(error=>error.message.includes("повторяет")));
state.resolutions=completeResolutions();
state.resolutions[0].capacity_plan_regenerated=false;
assert.ok(allErrors().some(error=>error.message.includes("capacity_plan_regenerated")));
state.resolutions=completeResolutions();
state.resolutions[0].resolved_canonical_paper_id="ARXIV:2401.01234";
assert.ok(allErrors().some(error=>error.message.includes("каноническим")));
state.resolutions=completeResolutions();
pendingHashCount=1;
assert.ok(allErrors().some(error=>error.row===null&&error.message.includes("SHA256")));
pendingHashCount=0;
const draftFile={name:"identity-draft.json"},draftInput={files:[draftFile]};
let capturedToken=++draftImportToken,capturedRevision=stateRevision;
assert.equal(draftImportCurrent(draftInput,draftFile,capturedToken,capturedRevision),true);
stateChanged();
assert.equal(draftImportCurrent(draftInput,draftFile,capturedToken,capturedRevision),false);
capturedToken=++draftImportToken;capturedRevision=stateRevision;
draftImportToken+=1;
assert.equal(draftImportCurrent(draftInput,draftFile,capturedToken,capturedRevision),false);
const storageMessages=[];
setMessage=(text,bad=false)=>storageMessages.push({text,bad});
storageSet=()=>{setMessage("storage failure",true);return false};
renderNav=()=>{};
assert.equal(save(),false);
assert.deepEqual(storageMessages.at(-1),{text:"storage failure",bad:true});
""",
        ]
    )
    _run_node_script(tmp_path, "identity-browser-guards.js", identity_checks)

    lineage_prefix, lineage_marker, _ = handoff._LINEAGE_JS.partition(
        '$("addSource").addEventListener'
    )
    assert lineage_marker
    lineage_checks = "\n".join(
        [
            _node_prelude(lineage_data),
            handoff._FORM_SHARED_JS,
            lineage_prefix,
            r"""
assert.equal(validDraft(envelope()),true);
const draftFile={name:"lineage-draft.json"},draftInput={files:[draftFile]};
let capturedToken=++draftImportToken,capturedRevision=stateRevision;
assert.equal(draftImportCurrent(draftInput,draftFile,capturedToken,capturedRevision),true);
stateChanged();
assert.equal(draftImportCurrent(draftInput,draftFile,capturedToken,capturedRevision),false);
capturedToken=++draftImportToken;capturedRevision=stateRevision;
draftInput.files=[{name:"different.json"}];
assert.equal(draftImportCurrent(draftInput,draftFile,capturedToken,capturedRevision),false);
const storageMessages=[];
setMessage=(text,bad=false)=>storageMessages.push({text,bad});
storageSet=()=>{setMessage("storage failure",true);return false};
assert.equal(save(),false);
assert.deepEqual(storageMessages.at(-1),{text:"storage failure",bad:true});
""",
        ]
    )
    _run_node_script(tmp_path, "lineage-browser-guards.js", lineage_checks)


def test_symlinked_output_ancestor_is_rejected(tmp_path: Path) -> None:
    dossiers = _synthetic_dossiers(tmp_path)
    destination = tmp_path / "symlink-destination"
    destination.mkdir()
    ancestor = tmp_path / "symlink-ancestor"
    try:
        ancestor.symlink_to(destination, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlink creation is unavailable: {exc}")

    with pytest.raises(handoff.HandoffError, match="symlink or reparse point"):
        handoff.generate_handoff(dossiers, AUDIT, ancestor / "handoff")
    assert not (destination / "handoff").exists()


def test_windows_junction_output_ancestor_is_rejected(tmp_path: Path) -> None:
    if sys.platform != "win32":
        pytest.skip("Windows junction creation is unavailable")
    cmd = shutil.which("cmd")
    if cmd is None:
        pytest.skip("cmd is unavailable for junction creation")

    dossiers = _synthetic_dossiers(tmp_path)
    destination = tmp_path / "junction-destination"
    destination.mkdir()
    ancestor = tmp_path / "junction-ancestor"
    try:
        created = subprocess.run(
            [cmd, "/d", "/c", "mklink", "/J", str(ancestor), str(destination)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        pytest.skip(f"junction creation is unavailable: {exc}")
    if created.returncode != 0 or not ancestor.is_dir():
        pytest.skip(f"junction creation is unavailable: {created.stderr.strip()}")

    attributes = getattr(ancestor.lstat(), "st_file_attributes", 0)
    reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    assert attributes & reparse_point
    with pytest.raises(handoff.HandoffError, match="symlink or reparse point"):
        handoff.generate_handoff(dossiers, AUDIT, ancestor / "handoff")
    assert not (destination / "handoff").exists()


def test_existing_different_workspace_is_rejected(tmp_path: Path) -> None:
    dossiers = _synthetic_dossiers(tmp_path)
    output = tmp_path / "handoff"
    handoff.generate_handoff(dossiers, AUDIT, output)
    (output / "README_RU.md").write_bytes((output / "README_RU.md").read_bytes() + b"tamper")

    with pytest.raises(handoff.HandoffError, match="different handoff workspace"):
        handoff.generate_handoff(dossiers, AUDIT, output)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("partition", "multiple categories"),
        ("binding", "does not bind its dossier"),
        ("url", "without credentials, query, or fragment"),
        ("hash", "lowercase SHA256"),
        ("probe_id", "lowercase canonical paper ID"),
        ("exception_id", "lowercase canonical paper ID"),
    ],
)
def test_tampered_audit_is_rejected(tmp_path: Path, mutation: str, message: str) -> None:
    dossiers = _synthetic_dossiers(tmp_path)
    audit = _audit()
    if mutation == "partition":
        audit["partition"]["categories"]["metadata_only_or_unlicensed"][0] = 1
    elif mutation == "binding":
        audit["source_probes"][0]["group_id"] = "0" * 64
    elif mutation == "url":
        audit["source_probes"][0]["direct_image_url"] += "?token=secret"
    elif mutation == "hash":
        audit["source_probes"][0]["direct_image_sha256"] = "A" * 64
    elif mutation == "probe_id":
        audit["source_probes"][0]["verified_canonical_paper_id"] = "ARXIV:2401.01234"
    else:
        audit["identity_exceptions"][0]["candidate_canonical_paper_id"] = "PAPER:openalex:w123"
    audit_path = _write_audit(tmp_path, audit)

    with pytest.raises(handoff.HandoffError, match=message):
        handoff.generate_handoff(dossiers, audit_path, tmp_path / "handoff")
    assert not (tmp_path / "handoff").exists()


@pytest.mark.parametrize("bad_value", ["duplicate", "nonfinite"])
def test_strict_json_rejects_duplicate_keys_and_nonfinite(tmp_path: Path, bad_value: str) -> None:
    dossiers = _synthetic_dossiers(tmp_path)
    text = AUDIT.read_text(encoding="ascii")
    if bad_value == "duplicate":
        text = text.replace(
            '"artifact_version": 1,',
            '"artifact_version": 1, "artifact_version": 1,',
            1,
        )
    else:
        text = text.replace('"artifact_version": 1', '"artifact_version": NaN', 1)
    audit_path = tmp_path / "inputs/audit.json"
    audit_path.write_text(text, encoding="ascii")

    with pytest.raises(handoff.HandoffError, match="duplicate JSON key|non-finite JSON"):
        handoff.generate_handoff(dossiers, audit_path, tmp_path / "handoff")
    assert not (tmp_path / "handoff").exists()
