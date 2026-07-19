# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import copy
import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pytest

from scireason.vlm_ab import cli
from scireason.vlm_ab.curator import (
    _IMAGE_EDIT_KEYS,
    _blank_edit,
    _browser_payload,
    _embedded_json,
    _merge_draft_edits,
    _source_inventory,
    generate_curator_workspace,
)
from scireason.vlm_ab.remediation import (
    RemediationError,
    _queue_material,
    _validate_decisions,
    _verify_queue_workspace,
)
from test_vlm_ab_remediation import (
    BENCHMARK_SCHEMA,
    PROVENANCE_SCHEMA,
    _curated_replacements,
    _generate_queue,
    _prepare_bundle,
)


def _generate(bundle: dict, queue: Path, output: Path) -> dict:
    return generate_curator_workspace(
        bundle["config"],
        bundle["prepare_manifest"],
        queue / "queue_manifest.json",
        output,
        benchmark_schema=BENCHMARK_SCHEMA,
        provenance_schema=PROVENANCE_SCHEMA,
    )


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_workspace_generation_integrity_deduplication_and_idempotency(tmp_path: Path) -> None:
    bundle = _prepare_bundle(
        tmp_path / "bundle",
        row_count=2,
        shared_image_bytes=True,
    )
    queue = tmp_path / "queue"
    queue_manifest = _generate_queue(bundle, queue)
    queue_before = _files(queue)
    output = tmp_path / "forms"

    manifest = _generate(bundle, queue, output)
    first = _files(output)
    repeated = _generate(bundle, queue, output)

    assert repeated == manifest
    assert _files(output) == first
    assert _files(queue) == queue_before
    assert manifest["artifact_version"] == 1
    assert manifest["queue_fingerprint"] == queue_manifest["queue_fingerprint"]
    assert manifest["task_count"] == 2
    assert len(manifest["source_images"]) == 1
    image = manifest["source_images"][0]
    assert len(image["source_references"]) == 2
    assert hashlib.sha256((output / image["path"]).read_bytes()).hexdigest() == image["sha256"]
    html_bytes = (output / "curator.html").read_bytes()
    assert hashlib.sha256(html_bytes).hexdigest() == manifest["html_sha256"]
    assert json.loads((output / "workspace_manifest.json").read_text(encoding="utf-8")) == manifest
    assert set(first) == {"curator.html", "workspace_manifest.json", image["path"]}

    material = _queue_material(
        bundle["config"], bundle["prepare_manifest"], BENCHMARK_SCHEMA, PROVENANCE_SCHEMA
    )
    task_ids = [task["task_id"] for task in material["tasks"]]
    html = html_bytes.decode("utf-8")
    assert all(task_id in html for task_id in task_ids)
    assert queue_manifest["queue_fingerprint"] in html
    assert '<html lang="ru">' in html
    assert "<title>Офлайн-курация VLM benchmark</title>" in html
    assert "Офлайн-курация benchmark" in html
    assert "Предыдущая" in html
    assert "Следующая" in html
    assert "Экспортировать черновик" in html
    assert "Объединить черновик" in html
    assert "Выбрать проверенный файл и вычислить SHA256" in html
    assert "проверка в браузере только предварительная" in html
    assert "connect-src 'none'" in html
    assert "default-src 'none'" in html
    assert "fetch(" not in html
    assert "eval(" not in html
    assert "innerHTML" not in html
    assert "<script src=" not in html
    assert "<link rel=" not in html
    assert 'src="http' not in html
    assert "import-decisions" not in html
    assert "Import decisions JSONL" not in html
    assert "конфликт merge для task_id" in html
    assert "изменения не применены" in html
    assert "state.edits=candidate" in html
    assert "MAX_DRAFT_BYTES=32*1024*1024" in html
    assert 'exactKeys(envelope,["artifact_version","queue_fingerprint","edits"])' in html
    assert "function storageGet(){try{" in html
    assert "function storageSet(value){try{" in html
    assert "function storageRemove(){try{" in html
    assert "регулярно экспортируйте черновик" in html
    assert "два указанных эксперта независимо проверили это решение" in html
    assert "independent_attestation" in html
    assert "validAssetPath" in html
    assert 'fileInput.accept="image/*"' in html
    assert "file.arrayBuffer()" in html
    assert 'crypto.subtle.digest("SHA-256",buffer)' in html
    assert "URL.createObjectURL(file)" in html
    assert "URL.revokeObjectURL" in html
    assert "const imageRuntime=new Map()" in html
    assert "let pendingHashCount=0" in html
    assert "runtime.token+=1" in html
    assert "currentHash(image,token,fileInput,file)" in html
    assert "discardImageRuntime(image)" in html
    assert "Финальный экспорт заблокирован: дождитесь вычисления SHA256" in html
    assert "image.image_path=" not in html
    assert "state.file" not in html
    assert "state.bytes" not in html
    assert "file" not in _blank_edit()
    assert "bytes" not in _blank_edit()
    assert {"token", "pending", "url", "file", "bytes"}.isdisjoint(_IMAGE_EDIT_KEYS)
    assert '.normalize("NFKC")' in html
    assert '.replace(/\\s+/gu," ").toLowerCase()' in html
    assert "const restoreCandidate=" in html
    assert "state=restoreCandidate" in html
    assert "state.edits[taskId]=validateEdit(value,taskId)" not in html
    assert "page:im.page.trim()" in html
    assert "Number(im.page)" not in html


def test_draft_merge_is_task_level_validated_and_atomic() -> None:
    task_ids = {"task-a", "task-b"}
    blank = _blank_edit()
    completed = _blank_edit()
    completed.update(
        disposition="exclude",
        exclusion_reason="Verified exclusion.",
        reviewed_by="expert-1\nexpert-2",
        independent_attestation=True,
    )
    current = {"task-a": copy.deepcopy(blank), "task-b": copy.deepcopy(completed)}
    envelope = {
        "artifact_version": 1,
        "queue_fingerprint": "f" * 64,
        "edits": {
            "task-a": copy.deepcopy(completed),
            "task-b": copy.deepcopy(completed),
        },
    }

    merged, summary = _merge_draft_edits(current, envelope, task_ids, "f" * 64)

    assert merged == {"task-a": completed, "task-b": completed}
    assert summary == {"merged": 1, "equal": 1, "blank_ignored": 0}
    assert current["task-a"] == blank

    conflicting = copy.deepcopy(envelope)
    conflicting["edits"]["task-b"]["exclusion_reason"] = "Different nonblank work."
    before = copy.deepcopy(current)
    with pytest.raises(ValueError, match="merge conflict for task_id task-b"):
        _merge_draft_edits(current, conflicting, task_ids, "f" * 64)
    assert current == before


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (lambda value: value.update(extra=True), "envelope fields"),
        (lambda value: value["edits"].update({"unknown": _blank_edit()}), "unknown task_id"),
        (
            lambda value: value["edits"]["task-a"].update(gold_enabled="false"),
            "non-boolean",
        ),
    ],
)
def test_draft_merge_rejects_malformed_envelopes(mutation, expected: str) -> None:
    envelope = {
        "artifact_version": 1,
        "queue_fingerprint": "f" * 64,
        "edits": {"task-a": _blank_edit()},
    }
    mutation(envelope)

    with pytest.raises(ValueError, match=expected):
        _merge_draft_edits({"task-a": _blank_edit()}, envelope, {"task-a"}, "f" * 64)


def test_embedded_json_escapes_script_breakout_and_unicode_separators() -> None:
    encoded = _embedded_json({"malicious": "</script>&>\u2028\u2029"})
    assert "</script>" not in encoded
    assert "&" not in encoded
    assert ">" not in encoded
    assert "\u2028" not in encoded
    assert "\u2029" not in encoded
    assert "\\u003c/script\\u003e" in encoded
    assert json.loads(encoded)["malicious"] == "</script>&>\u2028\u2029"


@pytest.mark.parametrize("location", ["same", "inside", "contains"])
def test_workspace_rejects_queue_related_output(tmp_path: Path, location: str) -> None:
    bundle = _prepare_bundle(tmp_path / "bundle")
    queue = tmp_path / "container" / "queue"
    _generate_queue(bundle, queue)
    output = {
        "same": queue,
        "inside": queue / "forms",
        "contains": queue.parent,
    }[location]
    before = _files(queue)

    with pytest.raises(RemediationError, match="separate from"):
        _generate(bundle, queue, output)

    assert _files(queue) == before


def test_workspace_rejects_tampered_queue(tmp_path: Path) -> None:
    bundle = _prepare_bundle(tmp_path / "bundle")
    queue = tmp_path / "queue"
    _generate_queue(bundle, queue)
    (queue / "tasks.jsonl").write_bytes((queue / "tasks.jsonl").read_bytes() + b"\n")

    with pytest.raises(RemediationError, match="tampered"):
        _generate(bundle, queue, tmp_path / "forms")

    assert not (tmp_path / "forms").exists()


@pytest.mark.parametrize("artifact", ["html", "image"])
def test_idempotent_regeneration_rejects_workspace_tampering(tmp_path: Path, artifact: str) -> None:
    bundle = _prepare_bundle(tmp_path / "bundle", row_count=1)
    queue = tmp_path / "queue"
    _generate_queue(bundle, queue)
    output = tmp_path / "forms"
    manifest = _generate(bundle, queue, output)
    target = (
        output / "curator.html"
        if artifact == "html"
        else output / manifest["source_images"][0]["path"]
    )
    target.write_bytes(target.read_bytes() + b"tampered")

    with pytest.raises(RemediationError, match="different curator workspace"):
        _generate(bundle, queue, output)


def test_browser_bindings_produce_server_valid_decision_rows(tmp_path: Path) -> None:
    bundle = _prepare_bundle(tmp_path / "bundle")
    queue = tmp_path / "queue"
    _generate_queue(bundle, queue)
    material = _queue_material(
        bundle["config"], bundle["prepare_manifest"], BENCHMARK_SCHEMA, PROVENANCE_SCHEMA
    )
    _, verified = _verify_queue_workspace(material, queue / "queue_manifest.json")
    inventory, _ = _source_inventory(material)
    payload = _browser_payload(material, verified, inventory)
    _, replacements = _curated_replacements(bundle, tmp_path / "curated")
    decisions = []
    for item, replacement in zip(payload["tasks"], replacements, strict=True):
        decisions.append(
            {
                **item["binding"],
                "status": "complete",
                "disposition": "retain",
                "exclusion_reason": None,
                "benchmark_row": copy.deepcopy(replacement[0]),
                "provenance_row": copy.deepcopy(replacement[1]),
                "reviewed_by": ["curator-1", "curator-2"],
                "notes": "Verified in the offline form.",
            }
        )

    retained, exclusions, reviewers = _validate_decisions(
        decisions, material["tasks"], verified["queue_fingerprint"]
    )

    assert len(decisions[0]) == 16
    assert len(retained) == 2
    assert exclusions == []
    assert reviewers == ["curator-1", "curator-2"]


def test_curate_forms_cli_parser_and_routing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("unused: true\n", encoding="utf-8")
    captured: dict = {}

    monkeypatch.setattr(cli, "load_experiment_config", lambda _path: {"loaded": True})
    monkeypatch.setattr(cli, "_repo_root", lambda _path, _root: tmp_path)

    def fake_command(args: Namespace, config: dict, root: Path) -> dict:
        captured.update(args=vars(args), config=config, root=root)
        return {"task_count": 386}

    monkeypatch.setattr(cli, "command_curate_forms", fake_command)
    code = cli.main(
        [
            "--config",
            str(config_path),
            "curate-forms",
            "--prepare-manifest",
            "prepare_manifest.json",
            "--queue-manifest",
            "queue/queue_manifest.json",
            "--output-dir",
            "curator-workspace",
        ]
    )

    assert code == 0
    assert captured["args"]["command"] == "curate-forms"
    assert captured["config"] == {"loaded": True}
    assert captured["root"] == tmp_path


def test_documented_full_suite_and_draft_workflow() -> None:
    root = Path(__file__).resolve().parents[1]
    next_steps = (root / "experiments/vlm_ab_evaluation/NEXT_STEPS_RU.md").read_text(
        encoding="utf-8"
    )
    readme = (root / "experiments/vlm_ab_evaluation/README.md").read_text(encoding="utf-8")

    assert "tests/test_vlm_ab_curator.py" in next_steps
    assert "Merge draft" in next_steps
    assert "без частичных изменений" in next_steps
    assert "Get-FileHash" in next_steps
    assert "browser сам криптографически workspace не проверяет" in next_steps
    assert "supports draft/existing-JSONL import" not in readme
    assert "does not import completed decisions JSONL" in readme
    assert "without partial changes" in readme
    assert "browser validation does not replace it" in readme
    assert "вычислить SHA256" in next_steps
    assert "отдельно копирует ровно выбранные bytes" in next_steps
    assert "calculate lowercase SHA256 via" in readme
    assert "separately copy the exact selected bytes" in readme
