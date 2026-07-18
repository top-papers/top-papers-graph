# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from scireason.vlm_ab.inference import (
    InferenceConfigurationError,
    InferenceInputError,
    build_condition_rows,
    load_transformers_model,
    manifest_path_for,
    parse_json_response,
    run_inference,
    validate_scireason_response,
)


def _valid_response() -> dict[str, object]:
    return {
        "answer": "The evidence supports the claim.",
        "evidence_used": [
            {
                "kind": "figure",
                "locator": "Fig. 1",
                "description": "The plotted trend supports the answer.",
            }
        ],
        "visual_facts": ["The plotted value increases."],
        "temporal_facts": ["The later measurement is larger."],
        "uncertainty": "low",
        "missing_evidence": [],
    }


def _contains_image(messages: list[dict[str, object]]) -> bool:
    return any(
        isinstance(block, dict) and block.get("type") == "image"
        for message in messages
        for block in (message.get("content") if isinstance(message.get("content"), list) else [])
    )


def _without_images(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    result = json.loads(json.dumps(messages))
    for message in result:
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = [
                block
                for block in content
                if not (isinstance(block, dict) and block.get("type") == "image")
            ]
    return result


def _benchmark(root: Path, count: int = 3) -> list[dict[str, object]]:
    images = root / "images"
    images.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for index in range(count):
        image = images / f"paper-{index}.png"
        image.write_bytes(f"image-{index}".encode("ascii"))
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "paper_id": f"paper-{index}",
                "images": [f"images/{image.name}"],
                "messages": [
                    {"role": "system", "content": "Return strict SciReason JSON."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": f"Inspect sample {index}."},
                        ],
                    },
                    {"role": "assistant", "content": "private reference answer"},
                ],
            }
        )
    return rows


def test_parse_json_response_and_schema_validation() -> None:
    valid = _valid_response()
    fenced = "analysis first\n```json\n" + json.dumps(valid) + "\n```\nafter"
    assert parse_json_response(fenced) == valid
    assert parse_json_response("prefix " + json.dumps(valid) + " suffix") == valid
    assert parse_json_response('{"a": 1, "a": 2}') is None
    assert parse_json_response('{"value": NaN}') is None
    assert parse_json_response("not JSON") is None

    assert validate_scireason_response(valid)
    assert not validate_scireason_response({**valid, "answer": ""})
    assert not validate_scireason_response({**valid, "evidence_used": [{"kind": "figure"}]})


def test_condition_rows_are_safe_deterministic_and_keep_control_text(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    rows = _benchmark(dataset_root)

    first = build_condition_rows(rows, dataset_root, seed=1729)
    second = build_condition_rows(list(reversed(rows)), dataset_root, seed=1729)
    first_by_key = {(row["sample_id"], row["condition"]): row for row in first}
    second_by_key = {(row["sample_id"], row["condition"]): row for row in second}
    assert first_by_key == second_by_key

    donors = {}
    for sample in rows:
        sample_id = str(sample["sample_id"])
        paper_id = str(sample["paper_id"])
        original = first_by_key[(sample_id, "original")]
        text_only = first_by_key[(sample_id, "text_only")]
        shuffled = first_by_key[(sample_id, "shuffled_images")]

        assert _contains_image(original["messages"])
        assert not _contains_image(text_only["messages"])
        assert _contains_image(shuffled["messages"])
        assert original["input_image_hashes"]
        assert text_only["input_image_hashes"] == []
        assert shuffled["input_image_hashes"]
        assert shuffled["shuffle_source_paper_id"] != paper_id
        donors[paper_id] = shuffled["shuffle_source_paper_id"]
        assert _without_images(original["messages"]) == text_only["messages"]
        assert _without_images(shuffled["messages"]) == text_only["messages"]
        assert all(
            "private reference answer" not in json.dumps(row["messages"])
            for row in (original, text_only, shuffled)
        )
        image_block = original["messages"][1]["content"][0]
        assert Path(image_block["image"]).is_absolute()

    assert set(donors) == set(donors.values())
    assert len(set(donors.values())) == len(donors)


def test_condition_rows_reject_path_traversal(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    row = {
        "sample_id": "unsafe",
        "paper_id": "paper",
        "images": ["../outside.png"],
        "messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "x"}]}
        ],
    }
    with pytest.raises(InferenceInputError, match="escapes dataset_root"):
        build_condition_rows([row], dataset_root)


def test_condition_rows_accept_absolute_paths_inside_root(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    rows = _benchmark(dataset_root, count=1)
    image = (dataset_root / str(rows[0]["images"][0])).resolve()
    rows[0]["images"] = [str(image)]

    built = build_condition_rows(rows, dataset_root, conditions=("original",))

    assert built[0]["image_uris"] == [str(image)]


def test_mock_run_is_strict_deterministic_and_resumable(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    rows = _benchmark(dataset_root, count=2)
    output = tmp_path / "predictions.jsonl"
    kwargs = {
        "benchmark": rows,
        "dataset_root": dataset_root,
        "arm_name": "base",
        "arm_config": {"label": "mock-base"},
        "processor_config": {"processor_id": "shared", "revision": "processor-rev"},
        "generation_config": {"max_new_tokens": 96},
        "output_jsonl": output,
        "conditions": ("original", "text_only", "shuffled_images"),
        "seed": 23,
        "backend": "mock",
    }
    first = run_inference(**kwargs)
    original_bytes = output.read_bytes()
    edited_rows = [json.loads(line) for line in original_bytes.decode("utf-8").splitlines()]
    edited_rows[0]["parsed_response"]["answer"] = "Edited but schema-valid answer."
    edited_rows[0]["raw_response"] = json.dumps(edited_rows[0]["parsed_response"])
    output.write_text(
        "".join(
            json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in edited_rows
        ),
        encoding="utf-8",
    )
    with pytest.raises(InferenceConfigurationError, match="prediction bytes changed"):
        run_inference(**kwargs)
    output.write_bytes(original_bytes)
    second = run_inference(**kwargs)
    reversed_resume = run_inference(**{**kwargs, "benchmark": list(reversed(rows))})

    assert first["written_rows"] == 6
    assert first["successful_rows"] == 6
    assert second["written_rows"] == 0
    assert second["existing_rows"] == 6
    assert reversed_resume["written_rows"] == 0
    assert reversed_resume["existing_rows"] == 6
    assert output.read_bytes() == original_bytes

    output_rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(output_rows) == 6
    assert len({(row["sample_id"], row["condition"]) for row in output_rows}) == 6
    assert all(row["status"] == "success" for row in output_rows)
    assert all(row["parse_valid"] and row["schema_valid"] for row in output_rows)
    assert all(validate_scireason_response(row["parsed_response"]) for row in output_rows)
    assert all(len(row["config_fingerprint"]) == 64 for row in output_rows)

    manifest = json.loads(manifest_path_for(output).read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["completed_rows"] == 6
    assert manifest["fingerprints"]["config"] == first["config_fingerprint"]
    assert manifest["runtime"]["backend"] == "mock"


def test_resume_rejects_fingerprint_mismatch_without_changing_output(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    rows = _benchmark(dataset_root, count=1)
    output = tmp_path / "predictions.jsonl"
    common = {
        "benchmark": rows,
        "dataset_root": dataset_root,
        "arm_name": "base",
        "arm_config": {"label": "mock-base"},
        "processor_config": {},
        "output_jsonl": output,
        "conditions": ("original",),
        "seed": 5,
        "backend": "mock",
    }
    run_inference(generation_config={"max_new_tokens": 32}, **common)
    before = output.read_bytes()
    with pytest.raises(InferenceConfigurationError, match="fingerprint mismatch"):
        run_inference(generation_config={"max_new_tokens": 64}, **common)
    assert output.read_bytes() == before


def test_resume_rejects_existing_output_when_manifest_was_deleted(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    rows = _benchmark(dataset_root, count=1)
    output = tmp_path / "predictions.jsonl"
    kwargs = {
        "benchmark": rows,
        "dataset_root": dataset_root,
        "arm_name": "base",
        "arm_config": {"label": "mock-base"},
        "processor_config": {},
        "generation_config": {"max_new_tokens": 32},
        "output_jsonl": output,
        "conditions": ("original",),
        "seed": 5,
        "backend": "mock",
    }
    run_inference(**kwargs)
    before = output.read_bytes()
    manifest_path_for(output).unlink()

    with pytest.raises(InferenceConfigurationError, match="without its integrity manifest"):
        run_inference(**kwargs)
    assert output.read_bytes() == before


def test_tuned_loader_explicitly_loads_pinned_base_then_adapter(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeBase:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls["base"] = (model_id, kwargs)
            return cls()

    class FakePeft:
        def __init__(self):
            self.active_adapter = "science"
            self.peft_config = {"science": object()}
            self.eval_called = False

        @classmethod
        def from_pretrained(cls, base, adapter_id, **kwargs):
            calls["adapter"] = (base, adapter_id, kwargs)
            return cls()

        def eval(self):
            self.eval_called = True
            return self

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.Qwen3VLForConditionalGeneration = FakeBase
    fake_peft = types.ModuleType("peft")
    fake_peft.PeftModel = FakePeft
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    model = load_transformers_model(
        "tuned",
        {
            "base_model_id": "Qwen/pinned-base",
            "base_revision": "base-commit",
            "model_kwargs": {"device_map": "cpu"},
            "adapter_id": "org/pinned-adapter",
            "adapter_revision": "adapter-commit",
            "adapter_name": "science",
        },
    )

    assert isinstance(model, FakePeft)
    assert model.eval_called
    assert calls["base"] == (
        "Qwen/pinned-base",
        {"revision": "base-commit", "device_map": "cpu"},
    )
    base, adapter_id, adapter_kwargs = calls["adapter"]
    assert isinstance(base, FakeBase)
    assert adapter_id == "org/pinned-adapter"
    assert adapter_kwargs == {
        "revision": "adapter-commit",
        "adapter_name": "science",
        "is_trainable": False,
    }


def test_transformers_loader_rejects_local_path_shadowing(tmp_path: Path) -> None:
    local_model = tmp_path / "Qwen" / "pinned-base"
    local_model.mkdir(parents=True)

    with pytest.raises(InferenceConfigurationError, match="local path"):
        load_transformers_model(
            "base",
            {
                "base_model_id": str(local_model),
                "base_revision": "a" * 40,
            },
        )
