# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

import scireason.vlm_ab.inference as inference_module
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
        assert all(isinstance(message["content"], list) for message in original["messages"])
        assert original["messages"][0]["content"] == [
            {"type": "text", "text": "Return strict SciReason JSON."}
        ]
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

        def load_adapter(self, adapter_id, **kwargs):
            calls["adapter_reload"] = (adapter_id, kwargs)
            return types.SimpleNamespace(missing_keys=[], unexpected_keys=[])

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
    assert calls["adapter_reload"] == (
        "org/pinned-adapter",
        {
            "revision": "adapter-commit",
            "adapter_name": "science",
            "is_trainable": False,
        },
    )


def _install_fake_nf4_runtime(monkeypatch, *, runtime_overrides=None, device_map=None):
    calls: list[tuple[str, dict[str, object]]] = []
    float16 = object()

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeBase:
        def __init__(self, quantization_config):
            self.is_loaded_in_4bit = True
            if device_map is not None:
                self.hf_device_map = device_map
            runtime = dict(quantization_config.__dict__)
            runtime.update(runtime_overrides or {})
            self.config = types.SimpleNamespace(
                quantization_config=FakeBitsAndBytesConfig(**runtime)
            )

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append((model_id, kwargs))
            return cls(kwargs["quantization_config"])

        def eval(self):
            return self

    class FakePeft:
        def __init__(self, base):
            self.base_model = types.SimpleNamespace(model=base)
            self.active_adapter = "science"
            self.peft_config = {"science": object()}

        @classmethod
        def from_pretrained(cls, base, adapter_id, **kwargs):
            return cls(base)

        def load_adapter(self, adapter_id, **kwargs):
            return types.SimpleNamespace(missing_keys=[], unexpected_keys=[])

        def eval(self):
            return self

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.BitsAndBytesConfig = FakeBitsAndBytesConfig
    fake_transformers.Qwen3VLForConditionalGeneration = FakeBase
    fake_peft = types.ModuleType("peft")
    fake_peft.PeftModel = FakePeft
    fake_torch = types.ModuleType("torch")
    fake_torch.float16 = float16
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "bitsandbytes", types.ModuleType("bitsandbytes"))
    return calls, FakeBitsAndBytesConfig, float16


def _nf4_model_config() -> dict[str, object]:
    return {
        "base_model_id": "Qwen/pinned-base",
        "base_revision": "base-commit",
        "model_kwargs": {
            "device_map": "auto",
            "quantization_config": {
                "load_in_4bit": True,
                "load_in_8bit": False,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "torch.float16",
                "bnb_4bit_use_double_quant": True,
            },
        },
    }


def test_nf4_kwargs_are_normalized_identically_for_base_and_tuned(monkeypatch) -> None:
    calls, config_class, float16 = _install_fake_nf4_runtime(monkeypatch)
    config = _nf4_model_config()

    load_transformers_model("base", config)
    load_transformers_model(
        "tuned",
        {
            **config,
            "adapter_id": "org/pinned-adapter",
            "adapter_revision": "adapter-commit",
            "adapter_name": "science",
        },
    )

    assert len(calls) == 2
    first = calls[0][1]
    second = calls[1][1]
    assert first.keys() == second.keys()
    assert first["device_map"] == second["device_map"] == "auto"
    assert isinstance(first["quantization_config"], config_class)
    assert isinstance(second["quantization_config"], config_class)
    assert first["quantization_config"].__dict__ == second["quantization_config"].__dict__
    assert first["quantization_config"].bnb_4bit_compute_dtype is float16


@pytest.mark.parametrize(
    "quantization_config, extra_kwargs, message",
    [
        ({}, {}, "load_in_4bit"),
        ({"load_in_4bit": True}, {}, "bnb_4bit_quant_type"),
        (
            {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "fp4",
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_use_double_quant": True,
            },
            {},
            "quant_type",
        ),
        (
            {
                "load_in_4bit": True,
                "load_in_8bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_use_double_quant": True,
            },
            {},
            "load_in_8bit",
        ),
        (
            {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_use_double_quant": True,
            },
            {},
            "compute_dtype",
        ),
        (
            {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_use_double_quant": 1,
            },
            {},
            "use_double_quant",
        ),
        (
            {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_use_double_quant": True,
                "llm_int8_threshold": 6.0,
            },
            {},
            "unknown fields",
        ),
        (
            {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_use_double_quant": True,
            },
            {"load_in_4bit": True},
            "legacy top-level",
        ),
    ],
)
def test_nf4_rejects_malformed_configuration(
    monkeypatch, quantization_config, extra_kwargs, message
) -> None:
    _install_fake_nf4_runtime(monkeypatch)
    config = _nf4_model_config()
    config["model_kwargs"] = {
        "quantization_config": quantization_config,
        **extra_kwargs,
    }

    with pytest.raises(InferenceConfigurationError, match=message):
        load_transformers_model("base", config)


def test_nf4_rejects_runtime_configuration_mismatch(monkeypatch) -> None:
    _install_fake_nf4_runtime(monkeypatch, runtime_overrides={"bnb_4bit_quant_type": "fp4"})

    with pytest.raises(InferenceConfigurationError, match="runtime NF4 configuration mismatch"):
        load_transformers_model("base", _nf4_model_config())


def test_nf4_balanced_device_map_uses_both_gpus_through_peft_wrapper(monkeypatch) -> None:
    _install_fake_nf4_runtime(
        monkeypatch,
        device_map={"visual": 0, "language.layers.0": "cuda:1"},
    )
    config = _nf4_model_config()
    config["model_kwargs"]["device_map"] = "balanced"

    load_transformers_model(
        "tuned",
        {
            **config,
            "adapter_id": "org/pinned-adapter",
            "adapter_revision": "adapter-commit",
            "adapter_name": "science",
        },
    )


@pytest.mark.parametrize(
    "device_map",
    [
        {"visual": 0, "language": 0},
        {"visual": 0, "language": 1, "head": "cpu"},
        {"visual": 0, "language": 1, "head": "cuda:2"},
    ],
)
def test_nf4_balanced_device_map_rejects_incomplete_or_cpu_placement(
    monkeypatch, device_map
) -> None:
    _install_fake_nf4_runtime(monkeypatch, device_map=device_map)
    config = _nf4_model_config()
    config["model_kwargs"]["device_map"] = "balanced"

    with pytest.raises(InferenceConfigurationError, match="exactly CUDA devices 0 and 1"):
        load_transformers_model("base", config)


def test_nf4_requires_bitsandbytes_package(monkeypatch) -> None:
    _install_fake_nf4_runtime(monkeypatch)
    real_import = inference_module.importlib.import_module

    def import_without_bitsandbytes(module_name):
        if module_name == "bitsandbytes":
            raise ImportError("package is not installed")
        return real_import(module_name)

    monkeypatch.setattr(inference_module.importlib, "import_module", import_without_bitsandbytes)

    with pytest.raises(InferenceConfigurationError, match="bitsandbytes.*required"):
        load_transformers_model("base", _nf4_model_config())


def test_nonquantized_model_kwargs_are_unchanged(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeBase:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(kwargs)
            return cls()

        def eval(self):
            return self

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.Qwen3VLForConditionalGeneration = FakeBase
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    load_transformers_model(
        "base",
        {
            "base_model_id": "Qwen/pinned-base",
            "base_revision": "base-commit",
            "model_kwargs": {"device_map": "cpu"},
        },
    )

    assert calls == [{"revision": "base-commit", "device_map": "cpu"}]


def _install_fake_fp16_runtime(
    monkeypatch,
    *,
    base_dtype: str = "float16",
    adapter_dtype: str = "float32",
    device_map=None,
    missing_keys=None,
    unexpected_keys=None,
    base_quantized: bool = False,
    base_floating: bool = True,
    active_adapters=None,
):
    calls: dict[str, object] = {}
    float16 = object()
    float32 = object()
    dtypes = {"float16": float16, "float32": float32}

    class FakeParameter:
        def __init__(self, dtype, floating=True):
            self.dtype = dtype
            self.floating = floating

        def is_floating_point(self):
            return self.floating

    class FakeBase:
        def __init__(self):
            self.hf_device_map = device_map or {"visual": 0, "language": 1}
            self.parameter = FakeParameter(dtypes[base_dtype], base_floating)
            self.is_quantized = base_quantized

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls["base"] = (model_id, kwargs)
            return cls()

        def named_parameters(self):
            return iter([("model.weight", self.parameter)])

        def eval(self):
            return self

    class FakePeft:
        def __init__(self, base, adapter_name):
            self.base_model = base
            self.active_adapter = adapter_name
            self.active_adapters = list(active_adapters or [adapter_name])
            self.peft_config = {adapter_name: object()}
            self.adapter_name = adapter_name
            self.adapter_parameter = FakeParameter(dtypes[adapter_dtype])

        @classmethod
        def from_pretrained(cls, base, adapter_id, **kwargs):
            calls["adapter"] = (adapter_id, kwargs)
            return cls(base, kwargs["adapter_name"])

        def load_adapter(self, adapter_id, **kwargs):
            calls["adapter_reload"] = (adapter_id, kwargs)
            return types.SimpleNamespace(
                missing_keys=list(missing_keys or []),
                unexpected_keys=list(unexpected_keys or []),
            )

        def named_parameters(self):
            yield from self.base_model.named_parameters()
            yield (
                f"base_model.model.q_proj.lora_A.{self.adapter_name}.weight",
                self.adapter_parameter,
            )

        def eval(self):
            return self

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.Qwen3VLForConditionalGeneration = FakeBase
    fake_peft = types.ModuleType("peft")
    fake_peft.PeftModel = FakePeft
    fake_torch = types.ModuleType("torch")
    fake_torch.float16 = float16
    fake_torch.float32 = float32
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return calls, float16


def _fp16_model_config() -> dict[str, object]:
    return {
        "base_model_id": "Qwen/pinned-base",
        "base_revision": "base-commit",
        "model_kwargs": {
            "torch_dtype": "float16",
            "device_map": "balanced",
        },
    }


def test_fp16_primary_verifies_base_adapter_and_balanced_map(monkeypatch) -> None:
    calls, float16 = _install_fake_fp16_runtime(monkeypatch)
    config = _fp16_model_config()

    load_transformers_model("base", config)
    load_transformers_model(
        "tuned",
        {
            **config,
            "adapter_id": "org/pinned-adapter",
            "adapter_revision": "adapter-commit",
            "adapter_name": "science",
            "adapter_kwargs": {"autocast_adapter_dtype": True},
        },
    )

    assert calls["base"][1]["torch_dtype"] is float16
    assert calls["adapter"][1]["autocast_adapter_dtype"] is True


@pytest.mark.parametrize(
    ("base_dtype", "adapter_dtype", "message"),
    [
        ("float32", "float32", "base contains non-FP16"),
        ("float16", "float16", "non-FP32 LoRA"),
    ],
)
def test_fp16_primary_rejects_runtime_dtype_mismatch(
    monkeypatch, base_dtype: str, adapter_dtype: str, message: str
) -> None:
    _install_fake_fp16_runtime(
        monkeypatch,
        base_dtype=base_dtype,
        adapter_dtype=adapter_dtype,
    )
    config = _fp16_model_config()
    target = "base" if base_dtype != "float16" else "tuned"
    if target == "tuned":
        config.update(
            {
                "adapter_id": "org/pinned-adapter",
                "adapter_revision": "adapter-commit",
                "adapter_kwargs": {"autocast_adapter_dtype": True},
            }
        )
    with pytest.raises(InferenceConfigurationError, match=message):
        load_transformers_model(target, config)


@pytest.mark.parametrize(
    ("missing_keys", "unexpected_keys"),
    [(["base_model.q_proj.lora_A.science.weight"], []), ([], ["unknown.weight"])],
)
def test_tuned_loader_rejects_incompatible_peft_checkpoint_keys(
    monkeypatch, missing_keys: list[str], unexpected_keys: list[str]
) -> None:
    _install_fake_fp16_runtime(
        monkeypatch,
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
    )
    config = {
        **_fp16_model_config(),
        "adapter_id": "org/pinned-adapter",
        "adapter_revision": "adapter-commit",
        "adapter_name": "science",
        "adapter_kwargs": {"autocast_adapter_dtype": True},
    }

    with pytest.raises(InferenceConfigurationError, match="incompatible adapter keys"):
        load_transformers_model("tuned", config)


@pytest.mark.parametrize(
    ("runtime_kwargs", "message"),
    [
        ({"base_quantized": True}, "quantization metadata"),
        ({"base_floating": False}, "non-floating parameters"),
    ],
)
def test_fp16_primary_rejects_other_quantization_or_packed_parameters(
    monkeypatch, runtime_kwargs: dict[str, bool], message: str
) -> None:
    _install_fake_fp16_runtime(monkeypatch, **runtime_kwargs)
    with pytest.raises(InferenceConfigurationError, match=message):
        load_transformers_model("base", _fp16_model_config())


def test_tuned_loader_rejects_multiple_active_adapters(monkeypatch) -> None:
    _install_fake_fp16_runtime(monkeypatch, active_adapters=["science", "other"])
    config = {
        **_fp16_model_config(),
        "adapter_id": "org/pinned-adapter",
        "adapter_revision": "adapter-commit",
        "adapter_name": "science",
        "adapter_kwargs": {"autocast_adapter_dtype": True},
    }
    with pytest.raises(InferenceConfigurationError, match="only active adapter"):
        load_transformers_model("tuned", config)


def test_tuned_loader_rejects_integer_adapter_boolean(monkeypatch) -> None:
    _install_fake_fp16_runtime(monkeypatch)
    config = {
        **_fp16_model_config(),
        "adapter_id": "org/pinned-adapter",
        "adapter_revision": "adapter-commit",
        "adapter_name": "science",
        "adapter_kwargs": {"autocast_adapter_dtype": 1},
    }

    with pytest.raises(InferenceConfigurationError, match="must be boolean"):
        load_transformers_model("tuned", config)


def test_tuned_loader_rechecks_prepared_adapter_checkpoint_attestation(
    monkeypatch, tmp_path: Path
) -> None:
    calls, _ = _install_fake_fp16_runtime(monkeypatch)
    checkpoint = tmp_path / "adapter_model.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    attestation = {"artifact_version": 1, "sha256": "a" * 64}
    fake_hub = types.ModuleType("huggingface_hub")

    def fake_download(**kwargs):
        calls["checkpoint_download"] = kwargs
        return str(checkpoint)

    fake_hub.hf_hub_download = fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    def fake_verify(path, expected, **kwargs):
        calls["checkpoint_verify"] = (path, expected, kwargs)
        return dict(expected)

    monkeypatch.setattr(
        inference_module,
        "verify_plain_fp32_lora_safetensors",
        fake_verify,
    )
    config = {
        **_fp16_model_config(),
        "adapter_id": "org/pinned-adapter",
        "adapter_revision": "adapter-commit",
        "adapter_name": "science",
        "adapter_kwargs": {"autocast_adapter_dtype": True},
        "adapter_checkpoint_attestation": attestation,
    }

    load_transformers_model("tuned", config)

    assert calls["checkpoint_download"] == {
        "repo_id": "org/pinned-adapter",
        "repo_type": "model",
        "revision": "adapter-commit",
        "filename": "adapter_model.safetensors",
    }
    assert calls["checkpoint_verify"] == (
        str(checkpoint),
        attestation,
        {
            "repo_id": "org/pinned-adapter",
            "revision": "adapter-commit",
            "error_type": InferenceConfigurationError,
        },
    )


def test_runtime_environment_records_bitsandbytes(monkeypatch) -> None:
    versions = {"bitsandbytes": "0.48.1"}
    monkeypatch.setattr(
        inference_module.importlib.metadata,
        "version",
        lambda distribution: versions.get(distribution) or "test-version",
    )

    environment = inference_module._runtime_environment("mock")

    assert environment["packages"]["bitsandbytes"] == "0.48.1"


def _kaggle_runtime() -> dict[str, object]:
    return {
        "gpu_count": 2,
        "gpu_names": ["Tesla T4", "NVIDIA T4"],
        "cuda_version": "12.6",
        "packages": {
            **inference_module._KAGGLE_RUNTIME_VERSIONS,
            "torch": "2.3.1+cu121",
            "bitsandbytes": "0.48.1",
        },
    }


def test_publication_runtime_requires_exact_t4x2_and_pinned_packages() -> None:
    inference_module._validate_kaggle_runtime_environment(
        _kaggle_runtime(),
        "publication_candidate",
    )
    inference_module._validate_kaggle_runtime_environment(
        _kaggle_runtime(),
        "automatic_sensitivity_only",
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda runtime: runtime.update(gpu_names=["Tesla T4"]), "two NVIDIA T4"),
        (lambda runtime: runtime.update(gpu_names=["A100", "A100"]), "two NVIDIA T4"),
        (
            lambda runtime: runtime["packages"].update(peft="0.18.0"),
            "peft='0.18.0'",
        ),
        (
            lambda runtime: runtime["packages"].pop("bitsandbytes"),
            "bitsandbytes=None",
        ),
    ],
)
def test_publication_runtime_rejects_hardware_or_package_drift(mutate, message: str) -> None:
    runtime = _kaggle_runtime()
    mutate(runtime)
    scope = "automatic_sensitivity_only" if "bitsandbytes" in message else "publication_candidate"

    with pytest.raises(InferenceConfigurationError, match=message):
        inference_module._validate_kaggle_runtime_environment(runtime, scope)


def test_engine_version_change_invalidates_resume(tmp_path: Path, monkeypatch) -> None:
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
    assert inference_module.ENGINE_VERSION == "5"
    monkeypatch.setattr(inference_module, "ENGINE_VERSION", "previous-engine")

    with pytest.raises(InferenceConfigurationError, match="config fingerprint mismatch"):
        run_inference(**kwargs)
    assert output.read_bytes() == before


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
