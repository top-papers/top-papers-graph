from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


class _Dummy:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls(*args, **kwargs)


def _install_dpo_training_stubs(monkeypatch):
    datasets = types.ModuleType("datasets")
    datasets.Dataset = _Dummy
    datasets.Image = _Dummy
    datasets.Sequence = lambda feature: ("sequence", feature)
    datasets.load_dataset = lambda *args, **kwargs: None

    peft = types.ModuleType("peft")
    peft.LoraConfig = _Dummy
    peft.PeftModel = _Dummy
    peft.get_peft_model = lambda model, config: model
    peft.prepare_model_for_kbit_training = lambda model: model

    transformers = types.ModuleType("transformers")
    for attr in [
        "AutoProcessor",
        "AutoTokenizer",
        "BitsAndBytesConfig",
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen3VLForConditionalGeneration",
    ]:
        setattr(transformers, attr, _Dummy)
    transformers.set_seed = lambda seed: None

    trl = types.ModuleType("trl")
    trl.DPOConfig = _Dummy
    trl.DPOTrainer = _Dummy

    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    torch.float16 = "float16"
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    monkeypatch.setitem(sys.modules, "datasets", datasets)
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "trl", trl)
    monkeypatch.setitem(sys.modules, "torch", torch)


def _load_dpo_script(module_name: str):
    path = Path("experiments/vlm_finetuning/scripts/train_vlm_dpo.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dpo_resolves_auto_attention_to_supported_backend(monkeypatch):
    _install_dpo_training_stubs(monkeypatch)
    mod = _load_dpo_script("train_vlm_dpo_attn_resolution_test")

    monkeypatch.setattr(mod.importlib.util, "find_spec", lambda name: None if name == "flash_attn" else object())

    assert mod.resolve_attn_implementation("auto") == "sdpa"
    assert mod.resolve_attn_implementation("eager") == "eager"


def test_dpo_model_loader_never_forwards_auto_attention(monkeypatch):
    _install_dpo_training_stubs(monkeypatch)
    mod = _load_dpo_script("train_vlm_dpo_model_loader_attn_test")
    monkeypatch.setattr(mod.importlib.util, "find_spec", lambda name: None if name == "flash_attn" else object())

    model = mod.load_qwen_model(
        "Qwen/Qwen3-VL-8B-Instruct",
        qlora=False,
        bf16=True,
        fp16=False,
        trust_remote_code=True,
        attn_impl="auto",
    )

    assert model.kwargs["attn_implementation"] == "sdpa"
    assert model.kwargs["trust_remote_code"] is True


def test_dpo_disables_precompute_ref_log_probs_for_vlm(monkeypatch):
    _install_dpo_training_stubs(monkeypatch)
    mod = _load_dpo_script("train_vlm_dpo_precompute_vlm_guard_test")

    requested = types.SimpleNamespace(precompute_ref_log_probs=True)
    assert mod.resolve_precompute_ref_log_probs(requested, "vlm") is False
    assert mod.resolve_precompute_ref_log_probs(requested, "text") is True

    not_requested = types.SimpleNamespace(precompute_ref_log_probs=False)
    assert mod.resolve_precompute_ref_log_probs(not_requested, "vlm") is False


def test_dpo_run_config_json_handles_path_arguments(monkeypatch):
    _install_dpo_training_stubs(monkeypatch)
    mod = _load_dpo_script("train_vlm_dpo_run_config_json_test")

    run_config = {
        "output_dir": Path("outputs/dpo"),
        "train_file": Path("data/dpo_train.jsonl"),
        "resolved_mode": "vlm",
    }

    encoded = mod.json.dumps(run_config, ensure_ascii=False, indent=2, default=str)
    assert "outputs/dpo" in encoded
    assert "data/dpo_train.jsonl" in encoded



def _count_image_placeholders(messages):
    count = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            count += sum(1 for block in content if isinstance(block, dict) and block.get("type") == "image")
    return count


class _MiniDataset:
    def __init__(self, rows):
        self.rows = list(rows)

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)

    def map(self, fn, desc=None):
        return _MiniDataset([fn(dict(row)) for row in self.rows])


def test_dpo_formatter_aligns_prompt_placeholders_to_images(monkeypatch, tmp_path):
    _install_dpo_training_stubs(monkeypatch)
    mod = _load_dpo_script("train_vlm_dpo_placeholder_formatter_test")

    formatter = mod.make_dpo_formatter(tmp_path)
    row = {
        "images": ["fig1.png", "fig2.png"],
        "image": "fig2.png",
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": "compare the figures"},
                ],
            }
        ],
        "chosen": "better answer",
        "rejected": "worse answer",
    }

    formatted = formatter(row)

    assert len(formatted["images"]) == 2
    assert _count_image_placeholders(formatted["prompt"]) == len(formatted["images"])
    assert formatted["chosen"] == [{"role": "assistant", "content": "better answer"}]
    assert formatted["rejected"] == [{"role": "assistant", "content": "worse answer"}]


def test_dpo_image_cap_realigns_prompt_placeholders(monkeypatch, tmp_path):
    _install_dpo_training_stubs(monkeypatch)
    mod = _load_dpo_script("train_vlm_dpo_placeholder_cap_test")

    row = {
        "images": [str(tmp_path / f"fig{i}.png") for i in range(4)],
        "image": str(tmp_path / "fig0.png"),
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": "answer using the evidence"},
                ],
            }
        ],
        "chosen": "yes",
        "rejected": "no",
    }

    capped, stats = mod.cap_dpo_images_for_memory(_MiniDataset([row]), 1, "train")
    capped_row = capped.rows[0]

    assert stats["truncated_rows"] == 1
    assert len(capped_row["images"]) == 1
    assert _count_image_placeholders(capped_row["prompt"]) == 1


class _MiniColumnsDataset:
    def __init__(self, rows, features=None):
        self.rows = [dict(row) for row in rows]
        self.features = dict(features or {key: object() for key in self.rows[0]})
        self.column_names = list(self.features)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        if isinstance(item, slice):
            subset = self.rows[item]
            return {key: [row.get(key) for row in subset] for key in self.column_names}
        return self.rows[item]

    def cast(self, features):
        self.features = dict(features)
        self.column_names = list(self.features)
        return self

    def cast_column(self, column, feature):
        self.features[column] = feature
        if column not in self.column_names:
            self.column_names.append(column)
        return self

    def remove_columns(self, columns):
        columns = set(columns)
        self.rows = [{key: value for key, value in row.items() if key not in columns} for row in self.rows]
        self.features = {key: value for key, value in self.features.items() if key not in columns}
        self.column_names = [key for key in self.column_names if key not in columns]
        return self


def test_dpo_vlm_plural_images_path_drops_singleton_image_column(monkeypatch):
    _install_dpo_training_stubs(monkeypatch)
    mod = _load_dpo_script("train_vlm_dpo_drop_singleton_image_test")

    ds = _MiniColumnsDataset([
        {
            "prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": "compare"}]}],
            "chosen": [{"role": "assistant", "content": "good"}],
            "rejected": [{"role": "assistant", "content": "bad"}],
            "images": ["fig1.png", "fig2.png"],
            "image": "fig1.png",
        }
    ])

    prepared, mode = mod.maybe_prepare_dataset(ds, "images", "vlm")

    assert mode == "vlm"
    assert "images" in prepared.column_names
    assert "image" not in prepared.column_names


def test_dpo_singleton_image_column_would_collapse_plural_images_without_guard():
    example = {"images": ["fig1.png", "fig2.png", "fig3.png"], "image": "fig1.png"}

    # Mirrors TRL's vision DPO collator branch: if ``image`` is present, it
    # overwrites the plural list with a one-element list. Our dataset guard must
    # therefore remove ``image`` on the plural-image path before collation.
    if "image" in example:
        example["images"] = [example.pop("image")]

    assert example["images"] == ["fig1.png"]
