from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


class _Dummy:
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


def _install_training_stubs(monkeypatch):
    datasets = types.ModuleType("datasets")
    datasets.Image = _Dummy
    datasets.Sequence = lambda feature: ("sequence", feature)
    datasets.load_dataset = lambda *args, **kwargs: None

    peft = types.ModuleType("peft")
    peft.LoraConfig = _Dummy
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
    for attr in ["SFTConfig", "SFTTrainer", "GRPOConfig", "GRPOTrainer"]:
        setattr(trl, attr, _Dummy)

    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    torch.float16 = "float16"
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    monkeypatch.setitem(sys.modules, "datasets", datasets)
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "trl", trl)
    monkeypatch.setitem(sys.modules, "torch", torch)


def _load_script(path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sample_chat():
    return {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "sys", "trainable": False}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "question"},
                    {"type": "image", "image": "/tmp/page_001.png", "meta": {"page": 1}},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "answer", "trainable": True}]},
        ]
    }


def test_sft_normalizes_chat_to_messages_and_images(monkeypatch):
    _install_training_stubs(monkeypatch)
    mod = _load_script("experiments/vlm_finetuning/scripts/train_vlm_sft.py", "train_vlm_sft_test")

    row = {"chat": _sample_chat()}
    out = mod._normalise_sft_example(row)

    assert out["messages"][0] == {"role": "system", "content": "sys"}
    assert out["messages"][1]["content"] == [
        {"type": "text", "text": "question"},
        {"type": "image"},
    ]
    assert out["images"] == ["/tmp/page_001.png"]


def test_grpo_normalizes_prompt_chat_to_prompt_and_images(monkeypatch):
    _install_training_stubs(monkeypatch)
    mod = _load_script("experiments/vlm_finetuning/scripts/train_vlm_grpo.py", "train_vlm_grpo_test")

    row = {"prompt_chat": _sample_chat()}
    out = mod.format_grpo_keys(row)

    assert out["prompt"][0] == {"role": "system", "content": "sys"}
    assert out["prompt"][1]["content"] == [
        {"type": "text", "text": "question"},
        {"type": "image"},
    ]
    assert out["images"] == ["/tmp/page_001.png"]
