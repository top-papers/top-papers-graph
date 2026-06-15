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
        "AutoModelForImageTextToText",
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

    assert out["messages"][0] == {"role": "system", "content": [{"type": "text", "text": "sys"}]}
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



def test_sft_converts_raw_non_dict_content_to_text_blocks(monkeypatch):
    _install_training_stubs(monkeypatch)
    mod = _load_script("experiments/vlm_finetuning/scripts/train_vlm_sft.py", "train_vlm_sft_raw_content_test")

    row = {
        "messages": [
            {
                "role": "user",
                "content": [
                    123,
                    {"type": "image", "image": "/tmp/page_002.png"},
                    {"unexpected": 456},
                ],
            },
            {"role": "assistant", "content": [789]},
        ]
    }
    out = mod._normalise_sft_example(row)

    assert out["messages"][0]["content"] == [
        {"type": "text", "text": "123"},
        {"type": "image"},
        {"type": "text", "text": '{"unexpected": 456}'},
    ]
    assert out["messages"][1]["content"] == [{"type": "text", "text": "789"}]
    assert out["images"] == ["/tmp/page_002.png"]


def test_grpo_training_formatter_converts_raw_non_dict_content_to_text_blocks(monkeypatch):
    _install_training_stubs(monkeypatch)
    mod = _load_script("experiments/vlm_finetuning/scripts/train_vlm_grpo.py", "train_vlm_grpo_raw_content_test")

    formatter = mod.make_grpo_formatter(Path("."))
    row = {
        "prompt": [
            {
                "role": "user",
                "content": [
                    123,
                    {"type": "image", "image": "/tmp/page_003.png"},
                    {"unexpected": 456},
                ],
            }
        ]
    }
    out = formatter(row)

    assert out["prompt"][0]["content"] == [
        {"type": "text", "text": "123"},
        {"type": "image"},
        {"type": "text", "text": '{"unexpected": 456}'},
    ]
    assert out["images"] == ["/tmp/page_003.png"]


def test_sft_vlm_disables_assistant_only_loss_before_sftconfig(monkeypatch, capsys):
    _install_training_stubs(monkeypatch)
    mod = _load_script("experiments/vlm_finetuning/scripts/train_vlm_sft.py", "train_vlm_sft_assistant_only_test")

    args = types.SimpleNamespace(assistant_only_loss=True)
    mod.disable_unsupported_vlm_assistant_only_loss(args, "vlm")

    assert args.assistant_only_loss is False
    assert "assistant_only_loss=True is not supported" in capsys.readouterr().out

    text_args = types.SimpleNamespace(assistant_only_loss=True)
    mod.disable_unsupported_vlm_assistant_only_loss(text_args, "text")
    assert text_args.assistant_only_loss is True



def test_sft_aligns_image_placeholders_to_capped_images(monkeypatch):
    _install_training_stubs(monkeypatch)
    mod = _load_script("experiments/vlm_finetuning/scripts/train_vlm_sft.py", "train_vlm_sft_placeholder_align_test")

    row = {
        "images": ["/tmp/kept_1.png", "/tmp/kept_2.png", "/tmp/kept_3.png"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": "compare"},
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "image"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        ],
    }

    out = mod._sanitize_sft_row_for_trl(row)
    assert mod._count_image_placeholders(out["messages"]) == 3
    assert out["messages"][0]["content"] == [
        {"type": "image"},
        {"type": "image"},
        {"type": "text", "text": "compare"},
        {"type": "image"},
    ]


def test_sft_adds_missing_placeholders_for_images(monkeypatch):
    _install_training_stubs(monkeypatch)
    mod = _load_script("experiments/vlm_finetuning/scripts/train_vlm_sft.py", "train_vlm_sft_placeholder_add_test")

    row = {
        "images": ["/tmp/kept_1.png", "/tmp/kept_2.png"],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "describe"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        ],
    }

    out = mod._sanitize_sft_row_for_trl(row)
    assert mod._count_image_placeholders(out["messages"]) == 2
    assert out["messages"][0]["content"][:2] == [{"type": "image"}, {"type": "image"}]


def test_grpo_aligns_image_placeholders_to_capped_images(monkeypatch):
    _install_training_stubs(monkeypatch)
    mod = _load_script("experiments/vlm_finetuning/scripts/train_vlm_grpo.py", "train_vlm_grpo_placeholder_align_test")

    formatter = mod.make_grpo_formatter(Path("."))
    row = {
        "images": ["/tmp/kept_1.png", "/tmp/kept_2.png"],
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "review"},
                    {"type": "image"},
                    {"type": "image"},
                ],
            }
        ],
    }
    out = formatter(row)
    placeholders = sum(
        1
        for msg in out["prompt"]
        for block in msg.get("content", [])
        if isinstance(block, dict) and block.get("type") == "image"
    )
    assert placeholders == 2



def test_sft_auto_enables_ddp_find_unused_for_distributed_vlm(monkeypatch):
    _install_training_stubs(monkeypatch)
    monkeypatch.setenv("WORLD_SIZE", "2")
    mod = _load_script("experiments/vlm_finetuning/scripts/train_vlm_sft.py", "train_vlm_sft_ddp_unused_test")

    args = types.SimpleNamespace(ddp_find_unused_parameters=None)
    assert mod.resolve_ddp_find_unused_parameters(args, "vlm") is True
    assert mod.resolve_ddp_find_unused_parameters(args, "text") is False

    forced = types.SimpleNamespace(ddp_find_unused_parameters=False)
    assert mod.resolve_ddp_find_unused_parameters(forced, "vlm") is False


def test_grpo_auto_enables_ddp_find_unused_for_distributed_vlm(monkeypatch):
    _install_training_stubs(monkeypatch)
    monkeypatch.setenv("WORLD_SIZE", "2")
    mod = _load_script("experiments/vlm_finetuning/scripts/train_vlm_grpo.py", "train_vlm_grpo_ddp_unused_test")

    args = types.SimpleNamespace(ddp_find_unused_parameters=None)
    assert mod.resolve_ddp_find_unused_parameters(args, "vlm") is True
    assert mod.resolve_ddp_find_unused_parameters(args, "text") is False

    forced = types.SimpleNamespace(ddp_find_unused_parameters=False)
    assert mod.resolve_ddp_find_unused_parameters(forced, "vlm") is False
