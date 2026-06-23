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
