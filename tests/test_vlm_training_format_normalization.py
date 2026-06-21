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
    peft.PeftModel = _Dummy
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



def test_sft_enables_ddp_find_unused_for_distributed_vlm_by_default(monkeypatch):
    _install_training_stubs(monkeypatch)
    monkeypatch.setenv("WORLD_SIZE", "2")
    mod = _load_script("experiments/vlm_finetuning/scripts/train_vlm_sft.py", "train_vlm_sft_ddp_unused_test")

    args = types.SimpleNamespace(ddp_find_unused_parameters=None)
    assert mod.resolve_ddp_find_unused_parameters(args, "vlm") is True
    assert mod.resolve_ddp_find_unused_parameters(args, "text") is False

    monkeypatch.setenv("WORLD_SIZE", "1")
    assert mod.resolve_ddp_find_unused_parameters(args, "vlm") is False
    monkeypatch.setenv("WORLD_SIZE", "2")

    forced_on = types.SimpleNamespace(ddp_find_unused_parameters=True)
    assert mod.resolve_ddp_find_unused_parameters(forced_on, "vlm") is True
    forced_off = types.SimpleNamespace(ddp_find_unused_parameters=False)
    assert mod.resolve_ddp_find_unused_parameters(forced_off, "vlm") is False


def test_grpo_enables_ddp_find_unused_for_distributed_vlm_by_default(monkeypatch):
    _install_training_stubs(monkeypatch)
    monkeypatch.setenv("WORLD_SIZE", "2")
    mod = _load_script("experiments/vlm_finetuning/scripts/train_vlm_grpo.py", "train_vlm_grpo_ddp_unused_test")

    args = types.SimpleNamespace(ddp_find_unused_parameters=None)
    assert mod.resolve_ddp_find_unused_parameters(args, "vlm") is True
    assert mod.resolve_ddp_find_unused_parameters(args, "text") is False

    monkeypatch.setenv("WORLD_SIZE", "1")
    assert mod.resolve_ddp_find_unused_parameters(args, "vlm") is False
    monkeypatch.setenv("WORLD_SIZE", "2")

    forced_on = types.SimpleNamespace(ddp_find_unused_parameters=True)
    assert mod.resolve_ddp_find_unused_parameters(forced_on, "vlm") is True
    forced_off = types.SimpleNamespace(ddp_find_unused_parameters=False)
    assert mod.resolve_ddp_find_unused_parameters(forced_off, "vlm") is False


class _TinyDataset:
    def __init__(self, rows):
        self.rows = [dict(row) for row in rows]
        self.features = {key: object() for row in self.rows for key in row.keys()}

    @property
    def column_names(self):
        keys = []
        for row in self.rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        return keys

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            rows = self.rows[idx]
            return {key: [row.get(key) for row in rows] for key in self.column_names}
        return self.rows[idx]

    def add_column(self, name, values):
        assert len(values) == len(self.rows)
        rows = [dict(row, **{name: value}) for row, value in zip(self.rows, values)]
        return _TinyDataset(rows)

    def remove_columns(self, columns):
        drop = set(columns)
        rows = [{key: value for key, value in row.items() if key not in drop} for row in self.rows]
        return _TinyDataset(rows)

    def map(self, fn, *args, **kwargs):
        rows = []
        for row in self.rows:
            update = fn(dict(row))
            merged = dict(row)
            if update:
                merged.update(update)
            rows.append(merged)
        return _TinyDataset(rows)

    def filter(self, fn, *args, **kwargs):
        return _TinyDataset([row for row in self.rows if fn(dict(row))])

    def cast(self, features):
        self.features = dict(features)
        return self

    def cast_column(self, column, feature):
        self.features[column] = feature
        return self


def test_sft_eval_vlm_keeps_empty_images_column_for_text_only_eval(monkeypatch):
    _install_training_stubs(monkeypatch)
    mod = _load_script("experiments/vlm_finetuning/scripts/train_vlm_sft.py", "train_vlm_sft_eval_images_test")

    ds = _TinyDataset([
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "text-only eval"}]}]},
    ])
    prepared, mode = mod.maybe_prepare_dataset(ds, "images", "vlm")

    assert mode == "vlm"
    assert "images" in prepared.column_names
    assert prepared[0]["images"] == []


def test_grpo_eval_vlm_keeps_empty_images_column_for_text_only_eval(monkeypatch):
    _install_training_stubs(monkeypatch)
    mod = _load_script("experiments/vlm_finetuning/scripts/train_vlm_grpo.py", "train_vlm_grpo_eval_images_test")

    ds = _TinyDataset([
        {"prompt": [{"role": "user", "content": [{"type": "text", "text": "text-only eval"}]}]},
    ])
    prepared, mode = mod.maybe_prepare_dataset(ds, "images", "vlm")

    assert mode == "vlm"
    assert "images" in prepared.column_names
    assert prepared[0]["images"] == []



def test_grpo_vlm_filter_drops_text_only_rows_before_qwen_processor(monkeypatch):
    _install_training_stubs(monkeypatch)
    mod = _load_script("experiments/vlm_finetuning/scripts/train_vlm_grpo.py", "train_vlm_grpo_filter_text_only_test")

    ds = _TinyDataset([
        {"prompt": [{"role": "user", "content": [{"type": "text", "text": "text-only"}]}], "images": []},
        {"prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "has image"}]}], "images": ["/tmp/page_001.png"]},
        {"prompt": [{"role": "user", "content": [{"type": "text", "text": "none image"}]}], "images": [None]},
    ])

    filtered = mod._filter_text_only_vlm_grpo_rows(ds, "train", required=True)

    assert len(filtered) == 1
    assert filtered[0]["images"] == ["/tmp/page_001.png"]


def test_grpo_vlm_filter_fails_fast_when_forced_vlm_has_no_images(monkeypatch):
    _install_training_stubs(monkeypatch)
    mod = _load_script("experiments/vlm_finetuning/scripts/train_vlm_grpo.py", "train_vlm_grpo_filter_required_test")

    ds = _TinyDataset([
        {"prompt": [{"role": "user", "content": [{"type": "text", "text": "text-only"}]}], "images": []},
    ])

    try:
        mod._filter_text_only_vlm_grpo_rows(ds, "train", required=True)
    except ValueError as exc:
        assert "no rows with images" in str(exc)
    else:
        raise AssertionError("expected forced VLM GRPO split without images to fail fast")

def test_grpo_installs_fsdpmodule_alias_for_torch_25_import(monkeypatch):
    _install_training_stubs(monkeypatch)

    torch_mod = sys.modules["torch"]
    torch_mod.__path__ = []
    distributed = types.ModuleType("torch.distributed")
    fsdp = types.ModuleType("torch.distributed.fsdp")

    class LegacyFullyShardedDataParallel:
        pass

    fsdp.FullyShardedDataParallel = LegacyFullyShardedDataParallel
    distributed.fsdp = fsdp
    torch_mod.distributed = distributed
    monkeypatch.setitem(sys.modules, "torch.distributed", distributed)
    monkeypatch.setitem(sys.modules, "torch.distributed.fsdp", fsdp)

    mod = _load_script(
        "experiments/vlm_finetuning/scripts/train_vlm_grpo.py",
        "train_vlm_grpo_fsdpmodule_compat_test",
    )

    assert mod.install_torch_fsdp_module_import_compat() is False
    assert fsdp.FSDPModule is LegacyFullyShardedDataParallel

def test_grpo_defines_main_process_helper_for_checkpoint_hooks(monkeypatch):
    _install_training_stubs(monkeypatch)
    monkeypatch.setenv("RANK", "0")
    mod = _load_script("experiments/vlm_finetuning/scripts/train_vlm_grpo.py", "train_vlm_grpo_main_process_test")

    assert mod.is_main_process() is True
    monkeypatch.setenv("RANK", "1")
    assert mod.is_main_process() is False


def test_grpo_enforces_minimum_generation_count(monkeypatch):
    _install_training_stubs(monkeypatch)
    mod = _load_script("experiments/vlm_finetuning/scripts/train_vlm_grpo.py", "train_vlm_grpo_num_gen_test")

    args = types.SimpleNamespace(num_generations=1, num_generations_eval=1)
    mod.enforce_minimum_grpo_generations(args)

    assert args.num_generations == 2
    assert args.num_generations_eval == 2

    args_ok = types.SimpleNamespace(num_generations=3, num_generations_eval=2)
    mod.enforce_minimum_grpo_generations(args_ok)
    assert args_ok.num_generations == 3
    assert args_ok.num_generations_eval == 2



def test_grpo_schema_reward_gives_dense_signal_for_truncated_json(monkeypatch):
    _install_training_stubs(monkeypatch)
    mod = _load_script("experiments/vlm_finetuning/scripts/train_vlm_grpo.py", "train_vlm_grpo_dense_reward_test")

    rewards = mod.reward_schema_validity(
        prompts=[None, None, None],
        completions=[
            "not json at all",
            '{"verdict": "reject",',
            '{"verdict": "reject", "rationale": "missing evidence"',
        ],
        task_family=["assertion_review_rl", "assertion_review_rl", "assertion_review_rl"],
    )

    assert len(set(rewards)) > 1
    assert all(-1.0 <= value < 0.5 for value in rewards)
    assert rewards[0] < rewards[1] <= rewards[2]


def test_grpo_graph_reward_is_neutral_for_review_without_predicted_graph(monkeypatch):
    _install_training_stubs(monkeypatch)
    mod = _load_script("experiments/vlm_finetuning/scripts/train_vlm_grpo.py", "train_vlm_grpo_review_graph_neutral_test")

    rewards = mod.reward_graph_consistency(
        prompts=[None],
        completions=['{"verdict": "reject", "rationale": "not supported"}'],
        reference_assertions_json=['[{"subject":"A","predicate":"causes","object":"B"}]'],
        task_family=["assertion_review_rl"],
    )

    assert rewards == [0.0]


def test_grpo_expert_reward_distinguishes_wrong_valid_verdict_from_missing(monkeypatch):
    _install_training_stubs(monkeypatch)
    mod = _load_script("experiments/vlm_finetuning/scripts/train_vlm_grpo.py", "train_vlm_grpo_dense_dict_verdict_test")

    rewards = mod.reward_expert_override_match(
        prompts=[None, None, None],
        completions=[
            '{"verdict": "reject", "rationale": "ok"}',
            '{"verdict": "accept", "rationale": "ok"}',
            '{"rationale": "ok"}',
        ],
        expected_verdict=["reject", "reject", "reject"],
        task_family=["assertion_review_rl", "assertion_review_rl", "assertion_review_rl"],
    )

    assert rewards == [1.0, -0.55, -0.9]


def test_grpo_expert_reward_uses_partial_verdict_signal_for_truncated_json(monkeypatch):
    _install_training_stubs(monkeypatch)
    mod = _load_script("experiments/vlm_finetuning/scripts/train_vlm_grpo.py", "train_vlm_grpo_partial_verdict_test")

    rewards = mod.reward_expert_override_match(
        prompts=[None, None],
        completions=['{"verdict": "reject",', '{"verdict": "accept",'],
        expected_verdict=["reject", "reject"],
        task_family=["assertion_review_rl", "assertion_review_rl"],
    )

    assert rewards[0] > rewards[1]
    assert rewards[0] < 1.0


def test_sft_text_char_guard_drops_pathological_rows(monkeypatch):
    _install_training_stubs(monkeypatch)
    mod = _load_script("experiments/vlm_finetuning/scripts/train_vlm_sft.py", "train_vlm_sft_text_guard_test")

    class FakeDataset(list):
        def filter(self, fn, desc=None):
            return FakeDataset([row for row in self if fn(row)])

    ds = FakeDataset([
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "short"}]}]},
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "x" * 50}]}]},
    ])

    filtered, dropped = mod.filter_dataset_by_text_chars(ds, 10, "train")

    assert dropped == 1
    assert len(filtered) == 1
    assert mod._row_text_chars(filtered[0]) == 5


def test_sft_text_char_guard_can_be_disabled(monkeypatch):
    _install_training_stubs(monkeypatch)
    mod = _load_script("experiments/vlm_finetuning/scripts/train_vlm_sft.py", "train_vlm_sft_text_guard_disabled_test")

    class FakeDataset(list):
        def filter(self, fn, desc=None):
            raise AssertionError("filter should not be called when the guard is disabled")

    ds = FakeDataset([{"messages": [{"role": "user", "content": "x" * 1000}]}])
    filtered, dropped = mod.filter_dataset_by_text_chars(ds, 0, "train")

    assert filtered is ds
    assert dropped == 0


def test_sft_peft_best_checkpoint_uses_safe_copy_by_default(monkeypatch):
    _install_training_stubs(monkeypatch)
    mod = _load_script("experiments/vlm_finetuning/scripts/train_vlm_sft.py", "train_vlm_sft_best_checkpoint_test")

    args = types.SimpleNamespace(load_best_model_at_end=True, native_load_best_model_at_end=False)
    model = types.SimpleNamespace(peft_config={"default": object()})

    assert mod.should_native_load_best_model(args, object(), model) is False


def test_dpo_peft_best_checkpoint_uses_safe_copy_by_default(monkeypatch):
    _install_training_stubs(monkeypatch)
    trl = sys.modules["trl"]
    trl.DPOConfig = _Dummy
    trl.DPOTrainer = _Dummy
    mod = _load_script("experiments/vlm_finetuning/scripts/train_vlm_dpo.py", "train_vlm_dpo_best_checkpoint_test")

    args = types.SimpleNamespace(load_best_model_at_end=True, native_load_best_model_at_end=False)
    model = types.SimpleNamespace(peft_config={"default": object()})

    assert mod.should_native_load_best_model(args, object(), model) is False


def test_sft_training_image_cap_prefers_evidence_named_image(monkeypatch):
    _install_training_stubs(monkeypatch)
    mod = _load_script("experiments/vlm_finetuning/scripts/train_vlm_sft.py", "train_vlm_sft_evidence_image_select_test")

    row = {
        "claim": "See figure seven for the ablation",
        "evidence": {"figure": "figure_7"},
    }
    images = ["/tmp/page_001.png", "/tmp/figure_7_panel.png", "/tmp/table_2.png"]

    assert mod._select_training_images_for_memory(row, images, 1) == ["/tmp/figure_7_panel.png"]


def test_grpo_canonical_verdict_accepts_common_aliases(monkeypatch):
    _install_training_stubs(monkeypatch)
    mod = _load_script("experiments/vlm_finetuning/scripts/train_vlm_grpo.py", "train_vlm_grpo_verdict_alias_test")

    rewards = mod.reward_expert_override_match(
        prompts=[None, None],
        completions=[
            '{"verdict": "unsupported", "rationale": "evidence contradicts it"}',
            '{"verdict": "approved", "rationale": "ok"}',
        ],
        expected_verdict=["reject", "accept"],
        task_family=["assertion_review_rl", "assertion_review_rl"],
    )

    assert rewards == [1.0, 1.0]
