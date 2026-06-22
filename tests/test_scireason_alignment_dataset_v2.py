from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_builder(monkeypatch):
    hub = ModuleType("huggingface_hub")
    hub.snapshot_download = lambda *args, **kwargs: "unused"
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    path = Path("experiments/vlm_finetuning/scripts/build_scireason_alignment_datasets.py")
    spec = importlib.util.spec_from_file_location("build_scireason_alignment_datasets_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_leakage_safe_split_keeps_source_group_on_one_side(monkeypatch):
    mod = _load_builder(monkeypatch)
    rows = [
        {"id": "a1", "task_family": "trajectory_reasoning", "metadata": {"source_file": "paper_a.json"}},
        {"id": "a2", "task_family": "assertion_reconstruction", "metadata": {"source_file": "paper_a.json"}},
        {"id": "b1", "task_family": "trajectory_reasoning", "metadata": {"source_file": "paper_b.json"}},
        {"id": "c1", "task_family": "trajectory_reasoning", "metadata": {"source_file": "paper_c.json"}},
    ]
    train, eval_rows, report = mod.deterministic_group_split(rows, eval_ratio=0.5, seed=7)
    train_groups = {mod.leakage_group_key(row, i) for i, row in enumerate(train)}
    eval_groups = {mod.leakage_group_key(row, i) for i, row in enumerate(eval_rows)}
    assert train_groups.isdisjoint(eval_groups)
    assert report["group_overlap_count"] == 0


def test_relevance_image_selection_prefers_evidence_figure(monkeypatch):
    mod = _load_builder(monkeypatch)
    row = {"claim": "Figure 7 proves temporal effect", "evidence": {"figure": "figure_7"}}
    raw = ["assets/page_001.png", "assets/figure_7_panel.png", "assets/table_2.png"]
    chosen, meta = mod.select_relevant_images(raw, row, max_images=1)
    assert chosen == ["assets/figure_7_panel.png"]
    assert meta["policy"] == "dynamic_relevance_top_k_then_original_order"
    assert meta["dynamic_cap"] == 1


def test_dpo_rows_are_built_from_assistant_messages(monkeypatch):
    mod = _load_builder(monkeypatch)
    rows = [
        {
            "id": "r1",
            "task_family": "assertion_review",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "review"}]},
                {"role": "assistant", "content": [{"type": "text", "text": '{"verdict":"reject"}'}]},
            ],
        }
    ]
    dpo = mod.make_dpo_rows_from_sft(rows, synthetic_negatives=True)
    assert len(dpo) >= 2
    assert dpo[0]["chosen"] == '{"verdict":"reject"}'
    assert {row["metadata"]["pair_type"] for row in dpo}.issuperset({"verdict_flip", "evidence_drop"})
    assert all("pair_hardness" in row["metadata"] for row in dpo)


def test_dpo_rows_include_grpo_reward_target_bootstrap(monkeypatch):
    mod = _load_builder(monkeypatch)
    rows = [
        {
            "id": "g1",
            "task_family": "assertion_review_rl",
            "expected_verdict": "unsupported",
            "prompt": [{"role": "user", "content": [{"type": "text", "text": "review claim"}]}],
            "images": ["/tmp/figure_7.png"],
        }
    ]

    dpo = mod.make_dpo_rows_from_grpo(rows, synthetic_negatives=True)

    assert len(dpo) == 1
    assert '"verdict": "reject"' in dpo[0]["chosen"]
    assert '"verdict": "accept"' in dpo[0]["rejected"]
    assert dpo[0]["metadata"]["preference_source"] == "grpo_target_bootstrap"


def test_dpo_dedupe_merges_source_ids_for_full_data_audit(monkeypatch):
    mod = _load_builder(monkeypatch)
    rows = [
        {"id": "dpo:sft-a:0", "prompt": "p", "chosen": "c", "rejected": "r", "metadata": {"source_id": "sft-a", "pair_type": "verdict_flip"}},
        {"id": "dpo:sft-b:0", "prompt": "p", "chosen": "c", "rejected": "r", "metadata": {"source_id": "sft-b", "pair_type": "evidence_drop"}},
    ]

    out = mod.dedupe_dpo_rows(rows)

    assert len(out) == 1
    assert out[0]["metadata"]["source_id"] == "sft-a"
    assert out[0]["metadata"]["source_ids"] == ["sft-a", "sft-b"]
    assert out[0]["metadata"]["deduped_source_count"] == 2
    assert set(out[0]["metadata"]["pair_types"]) == {"verdict_flip", "evidence_drop"}
