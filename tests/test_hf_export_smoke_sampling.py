from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _load_builder(monkeypatch):
    datasets = types.ModuleType("datasets")
    datasets.ClassLabel = type("ClassLabel", (), {})
    datasets.load_dataset = lambda *args, **kwargs: None
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda *args, **kwargs: "."
    pil = types.ModuleType("PIL")
    image_mod = types.ModuleType("PIL.Image")
    image_mod.Image = type("Image", (), {})
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "PIL", pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", image_mod)

    path = Path("experiments/vlm_finetuning/scripts/build_hf_graph_experts_dataset.py")
    spec = importlib.util.spec_from_file_location("build_hf_graph_experts_dataset_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_collect_asset_allow_patterns_limits_to_selected_rows_and_images(monkeypatch) -> None:
    mod = _load_builder(monkeypatch)
    rows = [
        {"images": ["exports/colab-run-001/assets/a/page_000.png", "assets/a/page_001.png", "assets/a/page_002.png"]},
        {"image": {"path": "assets/b/page_000.png"}},
    ]

    patterns = mod.collect_asset_allow_patterns(rows, "exports/colab-run-001", max_images=2)

    assert patterns == [
        "exports/colab-run-001/assets/a/page_000.png",
        "exports/colab-run-001/assets/a/page_001.png",
        "exports/colab-run-001/assets/b/page_000.png",
    ]


def test_select_debug_rows_is_deterministic(monkeypatch) -> None:
    mod = _load_builder(monkeypatch)
    rows = [{"id": i} for i in range(20)]

    assert mod.select_debug_rows(rows, 5, 42) == mod.select_debug_rows(rows, 5, 42)
    assert len(mod.select_debug_rows(rows, 5, 42)) == 5
    assert mod.select_debug_rows(rows, 0, 42) == rows
