from __future__ import annotations

import json
from pathlib import Path

from scireason.config import settings
from scireason.llm import temporary_llm_selection
from scireason.mm.vlm import temporary_vlm_selection
from scireason.pipeline import task2_validation as pipeline


class _FakeKG:
    def dump_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"nodes": [], "edges": []}), encoding="utf-8")


def test_temporary_llm_selection_restores_settings() -> None:
    prev_provider = settings.llm_provider
    prev_model = settings.llm_model

    with temporary_llm_selection(g4f_model="deepseek-r1"):
        assert settings.llm_provider == "g4f"
        assert settings.llm_model == "deepseek-r1"

    assert settings.llm_provider == prev_provider
    assert settings.llm_model == prev_model


def test_prepare_task2_bundle_applies_local_model_override(tmp_path: Path, monkeypatch) -> None:
    trajectory = tmp_path / "trajectory.yaml"
    trajectory.write_text(
        """
submission_id: demo-task2
topic: Demo topic
papers: []
trajectory:
  - step_id: s1
    title: First step
    summary: Demo
""".strip(),
        encoding="utf-8",
    )

    captured: dict[str, str] = {}

    def _fake_build_temporal_kg(*args, **kwargs):
        captured["provider"] = settings.llm_provider
        captured["model"] = settings.llm_model
        return _FakeKG()

    monkeypatch.setattr(pipeline, "build_temporal_kg", _fake_build_temporal_kg)
    monkeypatch.setattr(pipeline, "resolve_papers_from_trajectory", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "acquire_pdfs", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "load_papers_from_processed", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "suggest_link_candidates", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "_flatten_automatic_graph", lambda kg: [])

    prev_provider = settings.llm_provider
    prev_model = settings.llm_model

    out_dir = pipeline.prepare_task2_validation_bundle(
        trajectory,
        out_dir=tmp_path / "out",
        include_multimodal=False,
        suggest_links=False,
        local_model="llama3.2",
    )

    assert captured == {"provider": "ollama", "model": "llama3.2"}
    assert settings.llm_provider == prev_provider
    assert settings.llm_model == prev_model
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["submission_id"] == "demo-task2"


def test_temporary_vlm_selection_restores_settings() -> None:
    prev_backend = settings.vlm_backend
    prev_model = settings.vlm_model_id

    with temporary_vlm_selection(vlm_backend="g4f", vlm_model_id="r1-1776"):
        assert settings.vlm_backend == "g4f"
        assert settings.vlm_model_id == "r1-1776"

    assert settings.vlm_backend == prev_backend
    assert settings.vlm_model_id == prev_model


def test_prepare_task2_bundle_records_vlm_override(tmp_path: Path, monkeypatch) -> None:
    trajectory = tmp_path / "trajectory.yaml"
    trajectory.write_text(
        """
submission_id: demo-task2-vlm
topic: Demo topic
papers: []
trajectory:
  - step_id: s1
    title: First step
    summary: Demo
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(pipeline, "build_temporal_kg", lambda *a, **k: _FakeKG())
    monkeypatch.setattr(pipeline, "resolve_papers_from_trajectory", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "acquire_pdfs", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "load_papers_from_processed", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "suggest_link_candidates", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "_flatten_automatic_graph", lambda kg: [])

    out_dir = pipeline.prepare_task2_validation_bundle(
        trajectory,
        out_dir=tmp_path / "out",
        include_multimodal=False,
        suggest_links=False,
        vlm_backend="qwen3_vl",
        vlm_model_id="Qwen/Qwen3-VL-8B-Instruct",
    )

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["submission_id"] == "demo-task2-vlm"
    assert manifest["vlm_effective_backend"] == "qwen3_vl"
    assert manifest["vlm_effective_model"] == "Qwen/Qwen3-VL-8B-Instruct"
