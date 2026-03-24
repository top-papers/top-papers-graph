from __future__ import annotations

import time
from pathlib import Path

import nbformat
from PIL import Image

from scireason.config import settings
from scireason.mm import vlm
from scireason.pipeline import task2_validation as pipe


def test_describe_image_disables_g4f_after_timeout(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (16, 16), color="white").save(image_path)

    monkeypatch.setattr(vlm, "_has_g4f", lambda: True)
    monkeypatch.setattr(vlm, "_has_local_vlm_stack", lambda: False)
    monkeypatch.setattr(vlm, "_G4F_AUTH_DISABLED", False)
    monkeypatch.setattr(vlm, "_G4F_TIMEOUT_DISABLED", False)
    monkeypatch.setattr(settings, "vlm_request_timeout_seconds", 0.05)

    def _slow_g4f(**kwargs):
        time.sleep(0.2)
        return vlm.VLMResult(caption="late")

    monkeypatch.setattr(vlm, "_describe_image_g4f", _slow_g4f)

    res = vlm.describe_image(image_path=image_path, prompt="demo", backend="g4f", model_id="auto")
    assert res.caption == ""
    assert vlm._G4F_TIMEOUT_DISABLED is True


def test_prepare_task2_bundle_emits_progress(monkeypatch, tmp_path: Path) -> None:
    yaml_path = tmp_path / "task1.yaml"
    yaml_path.write_text(
        """
topic: demo topic
submission_id: demo_progress
pipeline: full
papers: []
steps: []
""".strip(),
        encoding="utf-8",
    )

    class _DummyKG:
        def dump_json(self, path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"nodes": [], "edges": []}', encoding="utf-8")

    monkeypatch.setattr(pipe, "build_temporal_kg", lambda *args, **kwargs: _DummyKG())
    monkeypatch.setattr(pipe, "_flatten_automatic_graph", lambda *args, **kwargs: [])
    monkeypatch.setattr(pipe, "_compare_graphs", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(pipe, "_prefill_graph_review", lambda *args, **kwargs: {"items": []})
    monkeypatch.setattr(pipe, "_empty_temporal_corrections", lambda *args, **kwargs: {"items": []})
    monkeypatch.setattr(pipe, "load_papers_from_processed", lambda *args, **kwargs: [])
    monkeypatch.setattr(pipe, "suggest_link_candidates", lambda *args, **kwargs: [])

    events = []
    out_dir = pipe.prepare_task2_validation_bundle(yaml_path, out_dir=tmp_path / "runs", progress_callback=events.append)

    stages = [event["stage"] for event in events]
    assert stages[0] == "load"
    assert "reference" in stages
    assert "resolve" in stages
    assert "acquire" in stages
    assert "kg" in stages
    assert stages[-1] == "finalize"
    assert events[-1]["percent"] == 100
    assert (out_dir / "manifest.json").exists()


def test_notebook_exposes_progress_widgets_and_callback() -> None:
    nb = nbformat.read("notebooks/task2_temporal_graph_validation_colab.ipynb", as_version=4)
    cell4 = nb.cells[4].source

    assert "progress_bar = W.IntProgress" in cell4
    assert "def _task2_progress_callback(payload):" in cell4
    assert "'progress_callback': _task2_progress_callback" in cell4



def test_ingest_pdf_auto_falls_back_after_paddle_timeout(monkeypatch, tmp_path: Path) -> None:
    from scireason.ingest import pipeline as ingest_pipe
    from scireason.ingest.paddleocr_pipeline import PaddleOCRUnavailableError

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%test")
    meta = {"id": "paper-1", "title": "Demo"}

    monkeypatch.setattr(ingest_pipe, "ingest_pdf_paddleocr", lambda *args, **kwargs: (_ for _ in ()).throw(PaddleOCRUnavailableError("worker timed out")))

    fallback_dir = tmp_path / "processed" / "paper-1"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(ingest_pipe, "_ingest_pdf_pymupdf", lambda *args, **kwargs: fallback_dir)

    events = []
    out = ingest_pipe.ingest_pdf_auto(pdf_path, meta, tmp_path / "processed", progress_callback=events.append)

    assert out == fallback_dir
    assert any("fallback to pymupdf" in (e.get("message") or "").lower() for e in events)
