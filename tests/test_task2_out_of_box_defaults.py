from __future__ import annotations

import json
from pathlib import Path

import nbformat

from scireason.config import settings
from scireason.ingest import pipeline as ingest_pipeline
from scireason.mm import vlm
from scireason.task2_validation import (
    get_task2_review_state_paths,
    load_task2_review_state,
    save_task2_review_state,
)


def test_ingest_pdf_auto_skips_grobid_in_auto_mode(monkeypatch, tmp_path: Path) -> None:
    called = {"grobid": False, "fallback": False}

    def _raise_paddle(*args, **kwargs):
        raise ingest_pipeline.PaddleOCRUnavailableError("no paddle")

    def _boom_grobid(*args, **kwargs):
        called["grobid"] = True
        raise AssertionError("grobid should not be called in auto mode")

    def _fallback(*args, **kwargs):
        called["fallback"] = True
        return tmp_path / "paper"

    monkeypatch.setattr(ingest_pipeline, "ingest_pdf_paddleocr", _raise_paddle)
    monkeypatch.setattr(ingest_pipeline, "ingest_pdf", _boom_grobid)
    monkeypatch.setattr(ingest_pipeline, "_ingest_pdf_pymupdf", _fallback)
    monkeypatch.setattr(ingest_pipeline.settings, "ocr_backend", "auto")

    out = ingest_pipeline.ingest_pdf_auto(tmp_path / "demo.pdf", {"id": "demo"}, tmp_path)
    assert out == tmp_path / "paper"
    assert called["fallback"] is True
    assert called["grobid"] is False


def test_describe_image_uses_g4f_default_model_when_backend_is_g4f(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    from PIL import Image

    Image.new("RGB", (8, 8), color="white").save(image_path)

    monkeypatch.setattr(vlm, "_has_g4f", lambda: True)
    monkeypatch.setattr(vlm, "_has_local_vlm_stack", lambda: False)

    called = {}

    def _fake_g4f(*, image_path: Path, prompt: str, model_id: str):
        called["model_id"] = model_id
        return vlm.VLMResult(caption="ok")

    monkeypatch.setattr(vlm, "_describe_image_g4f", _fake_g4f)

    res = vlm.describe_image(
        image_path=image_path,
        prompt="demo",
        backend="g4f",
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
    )
    assert res.caption == "ok"
    assert called["model_id"] == settings.task2_default_g4f_model


def test_review_state_roundtrip(tmp_path: Path) -> None:
    payload = {
        "reviewer_id": "demo-reviewer",
        "review_state": {"edge:1": {"verdict": "accepted"}},
    }
    latest = save_task2_review_state(tmp_path, payload, label="checkpoint")
    assert latest.exists()

    loaded = load_task2_review_state(tmp_path)
    assert loaded["reviewer_id"] == "demo-reviewer"
    assert loaded["review_state"]["edge:1"]["verdict"] == "accepted"

    paths = get_task2_review_state_paths(tmp_path)
    assert paths["latest"] == latest
    assert paths["draft_dir"].exists()


def test_notebook_defaults_to_g4f_and_has_draft_controls() -> None:
    nb = nbformat.read("notebooks/task2_temporal_graph_validation_colab.ipynb", as_version=4)
    cell3 = nb.cells[3].source
    cell4 = nb.cells[4].source

    assert "save_task2_review_state" in nb.cells[2].source
    assert "('g4f (default)', 'g4f')" in cell4
    assert "value='g4f'" in cell4
    assert "Сохранить черновик" in cell3
    assert "Загрузить черновик" in cell3
    assert "Автосохранение черновика" in cell3
