from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import nbformat

from scireason.config import settings
from scireason.ingest import pipeline as ingest_pipeline
from scireason.mm import vlm
from scireason.task2_validation import (
    build_task2_offline_review_package,
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


def test_offline_review_package_contains_embedded_graphs_and_records(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "expert_validation" / "drafts").mkdir(parents=True)
    (bundle_dir / "reference_triplets.csv").write_text(
        "assertion_id,subject,predicate,object,start_date,end_date\n"
        "g1,A,relates_to,B,2021,2022\n",
        encoding="utf-8",
    )
    (bundle_dir / "reference_graph.json").write_text(
        json.dumps({"nodes": [{"id": "A", "label": "A"}, {"id": "B", "label": "B"}], "edges": [{"source": "A", "target": "B", "predicate": "relates_to"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle_dir / "reference_graph.html").write_text("<html><body>graph</body></html>", encoding="utf-8")

    manifest = {
        "bundle_dir": str(bundle_dir),
        "gold_graph": str(bundle_dir / "reference_graph.json"),
        "gold_graph_html": str(bundle_dir / "reference_graph.html"),
        "gold_triplets_csv": str(bundle_dir / "reference_triplets.csv"),
    }
    task1_doc = {"topic": "demo topic", "submission_id": "demo-submission", "cutoff_year": 2024}
    html_path = build_task2_offline_review_package(manifest, task1_doc)
    assert html_path.exists()
    html_text = html_path.read_text(encoding="utf-8")
    assert "Task 2 — автономная форма экспертной валидации" in html_text
    assert "demo-submission" in html_text
    assert "relates_to" in html_text
    assert "Скачать результаты ZIP" in html_text




def test_offline_review_package_embedded_script_is_valid_javascript(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "expert_validation" / "drafts").mkdir(parents=True)
    (bundle_dir / "reference_triplets.csv").write_text(
        "assertion_id,subject,predicate,object,start_date,end_date\n"
        "g1,A,relates_to,B,2021,2022\n",
        encoding="utf-8",
    )
    (bundle_dir / "reference_graph.json").write_text(
        json.dumps({"nodes": [{"id": "A", "label": "A"}, {"id": "B", "label": "B"}], "edges": [{"source": "A", "target": "B", "predicate": "relates_to"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle_dir / "reference_graph.html").write_text("<html><body>graph</body></html>", encoding="utf-8")

    manifest = {
        "bundle_dir": str(bundle_dir),
        "gold_graph": str(bundle_dir / "reference_graph.json"),
        "gold_graph_html": str(bundle_dir / "reference_graph.html"),
        "gold_triplets_csv": str(bundle_dir / "reference_triplets.csv"),
    }
    task1_doc = {"topic": "demo topic", "submission_id": "demo-submission", "cutoff_year": 2024}
    html_path = build_task2_offline_review_package(manifest, task1_doc)
    html_text = html_path.read_text(encoding="utf-8")
    match = re.search(r"<script>\s*(.*)\s*</script>", html_text, re.S)
    assert match, "embedded script not found"
    node = shutil.which("node")
    if not node:
        assert "placeholder: `paper_ids:" in html_text
        return
    script_path = tmp_path / "offline_review.js"
    script_path.write_text(match.group(1), encoding="utf-8")
    proc = subprocess.run([node, "--check", str(script_path)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_notebook_defaults_to_g4f_and_has_draft_controls() -> None:
    nb = nbformat.read("notebooks/task2_temporal_graph_validation_colab.ipynb", as_version=4)
    joined = "\n\n".join(cell.source for cell in nb.cells)

    assert "save_task2_review_state" in joined
    assert "('g4f (default)', 'g4f')" in joined
    assert "value='g4f'" in joined
    assert "Сохранить черновик" in joined
    assert "Загрузить черновик" in joined
    assert "Автосохранение черновика" in joined
    assert "Скачать автономную форму" in joined


def test_offline_review_package_supports_runtime_manifest_artifacts_layout(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "expert_validation" / "drafts").mkdir(parents=True)
    (bundle_dir / "automatic_graph").mkdir(parents=True)
    (bundle_dir / "reference_triplets.csv").write_text(
        "assertion_id,subject,predicate,object,start_date,end_date\n"
        "g1,A,relates_to,B,2021,2022\n",
        encoding="utf-8",
    )
    (bundle_dir / "automatic_triplets.csv").write_text(
        "assertion_id,subject,predicate,object,start_date,end_date\n"
        "a1,B,influences,C,2022,2023\n",
        encoding="utf-8",
    )
    (bundle_dir / "reference_graph.json").write_text(
        json.dumps({"nodes": [{"id": "A", "label": "A"}, {"id": "B", "label": "B"}], "edges": [{"source": "A", "target": "B", "predicate": "relates_to"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle_dir / "automatic_graph" / "temporal_kg.json").write_text(
        json.dumps({"nodes": [{"id": "B", "label": "B"}, {"id": "C", "label": "C"}], "edges": [{"source": "B", "target": "C", "predicate": "influences"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (bundle_dir / "comparison_summary.json").write_text(json.dumps({"automatic_edges": 1}, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "bundle_dir": str(bundle_dir),
        "artifacts": {
            "reference_graph": "reference_graph.json",
            "automatic_graph": "automatic_graph/temporal_kg.json",
            "comparison_summary": "comparison_summary.json",
        },
    }
    task1_doc = {"topic": "demo topic", "submission_id": "demo-submission", "cutoff_year": 2024}
    html_path = build_task2_offline_review_package(manifest, task1_doc)
    html_text = html_path.read_text(encoding="utf-8")
    assert '"graphs": {"gold": {"nodes": [{"id": "A"' in html_text
    assert '"auto": {"nodes": [{"id": "B"' in html_text
    assert '"records": [{"graph_kind": "gold"' in html_text
    assert 'influences' in html_text
