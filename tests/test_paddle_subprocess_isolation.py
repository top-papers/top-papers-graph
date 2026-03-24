from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scireason.ingest.paddleocr_pipeline import extract_pdf_chunks_paddleocr


def test_extract_pdf_chunks_paddleocr_uses_worker_output(monkeypatch, tmp_path: Path) -> None:
    out_records = [{
        "chunk_id": "paper:page:0:markdown",
        "paper_id": "paper",
        "page": 0,
        "text": "hello",
        "modality": "page",
        "source_backend": "PPStructureV3",
        "metadata": {},
    }]

    def fake_run(cmd, capture_output, text, env, timeout=None):
        out_index = cmd.index("--out") + 1
        Path(cmd[out_index]).write_text(json.dumps(out_records), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("scireason.ingest.paddleocr_pipeline.subprocess.run", fake_run)

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%test")

    records = extract_pdf_chunks_paddleocr(pdf_path=pdf_path, paper_id="paper")
    assert len(records) == 1
    assert records[0].text == "hello"
    assert records[0].source_backend == "PPStructureV3"



def test_extract_pdf_chunks_paddleocr_times_out_fast(monkeypatch, tmp_path: Path) -> None:
    from subprocess import TimeoutExpired

    def fake_run(cmd, capture_output, text, env, timeout):
        raise TimeoutExpired(cmd=cmd, timeout=timeout)

    monkeypatch.setattr("scireason.ingest.paddleocr_pipeline.subprocess.run", fake_run)

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%test")

    events = []
    monkeypatch.setattr("scireason.ingest.paddleocr_pipeline.settings", __import__("scireason.ingest.paddleocr_pipeline", fromlist=['settings']).settings)
    from scireason.ingest.paddleocr_pipeline import settings as paddle_settings
    monkeypatch.setattr(paddle_settings, "paddleocr_worker_timeout_seconds", 1)

    import pytest
    from scireason.ingest.paddleocr_pipeline import extract_pdf_chunks_paddleocr, PaddleOCRUnavailableError

    with pytest.raises(PaddleOCRUnavailableError, match="timed out"):
        extract_pdf_chunks_paddleocr(pdf_path=pdf_path, paper_id="paper", progress_callback=events.append)

    assert [e["event"] for e in events] == ["ocr_start", "ocr_timeout"]
