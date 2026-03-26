from __future__ import annotations

import json
from pathlib import Path

from scireason.mm import pdf_mm_extract
from scireason.temporal.temporal_kg_builder import load_papers_from_processed


class _DummyPixmap:
    def save(self, path: str) -> None:
        Path(path).write_bytes(b"png")


class _DummyPage:
    def __init__(self, number: int) -> None:
        self.number = number

    def get_text(self, kind: str) -> str:
        return f"page {self.number} text"

    def get_pixmap(self, matrix=None, alpha=False):
        return _DummyPixmap()


class _DummyDoc:
    def __init__(self, count: int = 3) -> None:
        self.count = count
        self.closed = False

    def __len__(self) -> int:
        return self.count

    def load_page(self, index: int) -> _DummyPage:
        return _DummyPage(index)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.closed = True


class _DummyFitz:
    class Matrix:
        def __init__(self, x: float, y: float) -> None:
            self.x = x
            self.y = y

    def __init__(self) -> None:
        self.last_doc: _DummyDoc | None = None

    def open(self, path: str) -> _DummyDoc:
        self.last_doc = _DummyDoc()
        return self.last_doc


def test_extract_pages_can_stream_without_collecting_records(tmp_path: Path, monkeypatch) -> None:
    dummy_fitz = _DummyFitz()
    monkeypatch.setitem(__import__("sys").modules, "fitz", dummy_fitz)
    monkeypatch.setattr(pdf_mm_extract, "describe_image", lambda *args, **kwargs: pdf_mm_extract.VLMResult(caption="caption"))

    out = pdf_mm_extract.extract_pages(
        pdf_path=tmp_path / "demo.pdf",
        paper_id="paper-1",
        out_dir=tmp_path / "paper",
        run_vlm=False,
        collect_records=False,
    )

    assert out == []
    pages_path = tmp_path / "paper" / "mm" / "pages.jsonl"
    lines = pages_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["paper_id"] == "paper-1"
    assert dummy_fitz.last_doc is not None and dummy_fitz.last_doc.closed is True


def test_load_papers_from_processed_streams_jsonl_lines(tmp_path: Path) -> None:
    paper_dir = tmp_path / "paper-1"
    paper_dir.mkdir(parents=True)
    (paper_dir / "meta.json").write_text(json.dumps({"id": "paper-1", "title": "Demo", "year": 2024}), encoding="utf-8")
    with (paper_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for idx in range(5):
            fh.write(json.dumps({"text": f"chunk-{idx}-" + ("x" * 30)}) + "\n")

    papers = load_papers_from_processed(tmp_path, max_chars_per_paper=50)

    assert len(papers) == 1
    assert papers[0].paper_id == "paper-1"
    assert len(papers[0].text) <= 52
    assert "chunk-0" in papers[0].text
