from __future__ import annotations

import json
from pathlib import Path

import yaml

from scireason.scidatapipe_bridge import export_dataset
from scireason.scidatapipe_bridge.download import DownloadSummary, DownloadedPaper


def _write_processed_paper(root: Path, *, paper_id: str, title: str = "Paper", year: int = 2024) -> Path:
    paper_dir = root / "paper-a"
    (paper_dir / "mm" / "images").mkdir(parents=True, exist_ok=True)
    (paper_dir / "meta.json").write_text(
        json.dumps({"id": paper_id, "title": title, "year": year}, ensure_ascii=False),
        encoding="utf-8",
    )
    image_path = paper_dir / "mm" / "images" / "page_001.png"
    image_path.write_bytes(b"fake-png-bytes")
    pages = [
        {
            "paper_id": paper_id,
            "page": 1,
            "text": "Figure shows catalyst dynamics.",
            "image_path": str(image_path.resolve().as_posix()),
            "vlm_caption": "Catalyst diagram",
            "tables_md": "year | effect\n2024 | gain",
            "equations_md": "",
        }
    ]
    with (paper_dir / "mm" / "pages.jsonl").open("w", encoding="utf-8") as fh:
        for row in pages:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return paper_dir


def _write_task1(path: Path, *, paper_id: str, submission_id: str) -> None:
    task1_payload = {
        "artifact_version": 2,
        "topic": "Test topic",
        "domain": "Science",
        "cutoff_year": 2024,
        "papers": [{"id": paper_id, "year": 2024, "title": "Test paper"}],
        "steps": [
            {
                "step_id": 1,
                "claim": "A claim",
                "sources": [
                    {
                        "type": "image",
                        "source": paper_id,
                        "page": 1,
                        "locator": "Figure 1",
                        "snippet_or_summary": "Image evidence",
                    }
                ],
                "conditions": {"system": "unknown", "environment": "unknown", "protocol": "unknown", "notes": ""},
                "inference": "Inference text",
                "next_question": "Next?",
            }
        ],
        "edges": [],
        "expert": {"latin_slug": "expert"},
        "submission_id": submission_id,
    }
    path.write_text(yaml.safe_dump(task1_payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _write_task2_bundle(bundle_dir: Path, *, paper_id: str, submission_id: str) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    edge_reviews = {
        "artifact_version": 1,
        "domain": "Science",
        "topic": "Task2 topic",
        "trajectory_submission_id": submission_id,
        "cutoff_year": 2024,
        "reviewer_id": "reviewer",
        "timestamp": "2026-01-01T00:00:00Z",
        "filter_settings": {},
        "assertions": [
            {
                "graph_kind": "auto",
                "assertion_id": "a1",
                "subject": "S",
                "predicate": "P",
                "object": "O",
                "start_date": "2024",
                "end_date": "2024",
                "time_interval": "evidence:2024..2024|valid:2024..+inf",
                "evidence_payload_full": str(
                    {
                        "page": 1,
                        "figure_or_table": "Figure 1",
                        "snippet_or_summary": "Task2 snippet",
                        "paper_id": paper_id,
                        "image_path": "",
                    }
                ),
                "paper_ids": [paper_id],
                "importance_score": "0.8",
                "verdict": "accepted",
                "rationale": "Looks good",
            }
        ],
        "added_edges": [],
    }
    (bundle_dir / "edge_reviews.json").write_text(json.dumps(edge_reviews, ensure_ascii=False, indent=2), encoding="utf-8")
    (bundle_dir / "temporal_corrections.json").write_text(json.dumps({"corrections": []}, ensure_ascii=False), encoding="utf-8")
    (bundle_dir / "review_state_latest.json").write_text(json.dumps({"reviews": []}, ensure_ascii=False), encoding="utf-8")


def test_export_dataset_includes_multimodal_context_and_images(tmp_path: Path) -> None:
    task1 = tmp_path / "expert.yaml"
    _write_task1(task1, paper_id="doi:10.1000/test", submission_id="expert__1")

    processed_root = tmp_path / "processed_papers"
    _write_processed_paper(processed_root, paper_id="doi:10.1000/test")

    result = export_dataset(task1_files=[task1], out_dir=tmp_path / "out", processed_papers_dirs=[processed_root])

    sft_lines = result.sft_path.read_text(encoding="utf-8").splitlines()
    assert len(sft_lines) == 1
    row = json.loads(sft_lines[0])
    user_content = row["chat"]["messages"][1]["content"]
    texts = [item["text"] for item in user_content if item["type"] == "text"]
    images = [item["image"] for item in user_content if item["type"] == "image"]
    assert any("Multimodal evidence extracted from cited articles" in text for text in texts)
    assert any("Catalyst diagram" in text for text in texts)
    assert images, "expected copied image attachments"
    assert Path(images[0]).exists()


def test_export_dataset_normalizes_task2_bundle_and_builds_grpo(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    _write_task2_bundle(bundle_dir, paper_id="doi:10.1000/test2", submission_id="expert__2")

    processed_root = tmp_path / "processed_papers"
    _write_processed_paper(processed_root, paper_id="doi:10.1000/test2")

    result = export_dataset(task2_inputs=[bundle_dir], out_dir=tmp_path / "out", processed_papers_dirs=[processed_root])

    assert result.grpo_path.exists()
    grpo_lines = result.grpo_path.read_text(encoding="utf-8").splitlines()
    assert len(grpo_lines) == 1
    row = json.loads(grpo_lines[0])
    user_content = row["prompt_chat"]["messages"][1]["content"]
    assert any(item["type"] == "image" for item in user_content)
    assert any(item["type"] == "text" and "Multimodal evidence extracted from cited articles" in item["text"] for item in user_content)


def test_export_dataset_discovers_inputs_from_directories(tmp_path: Path) -> None:
    task1_dir = tmp_path / "incoming_task1"
    task2_dir = tmp_path / "incoming_task2"
    task1_dir.mkdir()
    task2_dir.mkdir()
    _write_task1(task1_dir / "expert_dir.yaml", paper_id="doi:10.1000/dir", submission_id="expert__dir")
    _write_task2_bundle(task2_dir / "bundle_dir", paper_id="doi:10.1000/dir", submission_id="expert__dir")

    processed_root = tmp_path / "processed_papers"
    _write_processed_paper(processed_root, paper_id="doi:10.1000/dir")

    result = export_dataset(task1_dirs=[task1_dir], task2_dirs=[task2_dir], out_dir=tmp_path / "out", processed_papers_dirs=[processed_root])

    assert result.stats["discovered_task1_files"] == 1
    assert result.stats["discovered_task2_inputs"] == 1
    assert result.sft_path.exists()
    assert result.grpo_path.exists()


def test_export_dataset_discovers_inputs_from_mixed_input_dir(tmp_path: Path) -> None:
    mixed = tmp_path / "incoming_mixed"
    mixed.mkdir()
    _write_task1(mixed / "expert_mixed.yaml", paper_id="doi:10.1000/mixed", submission_id="expert__mixed")
    _write_task2_bundle(mixed / "bundle_mixed", paper_id="doi:10.1000/mixed", submission_id="expert__mixed")

    processed_root = tmp_path / "processed_papers"
    _write_processed_paper(processed_root, paper_id="doi:10.1000/mixed")

    result = export_dataset(input_dirs=[mixed], out_dir=tmp_path / "out", processed_papers_dirs=[processed_root])

    assert result.stats["discovered_task1_files"] == 1
    assert result.stats["discovered_task2_inputs"] == 1
    assert result.stats["input_dirs"] == [str(mixed)]


def test_export_dataset_can_download_and_ingest_from_references(tmp_path: Path, monkeypatch) -> None:
    task1 = tmp_path / "expert_download.yaml"
    _write_task1(task1, paper_id="doi:10.1000/download", submission_id="expert__download")

    def fake_download(refs, *, download_root, processed_papers_dir, existing_processed_papers_dirs, **kwargs):
        assert any(ref.id == "doi:10.1000/download" for ref in refs)
        assert processed_papers_dir is not None
        _write_processed_paper(processed_papers_dir, paper_id="doi:10.1000/download", title="Downloaded")
        return DownloadSummary(
            refs_total=len(refs),
            refs_supported=len(refs),
            pdf_downloaded=1,
            html_downloaded=0,
            ingested_processed_papers=1,
            skipped_existing=0,
            errors=0,
            produced_processed_papers_dir=processed_papers_dir,
            download_root=download_root,
            records=[DownloadedPaper(ref_id="doi:10.1000/download", paper_type="doi", slug="doi_10.1000_download", source="fake", ingested_paper_dir=processed_papers_dir / "paper-a")],
        )

    monkeypatch.setattr("scireason.scidatapipe_bridge.builder.download_and_ingest_refs", fake_download)

    result = export_dataset(
        task1_files=[task1],
        out_dir=tmp_path / "out",
        download_referenced_papers=True,
        ingest_downloaded_papers=True,
    )

    assert result.stats["download_pdf_downloaded"] == 1
    row = json.loads(result.sft_path.read_text(encoding="utf-8").splitlines()[0])
    user_content = row["chat"]["messages"][1]["content"]
    assert any(item["type"] == "image" for item in user_content)


def test_export_dataset_can_upload_to_hf(tmp_path: Path, monkeypatch) -> None:
    task1 = tmp_path / "expert_upload.yaml"
    _write_task1(task1, paper_id="doi:10.1000/upload", submission_id="expert__upload")
    processed_root = tmp_path / "processed_papers"
    _write_processed_paper(processed_root, paper_id="doi:10.1000/upload")

    calls = {}

    class DummyResult:
        repo_id = "org/dataset"
        repo_url = "https://huggingface.co/datasets/org/dataset"
        path_in_repo = "exports/run1"
        commit_message = "upload"
        created_repo = True
        private = True
        upload_result = object()

    def fake_upload(export_root, **kwargs):
        calls["export_root"] = export_root
        calls.update(kwargs)
        readme = export_root / "README.md"
        if not readme.exists():
            readme.write_text("# card", encoding="utf-8")
        return DummyResult()

    monkeypatch.setattr("scireason.scidatapipe_bridge.builder.upload_export_to_hf", fake_upload)

    result = export_dataset(
        task1_files=[task1],
        out_dir=tmp_path / "out",
        processed_papers_dirs=[processed_root],
        hf_upload=True,
        hf_repo_id="org/dataset",
        hf_private=True,
        hf_path_in_repo="exports/run1",
        hf_commit_message="upload",
    )

    assert calls["repo_id"] == "org/dataset"
    assert calls["private"] is True
    assert calls["path_in_repo"] == "exports/run1"
    assert result.hf_repo_url == "https://huggingface.co/datasets/org/dataset"
    assert result.stats["hf_uploaded"] is True
    assert result.stats["hf_repo_id"] == "org/dataset"
