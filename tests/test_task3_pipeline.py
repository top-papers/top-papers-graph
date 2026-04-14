from __future__ import annotations

import json
import os
from pathlib import Path

from scireason.cli import app
from scireason.config import settings
from scireason.pipeline.task3_hypothesis_generation import prepare_task3_hypothesis_bundle


def _write_processed_paper(
    root: Path,
    *,
    paper_id: str,
    year: int,
    title: str,
    text: str,
    mm_text: str,
    vlm_caption: str = "",
    tables_md: str = "Year | effect\n2024 | latency reduction",
    equations_md: str = "",
    create_image: bool = False,
) -> None:
    paper_dir = root / paper_id
    (paper_dir / "mm").mkdir(parents=True)
    (paper_dir / "meta.json").write_text(
        json.dumps({"id": paper_id, "title": title, "year": year}, ensure_ascii=False),
        encoding="utf-8",
    )
    (paper_dir / "chunks.jsonl").write_text(
        json.dumps(
            {
                "chunk_id": f"{paper_id}:c1",
                "paper_id": paper_id,
                "text": text,
                "modality": "text",
                "source_backend": "unit_test",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    image_path = ""
    if create_image:
        image_file = paper_dir / "mm" / "images" / "page_001.png"
        image_file.parent.mkdir(parents=True, exist_ok=True)
        image_file.write_bytes(b"fake-png-bytes")
        image_path = str(image_file.resolve().as_posix())

    (paper_dir / "mm" / "pages.jsonl").write_text(
        json.dumps(
            {
                "paper_id": paper_id,
                "page": 1,
                "text": mm_text,
                "image_path": image_path,
                "vlm_caption": vlm_caption,
                "tables_md": tables_md,
                "equations_md": equations_md,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def test_task3_cli_commands_are_registered() -> None:
    names = {cmd.name for cmd in app.registered_commands}
    assert "prepare-task3-hypotheses" in names
    assert "task3-bundle" in names


def test_prepare_task3_bundle_from_processed_dir_offline(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _write_processed_paper(
        processed,
        paper_id="paper-1",
        year=2023,
        title="Temporal catalyst study",
        text=(
            "Catalyst A reduces latency in sensor networks during 2023 trials. "
            "Forecast accuracy improves after catalyst A deployment."
        ),
        mm_text="Page evidence: catalyst A, latency reduction, 2023 timeline.",
    )
    _write_processed_paper(
        processed,
        paper_id="paper-2",
        year=2024,
        title="Temporal catalyst follow-up",
        text=(
            "Catalyst A reduces latency and improves forecast stability in 2024 monitoring. "
            "Temporal graph analysis links catalyst A to latency and forecasting outcomes."
        ),
        mm_text="Figure summary: catalyst A maintains lower latency in 2024.",
    )

    prev = {
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "vlm_backend": settings.vlm_backend,
        "embed_provider": settings.embed_provider,
        "mm_embed_backend": settings.mm_embed_backend,
        "hyp_tgnn_enabled": settings.hyp_tgnn_enabled,
    }
    settings.llm_provider = "mock"
    settings.llm_model = "mock"
    settings.vlm_backend = "none"
    settings.embed_provider = "hash"
    settings.mm_embed_backend = "none"
    settings.hyp_tgnn_enabled = True

    try:
        result = prepare_task3_hypothesis_bundle(
            processed_dir=processed,
            query="temporal catalyst latency forecasting",
            out_dir=tmp_path / "runs",
            domain_id="science",
            top_papers=0,
            top_hypotheses=3,
            candidate_top_k=5,
            include_multimodal=True,
            run_vlm=False,
            edge_mode="cooccurrence",
            link_prediction_backend="heuristic",
            annoy_n_trees=8,
        )
    finally:
        settings.llm_provider = prev["llm_provider"]
        settings.llm_model = prev["llm_model"]
        settings.vlm_backend = prev["vlm_backend"]
        settings.embed_provider = prev["embed_provider"]
        settings.mm_embed_backend = prev["mm_embed_backend"]
        settings.hyp_tgnn_enabled = prev["hyp_tgnn_enabled"]

    assert result.bundle_dir.exists()
    assert result.manifest_path.exists()
    assert result.hypotheses_path.exists()

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    ranked = json.loads(result.hypotheses_path.read_text(encoding="utf-8"))
    link_predictions = json.loads(
        Path(manifest["artifacts"]["link_predictions"]).read_text(encoding="utf-8")
    )

    assert manifest["n_paper_records"] == 2
    assert manifest["n_chunks"] >= 4
    assert manifest["runtime"]["annoy_backend"] in {"annoy", "numpy_fallback"}
    assert manifest["runtime"]["link_prediction_used_backend"] == "heuristic"
    assert Path(manifest["artifacts"]["annoy_manifest"]).exists()
    assert Path(manifest["artifacts"]["multimodal_triplets"]).exists()
    assert ranked
    assert ranked[0]["hypothesis"]["title"]
    assert ranked[0]["hypothesis"]["premise"]
    assert ranked[0]["temporal_context"]["ordering"] in {
        "predicted_or_missing",
        "stable",
        "strengthening",
        "weakening",
        "persistent",
        "predicted_missing_link",
    }
    assert isinstance(link_predictions.get("predictions"), list)


def test_prepare_task3_bundle_can_hardlink_processed_dir_and_scrub_mm_metadata(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _write_processed_paper(
        processed,
        paper_id="paper-1",
        year=2023,
        title="Temporal catalyst study",
        text="Catalyst A reduces latency in 2023 trials.",
        mm_text="Page evidence: catalyst A, latency reduction, 2023 timeline.",
        vlm_caption="alpha-only caption",
        tables_md="alpha tables",
        equations_md="alpha equations",
        create_image=True,
    )

    prev = {
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "vlm_backend": settings.vlm_backend,
        "embed_provider": settings.embed_provider,
        "mm_embed_backend": settings.mm_embed_backend,
        "hyp_tgnn_enabled": settings.hyp_tgnn_enabled,
    }
    settings.llm_provider = "mock"
    settings.llm_model = "mock"
    settings.vlm_backend = "none"
    settings.embed_provider = "hash"
    settings.mm_embed_backend = "none"
    settings.hyp_tgnn_enabled = True

    try:
        result = prepare_task3_hypothesis_bundle(
            processed_dir=processed,
            query="temporal catalyst latency",
            out_dir=tmp_path / "runs",
            domain_id="science",
            top_papers=0,
            top_hypotheses=2,
            candidate_top_k=3,
            include_multimodal=True,
            run_vlm=False,
            processed_dir_link_mode="hardlink",
            processed_dir_strip_mm_vlm_metadata=True,
            edge_mode="cooccurrence",
            link_prediction_backend="heuristic",
            annoy_n_trees=4,
        )
    finally:
        settings.llm_provider = prev["llm_provider"]
        settings.llm_model = prev["llm_model"]
        settings.vlm_backend = prev["vlm_backend"]
        settings.embed_provider = prev["embed_provider"]
        settings.mm_embed_backend = prev["mm_embed_backend"]
        settings.hyp_tgnn_enabled = prev["hyp_tgnn_enabled"]

    copied_pages = result.bundle_dir / "processed_papers" / "paper-1" / "mm" / "pages.jsonl"
    copied_image = result.bundle_dir / "processed_papers" / "paper-1" / "mm" / "images" / "page_001.png"
    source_pages = processed / "paper-1" / "mm" / "pages.jsonl"
    source_image = processed / "paper-1" / "mm" / "images" / "page_001.png"

    copied_payload = json.loads(copied_pages.read_text(encoding="utf-8").splitlines()[0])
    source_payload = json.loads(source_pages.read_text(encoding="utf-8").splitlines()[0])

    assert copied_payload["vlm_caption"] == ""
    assert copied_payload["tables_md"] == ""
    assert copied_payload["equations_md"] == ""
    assert copied_payload["image_path"] == str(copied_image.resolve().as_posix())
    assert source_payload["vlm_caption"] == "alpha-only caption"
    assert source_payload["tables_md"] == "alpha tables"
    assert source_payload["equations_md"] == "alpha equations"
    assert os.stat(source_image).st_ino == os.stat(copied_image).st_ino
