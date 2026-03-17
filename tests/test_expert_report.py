from __future__ import annotations

import json

from scireason.report.expert_report import generate_expert_report
from scireason.schemas import Citation, HypothesisDraft
from scireason.temporal.temporal_kg_builder import EdgeStats, NodeStats, PaperRecord, TemporalKnowledgeGraph


def _make_processed_dir(tmp_path):
    processed = tmp_path / "processed"
    paper_dir = processed / "paper1"
    paper_dir.mkdir(parents=True)
    (paper_dir / "meta.json").write_text(
        json.dumps({"id": "doi:paper1", "title": "Demo paper", "year": 2024}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rows = [
        {
            "paper_id": "doi:paper1",
            "chunk_id": "doi:paper1:text:1",
            "modality": "text",
            "text": "Catalyst A improves yield B in 2024 and is discussed with process C.",
            "page": 3,
            "order": 1,
            "summary": "Catalyst A improves yield B.",
            "backend": "docling",
            "metadata": {},
        },
        {
            "paper_id": "doi:paper1",
            "chunk_id": "doi:paper1:figure:1",
            "modality": "figure",
            "text": "Figure 1 shows catalyst A and yield B trend.",
            "page": 4,
            "order": 2,
            "figure_or_table": "Figure 1",
            "image_path": "fig1.png",
            "summary": "Figure 1 trend.",
            "backend": "docling",
            "metadata": {},
        },
    ]
    with (paper_dir / "structured_chunks.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return processed


def test_generate_expert_report_writes_json_and_markdown(tmp_path) -> None:
    processed = _make_processed_dir(tmp_path)
    kg = TemporalKnowledgeGraph(
        nodes={
            "catalyst a": NodeStats(term="catalyst a"),
            "yield b": NodeStats(term="yield b"),
            "process c": NodeStats(term="process c"),
        },
        edges=[
            EdgeStats(
                source="Catalyst A",
                predicate="improves",
                target="Yield B",
                total_count=2,
                yearly_count={2024: 2},
                papers={"doi:paper1"},
                evidence_quotes=[{"paper_id": "doi:paper1", "quote": "Catalyst A improves yield B in 2024."}],
                score=1.6,
                features={"trend": 0.8},
            ),
            EdgeStats(
                source="Catalyst A",
                predicate="relates_to",
                target="Process C",
                total_count=1,
                yearly_count={2024: 1},
                papers={"doi:paper1"},
                evidence_quotes=[{"paper_id": "doi:paper1", "quote": "Catalyst A is discussed with process C."}],
                score=1.2,
                features={"trend": 0.4},
            ),
        ],
        meta={"years": [2024]},
    )
    papers = [PaperRecord(paper_id="doi:paper1", title="Demo paper", year=2024, text="Catalyst A improves yield B and relates to process C.")]
    hyps = [
        HypothesisDraft(
            title="Catalyst A improves Yield B",
            premise="Observed repeatedly in the corpus.",
            mechanism="Potential catalytic pathway.",
            time_scope="2024",
            proposed_experiment="Run controlled ablation.",
            supporting_evidence=[Citation(source_id="doi:paper1", text_snippet="Catalyst A improves yield B in 2024.")],
            confidence_score=7,
        )
    ]
    out_dir = tmp_path / "expert_report"
    payload = generate_expert_report(
        query="catalyst yield",
        domain_title="Science",
        domain_id="science",
        kg=kg,
        papers=papers,
        processed_dir=processed,
        out_dir=out_dir,
        collection_text=None,
        collection_mm=None,
        hypotheses=hyps,
        retrieval_k=5,
        indexing_status=[{"paper_id": "doi:paper1", "status": "ok"}],
    )
    assert (out_dir / "expert_report.json").exists()
    assert (out_dir / "expert_report.md").exists()
    assert payload["retrieval"]["text_backend"] == "local"
    assert payload["retrieval"]["mm_backend"] == "local_mm"
    assert payload["graph_analysis"]["summary"]["nodes"] >= 2
    assert payload["tgnn_analysis"]["predictions"] is not None
