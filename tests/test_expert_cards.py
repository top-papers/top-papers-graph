from __future__ import annotations

import json

from scireason.report.expert_cards import generate_chunk_cards, generate_task2_review_cards
from scireason.temporal.temporal_kg_builder import EdgeStats, NodeStats, TemporalKnowledgeGraph


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
            "text": "Catalyst A improves yield B at 25 C and 80% conversion in 2024.",
            "page": 3,
            "order": 1,
            "summary": "Catalyst A improves yield B.",
            "backend": "docling",
            "metadata": {},
        },
        {
            "paper_id": "doi:paper1",
            "chunk_id": "doi:paper1:table:1",
            "modality": "table",
            "text": "Table evidence for Catalyst A and yield B.",
            "page": 4,
            "order": 2,
            "figure_or_table": "Table 1",
            "table_markdown": "| Catalyst | Yield |\n| A | 80% |",
            "summary": "Table 1 with Catalyst A and Yield B.",
            "backend": "docling",
            "metadata": {},
        },
    ]
    with (paper_dir / "structured_chunks.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return processed


def test_generate_chunk_cards_extracts_modalities_and_conditions(tmp_path) -> None:
    processed = _make_processed_dir(tmp_path)
    out_path = tmp_path / "chunk_cards.jsonl"
    cards = generate_chunk_cards(processed, out_path=out_path)
    assert out_path.exists()
    assert len(cards) == 2
    assert {c["modality"] for c in cards} == {"text", "table"}
    text_card = next(c for c in cards if c["modality"] == "text")
    assert any("25 C" in hint or "80%" in hint for hint in text_card["condition_hints"])


def test_generate_task2_review_cards_creates_template_payloads(tmp_path) -> None:
    processed = _make_processed_dir(tmp_path)
    kg = TemporalKnowledgeGraph(
        nodes={
            "catalyst a": NodeStats(term="catalyst a"),
            "yield b": NodeStats(term="yield b"),
        },
        edges=[
            EdgeStats(
                source="Catalyst A",
                predicate="improves",
                target="Yield B",
                total_count=2,
                yearly_count={2024: 2},
                papers={"doi:paper1"},
                evidence_quotes=[{"paper_id": "doi:paper1", "quote": "Catalyst A improves yield B at 25 C."}],
                score=1.5,
            )
        ],
        meta={"years": [2024]},
    )
    out_dir = tmp_path / "graph_reviews_auto"
    manifest = generate_task2_review_cards(kg, processed_dir=processed, domain_id="science", out_dir=out_dir)
    assert manifest
    assert (out_dir / "index.json").exists()
    payload = json.loads((out_dir / manifest[0]["path"].split("/")[-1]).read_text(encoding="utf-8"))
    assert payload["domain"] == "science"
    assert payload["paper_id"] == "doi:paper1"
    assert payload["assertions"]
    first = payload["assertions"][0]
    assert first["evidence"]["page"] in {3, 4}
    assert first["verdict"] in {"accepted", "needs_time_fix", "needs_evidence_fix"}
