from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from rich.console import Console

from ..domain import load_domain_config
from ..hypotheses.temporal_graph_hypotheses import generate_hypotheses
from ..temporal.temporal_kg_builder import PaperRecord, build_temporal_kg

console = Console()


_DEMO_PAPERS = [
    {
        "id": "demo:paper1",
        "title": "Temporal knowledge graphs from scientific literature",
        "year": 2022,
        "text": (
            "We build a temporal knowledge graph by extracting subject-predicate-object triplets from papers. "
            "Entity linking and relation extraction enable longitudinal analysis of scientific claims. "
            "Graph centrality highlights influential concepts and methods."
        ),
    },
    {
        "id": "demo:paper2",
        "title": "Link prediction for hypothesis discovery",
        "year": 2023,
        "text": (
            "Link prediction methods such as Adamic-Adar and Jaccard coefficient can suggest missing edges in a graph. "
            "Community detection reveals subfields; cross-community bridges may correspond to novel hypotheses. "
            "We evaluate predicted edges with held-out publications."
        ),
    },
    {
        "id": "demo:paper3",
        "title": "Agentic graph analysis for science",
        "year": 2024,
        "text": (
            "A code-writing agent can analyze a knowledge graph using shortest paths and centrality metrics. "
            "Temporal trends reveal emerging relations between terms across years. "
            "Experts can label edges to improve extraction and hypothesis ranking."
        ),
    },
]


def demo_paper_records() -> List[PaperRecord]:
    recs: List[PaperRecord] = []
    for p in _DEMO_PAPERS:
        recs.append(
            PaperRecord(
                paper_id=p["id"],
                title=p["title"],
                year=int(p["year"]),
                text=p["text"],
                url="",
                source="demo",
            )
        )
    return recs


def run_demo_pipeline(
    *,
    query: str,
    domain_id: str = "science",
    edge_mode: str = "cooccurrence",
    out_dir: Path = Path("runs"),
    use_llm_for_hypotheses: bool = True,
    top_k: int = 6,
) -> Path:
    """Offline demo run that exercises the full pipeline.

    This is used for smoke tests and for classroom "first run" without external services.
    """

    domain = load_domain_config(domain_id)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    rid = f"demo_{ts}"
    out = out_dir / rid
    out.mkdir(parents=True, exist_ok=True)

    papers = demo_paper_records()
    (out / "paper_records.json").write_text(
        json.dumps([asdict(r) for r in papers], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    kg = build_temporal_kg(papers, domain=domain, query=query, edge_mode=edge_mode)  # type: ignore[arg-type]
    kg.dump_json(out / "temporal_kg.json")

    hyps = generate_hypotheses(
        kg,
        papers=papers,
        domain=domain.title,
        query=query,
        top_k=top_k,
        use_llm=use_llm_for_hypotheses,
    )
    (out / "hypotheses.json").write_text(
        json.dumps([h.model_dump(mode="json") for h in hyps], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    console.print(f"[green]Demo run saved:[/green] {out}")
    return out
