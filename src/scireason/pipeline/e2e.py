from __future__ import annotations

"""End-to-end pipeline: query -> papers -> temporal KG -> hypotheses.

This is the *missing orchestrator* that turns the repo into a fully automated pipeline.

Design goals
------------
* One command should be enough for the student:
    top-papers-graph run --query "..."
* Best-effort automation: if PDFs are not available, keep going with abstract-only mode.
* The pipeline should automatically pick up expert labels from `data/experts/...`.
"""

import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from rich.console import Console

from ..domain import load_domain_config
from ..graph.review_applier import compile_overrides
from ..hypotheses.temporal_graph_hypotheses import generate_hypotheses
from ..ingest.acquire import AcquireResult, acquire_pdfs
from ..ingest.mm_pipeline import ingest_pdf_multimodal_auto
from ..ingest.pipeline import ingest_pdf_auto
from ..papers.schema import PaperMetadata
from ..papers.service import search_papers
from ..temporal.temporal_kg_builder import PaperRecord, build_temporal_kg, load_papers_from_processed


console = Console()


def _slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9\-\s_]+", "", s)
    s = re.sub(r"\s+", "-", s).strip("-")
    return s[:60] or "query"


def _rank_papers(papers: Sequence[PaperMetadata], query: str) -> List[PaperMetadata]:
    """Heuristic reranker: relevance overlap + citations + OA PDF availability."""

    q = set(re.findall(r"[a-z0-9]+", (query or "").lower()))

    def score(p: PaperMetadata) -> float:
        text = f"{p.title} {p.abstract or ''}".lower()
        toks = set(re.findall(r"[a-z0-9]+", text))
        overlap = (len(q & toks) / float(max(1, len(q)))) if q else 0.0
        cites = float(p.citation_count or 0)
        year = float(p.year or 0)
        has_pdf = 1.0 if p.pdf_url else 0.0
        return 3.0 * overlap + 0.0008 * cites + 0.0005 * year + 0.6 * has_pdf

    return sorted(list(papers), key=score, reverse=True)


def _paperrecord_from_metadata(p: PaperMetadata) -> PaperRecord:
    text = (p.abstract or "").strip()
    if not text:
        text = p.title
    return PaperRecord(
        paper_id=p.id,
        title=p.title,
        year=p.year,
        text=text,
        url=p.url or "",
        source=str(p.source),
    )


def run_pipeline(
    *,
    query: str,
    domain_id: str = "science",
    sources: Optional[List[str]] = None,
    search_limit: int = 50,
    top_papers: int = 20,
    run_dir: Optional[Path] = None,
    include_multimodal: bool = False,
    use_llm_for_hypotheses: bool = True,
) -> Path:
    """Run end-to-end pipeline and return the run directory."""

    domain = load_domain_config(domain_id)

    # Run id
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    rid = f"{ts}_{_slugify(query)}"
    base = run_dir or Path("runs")
    out = base / rid
    out.mkdir(parents=True, exist_ok=True)

    # Compile expert overrides (if any) so this run automatically benefits from labels.
    overrides_path = Path("data/derived/expert_overrides.jsonl")
    try:
        stats = compile_overrides(Path("data/experts/graph_reviews"), overrides_path)
        console.print(
            f"[green]Expert overrides:[/green] accepted={stats.accepted} rejected={stats.rejected} needs_fix={stats.needs_fix} added={stats.added}"
        )
    except Exception as e:
        console.print(f"[yellow]Could not compile expert overrides: {e}[/yellow]")

    # 1) Search papers
    console.print(f"[bold cyan]Search[/bold cyan] query='{query}' sources={sources or ['all']}")
    papers = search_papers(query, sources=sources, limit=search_limit)
    papers = _rank_papers(papers, query)
    selected = papers[:top_papers]

    (out / "papers_selected.json").write_text(
        json.dumps([p.model_dump() for p in selected], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    console.print(f"[green]Selected papers:[/green] {len(selected)}")

    # 2) Acquire PDFs (best-effort)
    console.print("[bold cyan]Acquire PDFs[/bold cyan]")
    acq = acquire_pdfs(selected, raw_dir=out / "raw_pdfs", meta_dir=out / "raw_meta")
    (out / "acquire_results.json").write_text(
        json.dumps([asdict(a) | {"pdf_path": str(a.pdf_path) if a.pdf_path else None, "meta_path": str(a.meta_path)} for a in acq], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 3) Ingest PDFs (optional; pipeline still works without it)
    processed_dir = out / "processed_papers"
    processed_dir.mkdir(parents=True, exist_ok=True)

    ingested_ids: set[str] = set()
    for a, paper in zip(acq, selected):
        if not a.pdf_path:
            continue
        meta = paper.model_dump()
        try:
            if include_multimodal:
                ingest_pdf_multimodal_auto(a.pdf_path, meta, processed_dir)
            else:
                ingest_pdf_auto(a.pdf_path, meta, processed_dir)
            ingested_ids.add(paper.id)
        except Exception as e:
            console.print(f"[yellow]Ingest failed for {paper.id}: {e}[/yellow]")
            continue

    console.print(f"[green]Ingested PDFs:[/green] {len(ingested_ids)}")

    # 4) Load paper texts for KG
    pr_processed = load_papers_from_processed(processed_dir)
    pr_by_id = {p.paper_id: p for p in pr_processed}
    paper_records: List[PaperRecord] = []
    for p in selected:
        if p.id in pr_by_id:
            paper_records.append(pr_by_id[p.id])
        else:
            paper_records.append(_paperrecord_from_metadata(p))

    (out / "paper_records.json").write_text(
        json.dumps([asdict(r) for r in paper_records], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 5) Build temporal KG
    edge_mode = str((domain.term_graph or {}).get("edge_mode", "auto"))
    console.print(f"[bold cyan]Build temporal KG[/bold cyan] edge_mode={edge_mode}")
    kg = build_temporal_kg(
        paper_records,
        domain=domain,
        query=query,
        edge_mode=edge_mode,  # type: ignore[arg-type]
        expert_overrides_path=overrides_path,
    )
    kg.dump_json(out / "temporal_kg.json")
    console.print(f"[green]KG edges:[/green] {len(kg.edges)} nodes: {len(kg.nodes)}")

    # 6) Generate hypotheses
    console.print("[bold cyan]Generate hypotheses[/bold cyan]")
    hyps = generate_hypotheses(
        kg,
        papers=paper_records,
        domain=domain.title,
        query=query,
        top_k=8,
        use_llm=use_llm_for_hypotheses,
        expert_overrides_path=overrides_path,
    )

    (out / "hypotheses.json").write_text(
        json.dumps([h.model_dump() for h in hyps], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Markdown report
    md_lines = [f"# Hypotheses for: {query}", "", f"Domain: {domain.title} ({domain.domain_id})", ""]
    for i, h in enumerate(hyps, 1):
        md_lines += [
            f"## H-{i:03d}: {h.title}",
            "",
            f"**Premise:** {h.premise}",
            "",
            f"**Mechanism:** {h.mechanism}",
            "",
            f"**Time/conditions scope:** {h.time_scope}",
            "",
            f"**Proposed experiment:** {h.proposed_experiment}",
            "",
        ]
        if h.supporting_evidence:
            md_lines.append("**Evidence:**")
            for ev in h.supporting_evidence:
                md_lines.append(f"- {ev.source_id}: {ev.text_snippet}")
            md_lines.append("")
    (out / "hypotheses.md").write_text("\n".join(md_lines), encoding="utf-8")

    # 7) Export expert labeling templates (so quality can improve during the course)
    review_dir = out / "review_queue" / "hypothesis_reviews"
    review_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    for i, h in enumerate(hyps, 1):
        hid = f"H-{i:06d}"
        review = {
            "domain": domain.domain_id,
            "hypothesis_id": hid,
            "reviewer_id": "",
            "timestamp": now,
            "scores": {"novelty": 0, "soundness": 0, "testability": 0},
            "time_scope": h.time_scope,
            "major_issues": [],
            "minor_issues": [],
            "required_experiments": h.proposed_experiment,
            "accept": False,
            "suggested_revision": "",
            "hypothesis": h.model_dump(),
        }
        (review_dir / f"{hid}.json").write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")

    # Run config
    (out / "run_config.json").write_text(
        json.dumps(
            {
                "query": query,
                "domain_id": domain_id,
                "sources": sources,
                "search_limit": search_limit,
                "top_papers": top_papers,
                "include_multimodal": include_multimodal,
                "edge_mode": edge_mode,
                "use_llm_for_hypotheses": use_llm_for_hypotheses,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    console.print(f"[bold green]DONE[/bold green] Run dir: {out}")
    return out
