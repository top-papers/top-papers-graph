from __future__ import annotations

"""End-to-end pipeline: query -> papers -> temporal KG -> hypotheses -> expert report.

This is the main one-command orchestrator used by `top-papers-graph run`.
"""

import json
import re
from dataclasses import asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from rich.console import Console

from ..domain import load_domain_config
from ..graph.build_tg_mmkg import build_temporal_and_multimodal
from ..graph.review_applier import compile_overrides
from ..hypotheses.temporal_graph_hypotheses import generate_hypotheses
from ..ingest.acquire import acquire_pdfs
from ..ingest.mm_pipeline import ingest_pdf_multimodal_auto
from ..ingest.pipeline import ingest_pdf_auto
from ..papers.schema import PaperMetadata
from ..papers.service import search_papers
from ..report import generate_chunk_cards, generate_expert_report, generate_task2_review_cards
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
    include_multimodal: bool = True,
    use_llm_for_hypotheses: bool = True,
    collection_text: Optional[str] = None,
    collection_mm: Optional[str] = None,
    max_chunks_for_triplets: int = 24,
    retrieval_k: int = 10,
) -> Path:
    """Run end-to-end pipeline and return the run directory."""

    domain = load_domain_config(domain_id)
    kg_cfg = domain.kg or {}
    text_collection = collection_text or kg_cfg.get("collection") or domain.domain_id or "science"
    mm_collection = collection_mm if collection_mm is not None else (f"{text_collection}_mm" if include_multimodal else None)

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    rid = f"{ts}_{_slugify(query)}"
    base = run_dir or Path("runs")
    out = base / rid
    out.mkdir(parents=True, exist_ok=True)

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
        json.dumps([p.model_dump(mode="json") for p in selected], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    console.print(f"[green]Selected papers:[/green] {len(selected)}")

    # 2) Acquire PDFs
    console.print("[bold cyan]Acquire PDFs[/bold cyan]")
    acq = acquire_pdfs(selected, raw_dir=out / "raw_pdfs", meta_dir=out / "raw_meta")
    (out / "acquire_results.json").write_text(
        json.dumps(
            [
                asdict(a)
                | {"pdf_path": str(a.pdf_path) if a.pdf_path else None, "meta_path": str(a.meta_path)}
                for a in acq
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # 3) Ingest PDFs
    processed_dir = out / "processed_papers"
    processed_dir.mkdir(parents=True, exist_ok=True)

    ingested_ids: set[str] = set()
    paper_dirs: dict[str, Path] = {}
    for a, paper in zip(acq, selected):
        if not a.pdf_path:
            continue
        meta = paper.model_dump(mode="json")
        try:
            if include_multimodal:
                paper_dir = ingest_pdf_multimodal_auto(a.pdf_path, meta, processed_dir)
            else:
                paper_dir = ingest_pdf_auto(a.pdf_path, meta, processed_dir)
            ingested_ids.add(paper.id)
            paper_dirs[paper.id] = paper_dir
        except Exception as e:
            console.print(f"[yellow]Ingest failed for {paper.id}: {e}[/yellow]")
            continue

    console.print(f"[green]Ingested PDFs:[/green] {len(ingested_ids)}")

    # 4) Build DB indexes / multimodal logical graph
    console.print("[bold cyan]Index chunks + build temporal/MM graph[/bold cyan]")
    indexing_status: list[dict[str, Any]] = []
    for paper in selected:
        paper_dir = paper_dirs.get(paper.id)
        if paper_dir is None:
            continue
        has_mm = bool(include_multimodal and ((paper_dir / "structured_chunks.jsonl").exists() or (paper_dir / "mm" / "pages.jsonl").exists()))
        try:
            build_temporal_and_multimodal(
                paper_dir=paper_dir,
                collection_text=text_collection,
                collection_mm=(mm_collection if has_mm else None),
                domain=domain.title,
                max_chunks_for_triplets=max_chunks_for_triplets,
            )
            indexing_status.append(
                {
                    "paper_id": paper.id,
                    "paper_dir": str(paper_dir.as_posix()),
                    "status": "ok",
                    "multimodal": has_mm,
                }
            )
        except Exception as e:
            console.print(f"[yellow]Index/graph build failed for {paper.id}: {e}[/yellow]")
            indexing_status.append(
                {
                    "paper_id": paper.id,
                    "paper_dir": str(paper_dir.as_posix()),
                    "status": "failed",
                    "multimodal": has_mm,
                    "error": str(e),
                }
            )
    (out / "indexing_status.json").write_text(json.dumps(indexing_status, ensure_ascii=False, indent=2), encoding="utf-8")

    # 5) Load paper texts for KG
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

    # 6) Build temporal KG
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

    # 7) Generate hypotheses
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
        json.dumps([h.model_dump(mode="json") for h in hyps], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

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

    # 8) Review assets
    review_root = out / "review_queue"
    review_root.mkdir(parents=True, exist_ok=True)

    hypothesis_review_dir = review_root / "hypothesis_reviews"
    hypothesis_review_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
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
            "hypothesis": h.model_dump(mode="json"),
        }
        (hypothesis_review_dir / f"{hid}.json").write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")

    chunk_cards_path = review_root / "chunk_cards.jsonl"
    chunk_cards = generate_chunk_cards(processed_dir, out_path=chunk_cards_path)

    task2_dir = review_root / "graph_reviews_auto"
    task2_manifest = generate_task2_review_cards(
        kg,
        processed_dir=processed_dir,
        domain_id=domain.domain_id,
        out_dir=task2_dir,
        max_assertions=250,
    )

    # 9) Expert report + visuals
    report_dir = out / "expert_report"
    report_payload = generate_expert_report(
        query=query,
        domain_title=domain.title,
        domain_id=domain.domain_id,
        kg=kg,
        papers=paper_records,
        processed_dir=processed_dir,
        out_dir=report_dir,
        collection_text=text_collection,
        collection_mm=mm_collection if include_multimodal else None,
        hypotheses=hyps,
        retrieval_k=retrieval_k,
        indexing_status=indexing_status,
    )

    # 10) Run config + manifest
    artifact_manifest = {
        "query": query,
        "domain_id": domain.domain_id,
        "sources": sources,
        "search_limit": search_limit,
        "top_papers": top_papers,
        "include_multimodal": include_multimodal,
        "collection_text": text_collection,
        "collection_mm": mm_collection if include_multimodal else None,
        "max_chunks_for_triplets": max_chunks_for_triplets,
        "retrieval_k": retrieval_k,
        "paper_records": len(paper_records),
        "chunk_cards": len(chunk_cards),
        "task2_files": len(task2_manifest),
        "report_dir": str(report_dir.as_posix()),
        "indexing_status_path": str((out / 'indexing_status.json').as_posix()),
        "review_root": str(review_root.as_posix()),
        "expert_report": str((report_dir / 'expert_report.md').as_posix()),
        "visualizations": report_payload.get("visualizations", {}),
    }
    (out / "artifact_manifest.json").write_text(json.dumps(artifact_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

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
                "collection_text": text_collection,
                "collection_mm": mm_collection if include_multimodal else None,
                "max_chunks_for_triplets": max_chunks_for_triplets,
                "retrieval_k": retrieval_k,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    console.print(f"[bold green]DONE[/bold green] Run dir: {out}")
    return out
