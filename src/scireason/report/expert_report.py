from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..agentic.graph_tools import (
    build_nx_graph,
    communities_greedy_modularity,
    cross_community_bridges,
    graph_summary,
    link_prediction,
    spectral_link_prediction,
    top_central_nodes,
)
from ..graph.graphrag_query import retrieve_context
from ..graph.mm_retrieval import retrieve_mm
from ..schemas import HypothesisDraft
from ..temporal.temporal_kg_builder import PaperRecord, TemporalKnowledgeGraph
from ..tgnn.event_dataset import build_event_stream, event_stats
from ..tgnn.tgn_link_prediction import TGNLinkPredConfig, tgn_link_prediction
from .expert_cards import extract_condition_hints, generate_chunk_cards


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_\-]{3,}")


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "")}


def _preview(text: str, limit: int = 280) -> str:
    t = re.sub(r"\s+", " ", text or "").strip()
    if len(t) <= limit:
        return t
    return t[: limit - 1] + "…"


def _load_chunk_cards(processed_dir: Path) -> List[Dict[str, Any]]:
    return generate_chunk_cards(processed_dir, out_path=None)


def _fallback_local_retrieval(processed_dir: Path, query: str, *, limit: int = 8, image_only: bool = False) -> List[Dict[str, Any]]:
    q = _tokenize(query)
    rows: list[tuple[float, dict[str, Any]]] = []
    for card in _load_chunk_cards(processed_dir):
        text = str(card.get("text_preview") or card.get("summary") or "")
        if image_only and not card.get("image_path"):
            continue
        toks = _tokenize(text)
        overlap = len(q & toks)
        score = float(overlap) / float(max(1, len(q)))
        if card.get("modality") in {"figure", "table"}:
            score += 0.15
        if card.get("condition_hints"):
            score += 0.05
        if score <= 0.0:
            continue
        payload = {
            "paper_id": card.get("paper_id"),
            "chunk_id": card.get("chunk_id"),
            "modality": card.get("modality"),
            "page": card.get("page"),
            "figure_or_table": card.get("figure_or_table"),
            "text": text,
            "summary": card.get("summary"),
            "image_path": card.get("image_path"),
        }
        rows.append((score, {"id": card.get("chunk_id"), "score": round(score, 6), "payload": payload}))
    rows.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in rows[:limit]]


def _safe_retrieve_text(collection_text: Optional[str], processed_dir: Path, query: str, limit: int) -> Tuple[List[Dict[str, Any]], str]:
    if collection_text:
        try:
            return retrieve_context(collection=collection_text, query=query, limit=limit), "qdrant"
        except Exception:
            pass
    return _fallback_local_retrieval(processed_dir, query, limit=limit, image_only=False), "local"


def _safe_retrieve_mm(collection_mm: Optional[str], processed_dir: Path, query: str, limit: int) -> Tuple[List[Dict[str, Any]], str]:
    if collection_mm:
        try:
            return retrieve_mm(collection_mm=collection_mm, query=query, limit=limit), "qdrant_mm"
        except Exception:
            pass
    return _fallback_local_retrieval(processed_dir, query, limit=limit, image_only=True), "local_mm"


def _timeline_counts(kg: TemporalKnowledgeGraph) -> Dict[int, int]:
    counts: Dict[int, int] = defaultdict(int)
    for edge in kg.edges:
        for year, value in (edge.yearly_count or {}).items():
            try:
                counts[int(year)] += int(value)
            except Exception:
                continue
    return dict(sorted(counts.items()))


def _emerging_edges(kg: TemporalKnowledgeGraph, *, limit: int = 12) -> List[Dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for edge in kg.edges:
        rows.append(
            {
                "source": edge.source,
                "predicate": edge.predicate,
                "target": edge.target,
                "score": round(float(edge.score or 0.0), 6),
                "trend": round(float(edge.features.get("trend", 0.0) or 0.0), 6),
                "years": dict(sorted((edge.yearly_count or {}).items())),
                "papers": sorted(list(edge.papers))[:10],
            }
        )
    rows.sort(key=lambda r: (float(r["trend"]), float(r["score"])), reverse=True)
    return rows[:limit]


def _build_graph_analysis(kg: TemporalKnowledgeGraph) -> Dict[str, Any]:
    G = build_nx_graph(kg, directed=False, min_total_count=1)
    comms = communities_greedy_modularity(G, max_communities=10)
    analysis = {
        "summary": graph_summary(G),
        "centrality": top_central_nodes(G, k=12),
        "communities": comms,
        "community_sizes": [len(c) for c in comms],
        "cross_community_bridges": [
            {"source": u, "target": v, "score": round(float(s), 6)}
            for u, v, s in cross_community_bridges(G, comms, top_k=15)
        ],
        "classic_link_prediction": {
            "adamic_adar": [
                {"source": u, "target": v, "score": round(float(s), 6)}
                for u, v, s in link_prediction(G, method="adamic_adar", k=15)
            ],
            "spectral": [
                {"source": u, "target": v, "score": round(float(s), 6)}
                for u, v, s in spectral_link_prediction(G, dim=8, k=15)
            ],
        },
    }
    return analysis


def _build_tgnn_analysis(kg: TemporalKnowledgeGraph, papers: Sequence[PaperRecord]) -> Dict[str, Any]:
    events = build_event_stream(kg, papers=papers)
    cfg = TGNLinkPredConfig()
    preds = tgn_link_prediction(events, top_k=15, config=cfg)
    return {
        "event_stats": event_stats(events),
        "config": {
            "recent_window_years": int(cfg.recent_window_years),
            "recency_half_life_years": float(cfg.recency_half_life_years),
            "node_memory_weight": float(cfg.node_memory_weight),
            "pair_repeat_weight": float(cfg.pair_repeat_weight),
            "common_neighbor_weight": float(cfg.common_neighbor_weight),
            "min_candidate_score": float(cfg.min_candidate_score),
        },
        "predictions": [
            {"source": u, "target": v, "score": round(float(s), 6)} for u, v, s in preds
        ],
    }


def _paper_summary(papers: Sequence[PaperRecord]) -> Dict[str, Any]:
    years = [int(p.year) for p in papers if p.year is not None]
    return {
        "n_papers": len(papers),
        "years": {"min": min(years) if years else None, "max": max(years) if years else None},
        "paper_ids": [p.paper_id for p in papers[:50]],
    }


def _inventory_from_chunk_cards(cards: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    modality_counter = Counter(str(card.get("modality") or "unknown") for card in cards)
    image_counter = sum(1 for card in cards if card.get("image_path"))
    papers = {str(card.get("paper_id") or "") for card in cards if card.get("paper_id")}
    return {
        "n_chunk_cards": len(cards),
        "n_papers_with_chunks": len(papers),
        "modalities": dict(sorted(modality_counter.items())),
        "image_bearing_chunks": image_counter,
    }


def _graph_plot(kg: TemporalKnowledgeGraph, out_path: Path) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import networkx as nx  # type: ignore
    except Exception:
        return None

    G = build_nx_graph(kg, directed=False, min_total_count=1)
    if G.number_of_nodes() == 0:
        return None

    central = top_central_nodes(G, k=24).get("pagerank") or top_central_nodes(G, k=24).get("degree") or []
    nodes = [n for n, _ in central]
    if not nodes:
        nodes = list(G.nodes())[:24]
    H = G.subgraph(nodes).copy()
    if H.number_of_nodes() == 0:
        return None

    pos = nx.spring_layout(H, seed=7)
    plt.figure(figsize=(12, 8))
    nx.draw_networkx(H, pos=pos, with_labels=True, node_size=800, font_size=8, width=1.0)
    plt.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    return str(out_path.as_posix())


def _timeline_plot(kg: TemporalKnowledgeGraph, out_path: Path) -> Optional[str]:
    counts = _timeline_counts(kg)
    if not counts:
        return None
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return None

    years = list(sorted(counts))
    values = [counts[y] for y in years]
    plt.figure(figsize=(11, 4))
    plt.plot(years, values, marker="o")
    plt.xlabel("Year")
    plt.ylabel("Assertions / edge evidence")
    plt.title("Temporal KG evidence density")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    return str(out_path.as_posix())


def _community_plot(analysis: Dict[str, Any], out_path: Path) -> Optional[str]:
    sizes = list(analysis.get("community_sizes") or [])
    if not sizes:
        return None
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return None

    xs = list(range(1, len(sizes) + 1))
    plt.figure(figsize=(8, 4))
    plt.bar(xs, sizes)
    plt.xlabel("Community")
    plt.ylabel("Nodes")
    plt.title("Community sizes in the logical graph")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    return str(out_path.as_posix())


def _write_markdown(
    *,
    query: str,
    domain_title: str,
    collection_text: Optional[str],
    collection_mm: Optional[str],
    paper_summary: Dict[str, Any],
    inventory: Dict[str, Any],
    retrieval_text: List[Dict[str, Any]],
    retrieval_text_backend: str,
    retrieval_mm: List[Dict[str, Any]],
    retrieval_mm_backend: str,
    graph_analysis: Dict[str, Any],
    tgnn_analysis: Dict[str, Any],
    emerging_edges: List[Dict[str, Any]],
    hypotheses: Sequence[HypothesisDraft],
    visuals: Dict[str, Optional[str]],
    indexing_status: Sequence[Dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append(f"# Expert report for query: {query}")
    lines.append("")
    lines.append(f"Generated: {datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    lines.append("")
    lines.append("## Pipeline overview")
    lines.append("")
    lines.append(f"- Domain: {domain_title}")
    lines.append(f"- Papers in run: {paper_summary.get('n_papers', 0)}")
    lines.append(f"- Text collection: {collection_text or 'not configured'}")
    lines.append(f"- MM collection: {collection_mm or 'not configured'}")
    lines.append(f"- Chunk cards: {inventory.get('n_chunk_cards', 0)}")
    lines.append(f"- Chunk modalities: {json.dumps(inventory.get('modalities', {}), ensure_ascii=False)}")
    lines.append("")

    lines.append("## Vector retrieval (query-centred evidence)")
    lines.append("")
    lines.append(f"Text retrieval backend: **{retrieval_text_backend}**")
    for idx, hit in enumerate(retrieval_text[:8], start=1):
        payload = hit.get("payload", {})
        lines.append(
            f"{idx}. [{payload.get('modality') or 'text'}] {payload.get('paper_id')} / page {payload.get('page')} — {_preview(str(payload.get('text') or payload.get('summary') or ''), 220)}"
        )
    if not retrieval_text:
        lines.append("- No text hits found.")
    lines.append("")

    lines.append("Multimodal retrieval backend: **%s**" % retrieval_mm_backend)
    for idx, hit in enumerate(retrieval_mm[:8], start=1):
        payload = hit.get("payload", {})
        lines.append(
            f"{idx}. [{payload.get('modality') or 'mm'}] {payload.get('paper_id')} / {payload.get('figure_or_table') or 'image'} — {_preview(str(payload.get('text') or payload.get('summary') or ''), 220)}"
        )
    if not retrieval_mm:
        lines.append("- No multimodal hits found.")
    lines.append("")

    lines.append("## Temporal + logical graph analysis")
    lines.append("")
    lines.append(f"- Summary: {json.dumps(graph_analysis.get('summary', {}), ensure_ascii=False)}")
    lines.append(f"- Top central nodes: {json.dumps(graph_analysis.get('centrality', {}), ensure_ascii=False)}")
    lines.append(f"- Community sizes: {graph_analysis.get('community_sizes', [])}")
    lines.append("")

    lines.append("### Emerging temporal relations")
    for row in emerging_edges[:10]:
        lines.append(
            f"- {row['source']} | {row['predicate']} | {row['target']}  (trend={row['trend']}, score={row['score']}, years={row['years']})"
        )
    if not emerging_edges:
        lines.append("- No emerging edges detected.")
    lines.append("")

    lines.append("### TGNN candidate links")
    for row in (tgnn_analysis.get("predictions") or [])[:10]:
        lines.append(f"- {row['source']} ↔ {row['target']}  (score={row['score']})")
    if not tgnn_analysis.get("predictions"):
        lines.append("- No TGNN candidate links detected.")
    lines.append("")

    lines.append("## Hypotheses")
    lines.append("")
    for idx, hyp in enumerate(hypotheses[:8], start=1):
        lines.append(f"### H-{idx:03d}: {hyp.title}")
        lines.append("")
        lines.append(f"- Premise: {hyp.premise}")
        lines.append(f"- Mechanism: {hyp.mechanism}")
        lines.append(f"- Time/conditions: {hyp.time_scope}")
        lines.append(f"- Experiment: {hyp.proposed_experiment}")
        lines.append(f"- Confidence: {hyp.confidence_score}/10")
        if hyp.supporting_evidence:
            lines.append("- Evidence:")
            for ev in hyp.supporting_evidence[:4]:
                lines.append(f"  - {ev.source_id}: {ev.text_snippet}")
        lines.append("")

    lines.append("## Visualizations")
    lines.append("")
    for key, path in visuals.items():
        if path:
            lines.append(f"- {key}: `{path}`")
    if not any(visuals.values()):
        lines.append("- Matplotlib not available; plots were skipped.")
    lines.append("")

    if indexing_status:
        lines.append("## Database indexing status")
        lines.append("")
        for row in indexing_status:
            status = row.get("status")
            paper_id = row.get("paper_id")
            error = row.get("error")
            if error:
                lines.append(f"- {paper_id}: {status} — {error}")
            else:
                lines.append(f"- {paper_id}: {status}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def generate_expert_report(
    *,
    query: str,
    domain_title: str,
    domain_id: str,
    kg: TemporalKnowledgeGraph,
    papers: Sequence[PaperRecord],
    processed_dir: Path,
    out_dir: Path,
    collection_text: Optional[str] = None,
    collection_mm: Optional[str] = None,
    hypotheses: Optional[Sequence[HypothesisDraft]] = None,
    retrieval_k: int = 10,
    indexing_status: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_cards = _load_chunk_cards(processed_dir)
    inventory = _inventory_from_chunk_cards(chunk_cards)
    paper_summary = _paper_summary(papers)

    retrieval_text, retrieval_text_backend = _safe_retrieve_text(collection_text, processed_dir, query, retrieval_k)
    retrieval_mm, retrieval_mm_backend = _safe_retrieve_mm(collection_mm, processed_dir, query, retrieval_k)

    graph_analysis = _build_graph_analysis(kg)
    tgnn_analysis = _build_tgnn_analysis(kg, papers)
    emerging = _emerging_edges(kg, limit=15)

    visuals = {
        "temporal_graph": _graph_plot(kg, out_dir / "temporal_graph.png"),
        "timeline": _timeline_plot(kg, out_dir / "temporal_timeline.png"),
        "communities": _community_plot(graph_analysis, out_dir / "community_sizes.png"),
    }

    payload = {
        "query": query,
        "domain": {"id": domain_id, "title": domain_title},
        "paper_summary": paper_summary,
        "inventory": inventory,
        "retrieval": {
            "text_backend": retrieval_text_backend,
            "text_hits": retrieval_text,
            "mm_backend": retrieval_mm_backend,
            "mm_hits": retrieval_mm,
        },
        "graph_analysis": graph_analysis,
        "tgnn_analysis": tgnn_analysis,
        "emerging_edges": emerging,
        "hypotheses": [h.model_dump(mode="json") for h in (hypotheses or [])],
        "visualizations": visuals,
        "indexing_status": list(indexing_status or []),
    }
    (out_dir / "expert_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md = _write_markdown(
        query=query,
        domain_title=domain_title,
        collection_text=collection_text,
        collection_mm=collection_mm,
        paper_summary=paper_summary,
        inventory=inventory,
        retrieval_text=retrieval_text,
        retrieval_text_backend=retrieval_text_backend,
        retrieval_mm=retrieval_mm,
        retrieval_mm_backend=retrieval_mm_backend,
        graph_analysis=graph_analysis,
        tgnn_analysis=tgnn_analysis,
        emerging_edges=emerging,
        hypotheses=hypotheses or [],
        visuals=visuals,
        indexing_status=indexing_status or [],
    )
    (out_dir / "expert_report.md").write_text(md, encoding="utf-8")
    return payload
