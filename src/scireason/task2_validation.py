from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml
from rich.console import Console

from .llm import chat_json
from .papers.normalize import normalize_arxiv, normalize_openalex
from .papers.schema import ExternalIds, PaperMetadata, PaperSource
from .papers.service import get_paper_by_doi, search_papers
from .pipeline.e2e import run_pipeline
from .review_schema import NEG_INFINITY, POS_INFINITY, normalize_temporal_payload
from .connectors import arxiv as arxiv_connector, openalex as openalex_connector

console = Console()

_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
_WORD_RE = re.compile(r"[\w\-]{3,}", re.UNICODE)


@dataclass
class Task2Bundle:
    bundle_dir: Path
    gold_dir: Path
    auto_run_dir: Optional[Path]
    manifest_path: Path
    notebook_inputs_dir: Path


@dataclass
class ResolvedPaper:
    requested_id: str
    requested_title: str
    requested_year: Optional[int]
    metadata: PaperMetadata
    resolution: str
    score: float


# -----------------------------
# YAML loading / normalization
# -----------------------------


def load_task1_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Task1 YAML must be a mapping: {path}")
    return data


def _slugify(text: str, *, max_len: int = 80) -> str:
    text = re.sub(r"[^a-zA-Z0-9_\-]+", "_", str(text or "").strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return (text[:max_len] or "item").strip("_")


def _paper_spec_iter(data: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for rec in data.get("papers") or []:
        if isinstance(rec, dict):
            yield rec


# -----------------------------
# Paper resolution for auto-pipeline
# -----------------------------


def _normalize_doi(raw: str) -> Optional[str]:
    raw = str(raw or "").strip()
    if not raw:
        return None
    raw = raw.replace("DOI:", "").replace("doi:", "").strip()
    raw = raw.replace("https://doi.org/", "").replace("http://doi.org/", "")
    m = _DOI_RE.search(raw)
    if m:
        return m.group(0)
    return None


def _resolve_by_openalex(raw_id: str) -> Optional[PaperMetadata]:
    raw = str(raw_id or "").strip()
    if not raw:
        return None
    if "openalex.org" not in raw and not raw.lower().startswith("openalex:"):
        return None
    try:
        work_id = raw.split(":", 1)[1] if raw.lower().startswith("openalex:") else raw
        return normalize_openalex(openalex_connector.get_work(work_id))
    except Exception:
        return None


def _resolve_by_arxiv(raw_id: str) -> Optional[PaperMetadata]:
    raw = str(raw_id or "").strip()
    if not raw:
        return None
    raw = raw.replace("arxiv:", "")
    if "arxiv.org" in raw:
        raw = raw.rstrip("/").split("/")[-1].replace(".pdf", "")
    if not re.match(r"^\d{4}\.\d{4,5}(v\d+)?$|^[a-z\-]+/\d{7}(v\d+)?$", raw, re.IGNORECASE):
        return None
    try:
        rows = arxiv_connector.get_by_id(raw)
        if rows:
            return normalize_arxiv(rows[0])
    except Exception:
        return None
    return None


def _title_score(candidate: PaperMetadata, title: str, year: Optional[int]) -> float:
    qt = {t.lower() for t in _WORD_RE.findall(title or "")}
    ct = {t.lower() for t in _WORD_RE.findall(candidate.title or "")}
    overlap = len(qt & ct) / max(1.0, len(qt))
    year_bonus = 0.2 if year is not None and candidate.year == year else 0.0
    id_bonus = 0.1 if candidate.pdf_url else 0.0
    return float(overlap) + year_bonus + id_bonus


def resolve_task1_papers(
    data: Dict[str, Any],
    *,
    search_limit: int = 8,
) -> List[ResolvedPaper]:
    resolved: List[ResolvedPaper] = []
    for rec in _paper_spec_iter(data):
        raw_id = str(rec.get("id") or "").strip()
        title = str(rec.get("title") or "").strip()
        year = rec.get("year")
        try:
            year_i = int(year) if year not in (None, "") else None
        except Exception:
            year_i = None

        meta: Optional[PaperMetadata] = None
        resolution = "fallback"
        score = 0.0

        doi = _normalize_doi(raw_id)
        if doi:
            meta = get_paper_by_doi(doi)
            if meta is not None:
                resolution = "doi"
                score = 1.0

        if meta is None:
            meta = _resolve_by_openalex(raw_id)
            if meta is not None:
                resolution = "openalex"
                score = 0.95

        if meta is None:
            meta = _resolve_by_arxiv(raw_id)
            if meta is not None:
                resolution = "arxiv"
                score = 0.95

        if meta is None and title:
            try:
                candidates = search_papers(title, limit=search_limit)
            except Exception:
                candidates = []
            if candidates:
                best = max(candidates, key=lambda c: _title_score(c, title, year_i))
                best_score = _title_score(best, title, year_i)
                if best_score >= 0.35:
                    meta = best
                    resolution = "title_search"
                    score = best_score

        if meta is None:
            canonical_id = raw_id or f"manual:{_slugify(title or 'paper')}"
            ids = ExternalIds(doi=doi)
            meta = PaperMetadata(
                id=canonical_id,
                source=PaperSource.unknown,
                title=title or canonical_id,
                year=year_i,
                url=raw_id if raw_id.startswith("http") else None,
                ids=ids,
            )
            resolution = "manual_fallback"
            score = 0.0

        resolved.append(
            ResolvedPaper(
                requested_id=raw_id,
                requested_title=title,
                requested_year=year_i,
                metadata=meta,
                resolution=resolution,
                score=float(score),
            )
        )
    return resolved


# -----------------------------
# Gold graph construction
# -----------------------------


def _paper_year_map(data: Dict[str, Any]) -> Dict[str, Optional[int]]:
    out: Dict[str, Optional[int]] = {}
    for rec in _paper_spec_iter(data):
        key = str(rec.get("id") or "").strip()
        val = rec.get("year")
        try:
            out[key] = int(val) if val not in (None, "") else None
        except Exception:
            out[key] = None
    return out


def _paper_title_map(data: Dict[str, Any]) -> Dict[str, str]:
    return {str(rec.get("id") or "").strip(): str(rec.get("title") or "").strip() for rec in _paper_spec_iter(data)}


def _step_end_date(step: Dict[str, Any], year_map: Dict[str, Optional[int]]) -> str:
    years: List[int] = []
    for src in step.get("sources") or []:
        if not isinstance(src, dict):
            continue
        py = year_map.get(str(src.get("source") or "").strip())
        if py is not None:
            years.append(py)
    return str(max(years)) if years else POS_INFINITY


def build_gold_graph_from_trajectory(data: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = list(data.get("steps") or [])
    edges = list(data.get("edges") or [])
    year_map = _paper_year_map(data)
    title_map = _paper_title_map(data)

    nodes: List[Dict[str, Any]] = []
    graph_edges: List[Dict[str, Any]] = []
    triplets: List[Dict[str, Any]] = []

    for rec in _paper_spec_iter(data):
        paper_id = str(rec.get("id") or "").strip()
        nodes.append(
            {
                "id": f"paper::{paper_id}",
                "label": str(rec.get("title") or paper_id),
                "kind": "paper",
                "paper_id": paper_id,
                "year": rec.get("year"),
                "title": rec.get("title"),
            }
        )

    for step in steps:
        sid = int(step.get("step_id") or 0)
        claim = str(step.get("claim") or "").strip()
        inference = str(step.get("inference") or "").strip()
        nodes.append(
            {
                "id": f"step::{sid}",
                "label": claim or f"Step {sid}",
                "kind": "step",
                "step_id": sid,
                "claim": claim,
                "inference": inference,
                "next_question": str(step.get("next_question") or "").strip(),
            }
        )

        for idx, src in enumerate(step.get("sources") or [], start=1):
            if not isinstance(src, dict):
                continue
            paper_id = str(src.get("source") or "").strip()
            end_hint = str(year_map.get(paper_id)) if year_map.get(paper_id) is not None else POS_INFINITY
            temporal = normalize_temporal_payload(
                {},
                start_date_hint=NEG_INFINITY,
                end_date_hint=end_hint,
                valid_from_hint=end_hint,
                valid_to_hint=POS_INFINITY,
                temporal_basis="manual_yaml_source",
            )
            edge = {
                "source": f"paper::{paper_id}",
                "target": f"step::{sid}",
                "predicate": "supports_claim",
                "evidence": {
                    "type": src.get("type"),
                    "snippet_or_summary": src.get("snippet_or_summary"),
                    "locator": src.get("locator"),
                },
                "conditions": dict(step.get("conditions") or {}),
                **temporal,
            }
            graph_edges.append(edge)
            triplets.append(
                {
                    "graph": "gold",
                    "triple_id": f"gold-support-{sid}-{idx}",
                    "subject": title_map.get(paper_id) or paper_id,
                    "predicate": "supports_claim",
                    "object": claim,
                    **temporal,
                    "conditions": dict(step.get("conditions") or {}),
                    "evidence_summary": src.get("snippet_or_summary") or "",
                    "locator": src.get("locator") or "",
                    "paper_id": paper_id,
                    "step_id": sid,
                    "relation_kind": "manual_support",
                }
            )

    step_by_id = {int(s.get("step_id") or 0): s for s in steps if int(s.get("step_id") or 0) > 0}
    for idx, pair in enumerate(edges, start=1):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        src_id, dst_id = int(pair[0]), int(pair[1])
        src_step = step_by_id.get(src_id, {})
        dst_step = step_by_id.get(dst_id, {})
        end_hint = _step_end_date(dst_step, year_map)
        temporal = normalize_temporal_payload(
            {},
            start_date_hint=NEG_INFINITY,
            end_date_hint=end_hint,
            valid_from_hint=end_hint,
            valid_to_hint=POS_INFINITY,
            temporal_basis="manual_reasoning_edge",
        )
        edge = {
            "source": f"step::{src_id}",
            "target": f"step::{dst_id}",
            "predicate": "leads_to",
            "evidence": {"snippet_or_summary": str(src_step.get("inference") or "")},
            **temporal,
        }
        graph_edges.append(edge)
        triplets.append(
            {
                "graph": "gold",
                "triple_id": f"gold-reason-{idx}",
                "subject": str(src_step.get("claim") or f"Step {src_id}"),
                "predicate": "leads_to",
                "object": str(dst_step.get("claim") or f"Step {dst_id}"),
                **temporal,
                "conditions": dict(dst_step.get("conditions") or {}),
                "evidence_summary": str(src_step.get("inference") or ""),
                "locator": "",
                "paper_id": "",
                "step_id": src_id,
                "relation_kind": "manual_reasoning",
            }
        )

    payload = {
        "graph_type": "gold_manual_trajectory",
        "topic": data.get("topic"),
        "domain": data.get("domain"),
        "submission_id": data.get("submission_id"),
        "nodes": nodes,
        "edges": graph_edges,
        "triplets": triplets,
    }
    (out_dir / "gold_graph.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_triplets(triplets, out_dir / "gold_triplets.jsonl", out_dir / "gold_triplets.csv")
    _write_graph_html(payload, out_dir / "gold_graph.html", title=f"Gold graph: {data.get('topic')}")
    return payload


# -----------------------------
# Auto graph exports / comparison
# -----------------------------


def _write_triplets(rows: Sequence[Dict[str, Any]], jsonl_path: Path, csv_path: Path) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def export_auto_triplets(auto_run_dir: Path, out_dir: Path) -> Dict[str, Any]:
    kg_path = auto_run_dir / "temporal_kg.json"
    if not kg_path.exists():
        raise FileNotFoundError(f"Missing temporal_kg.json in {auto_run_dir}")
    kg = json.loads(kg_path.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    graph_edges: List[Dict[str, Any]] = []
    nodes: Dict[str, Dict[str, Any]] = {}

    for edge in kg.get("edges") or []:
        years = sorted(int(y) for y in (edge.get("yearly_count") or {}).keys())
        temporal = normalize_temporal_payload(
            {},
            start_date_hint=str(years[0]) if years else NEG_INFINITY,
            end_date_hint=str(years[-1]) if years else POS_INFINITY,
            valid_from_hint=str(years[-1]) if years else POS_INFINITY,
            valid_to_hint=POS_INFINITY,
            temporal_basis="auto_temporal_kg",
        )
        subject = str(edge.get("source") or "")
        predicate = str(edge.get("predicate") or "may_relate_to")
        obj = str(edge.get("target") or "")
        rows.append(
            {
                "graph": "auto",
                "triple_id": _slugify(f"{subject}|{predicate}|{obj}"),
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                **temporal,
                "conditions": {"yearly_count": edge.get("yearly_count") or {}},
                "evidence_summary": ((edge.get("evidence_quotes") or [{}])[0] or {}).get("quote") or "",
                "locator": "",
                "paper_id": ", ".join(edge.get("papers") or []),
                "step_id": "",
                "relation_kind": "auto_edge",
                "score": edge.get("score"),
            }
        )
        nodes.setdefault(subject, {"id": subject, "label": subject, "kind": "auto_term"})
        nodes.setdefault(obj, {"id": obj, "label": obj, "kind": "auto_term"})
        graph_edges.append({"source": subject, "target": obj, "predicate": predicate, **temporal, "score": edge.get("score")})

    payload = {
        "graph_type": "auto_temporal_kg",
        "meta": kg.get("meta") or {},
        "nodes": list(nodes.values()),
        "edges": graph_edges,
        "triplets": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "auto_graph.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_triplets(rows, out_dir / "auto_triplets.jsonl", out_dir / "auto_triplets.csv")
    _write_graph_html(payload, out_dir / "auto_graph.html", title="Auto temporal graph")
    return payload


def compare_gold_vs_auto(gold_graph: Dict[str, Any], auto_graph: Dict[str, Any], out_path: Path) -> Dict[str, Any]:
    gold_tokens = {t.lower() for row in gold_graph.get("triplets") or [] for t in _WORD_RE.findall(f"{row.get('subject','')} {row.get('object','')}")}
    auto_tokens = {t.lower() for row in auto_graph.get("triplets") or [] for t in _WORD_RE.findall(f"{row.get('subject','')} {row.get('object','')}")}
    overlap = sorted(gold_tokens & auto_tokens)
    report = {
        "gold_triplets": len(gold_graph.get("triplets") or []),
        "auto_triplets": len(auto_graph.get("triplets") or []),
        "gold_nodes": len(gold_graph.get("nodes") or []),
        "auto_nodes": len(auto_graph.get("nodes") or []),
        "token_overlap_count": len(overlap),
        "token_overlap_sample": overlap[:50],
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


# -----------------------------
# Reference scout (TODO #4)
# -----------------------------

_REF_SCHEMA = """
{
  "queries": ["search query 1", "search query 2", "search query 3"]
}
"""


def build_reference_scout(data: Dict[str, Any], out_dir: Path, *, max_candidates_per_query: int = 5) -> Dict[str, Any]:
    topic = str(data.get("topic") or "").strip()
    step_claims = [str(s.get("claim") or "").strip() for s in (data.get("steps") or [])[:5] if isinstance(s, dict)]
    queries = [topic, *step_claims]
    llm_queries: List[str] = []
    try:
        prompt = json.dumps({"topic": topic, "claims": step_claims}, ensure_ascii=False)
        resp = chat_json(
            system=(
                "You are a literature scout for temporal knowledge graph validation. "
                "Return 3-6 precise scholarly search queries that could find validating or contradicting papers."
            ),
            user=prompt,
            schema_hint=_REF_SCHEMA,
            temperature=0.0,
        )
        llm_queries = [str(q).strip() for q in (resp.get("queries") or []) if str(q).strip()]
    except Exception:
        llm_queries = []

    merged_queries: List[str] = []
    for q in [*queries, *llm_queries]:
        if q and q not in merged_queries:
            merged_queries.append(q)

    hits: List[Dict[str, Any]] = []
    for q in merged_queries[:8]:
        try:
            candidates = search_papers(q, limit=max_candidates_per_query)
        except Exception:
            candidates = []
        hits.append(
            {
                "query": q,
                "results": [
                    {
                        "id": p.id,
                        "title": p.title,
                        "year": p.year,
                        "url": p.url,
                        "pdf_url": p.pdf_url,
                        "source": str(p.source),
                    }
                    for p in candidates[:max_candidates_per_query]
                ],
            }
        )

    payload = {"topic": topic, "queries": merged_queries, "hits": hits}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reference_scout.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


# -----------------------------
# Visualization helpers
# -----------------------------


def _write_graph_html(graph_payload: Dict[str, Any], html_path: Path, *, title: str) -> Optional[Path]:
    try:
        from pyvis.network import Network  # type: ignore
    except Exception:
        return None

    html_path.parent.mkdir(parents=True, exist_ok=True)
    directed = True
    net = Network(height="760px", width="100%", directed=directed, bgcolor="#ffffff", font_color="#222222")
    net.barnes_hut(gravity=-4500, central_gravity=0.25, spring_length=180, spring_strength=0.02, overlap=0.05)

    for node in graph_payload.get("nodes") or []:
        label = str(node.get("label") or node.get("id") or "")
        title_parts = [f"<b>{label}</b>"]
        for k in ("kind", "paper_id", "year", "claim", "inference", "next_question"):
            if node.get(k) not in (None, ""):
                title_parts.append(f"<br><b>{k}:</b> {node.get(k)}")
        net.add_node(str(node.get("id") or label), label=label[:80], title="".join(title_parts), shape="dot")

    for edge in graph_payload.get("edges") or []:
        e_title = [f"<b>{edge.get('predicate') or ''}</b>"]
        for k in ("start_date", "end_date", "valid_from", "valid_to", "score"):
            if edge.get(k) not in (None, ""):
                e_title.append(f"<br><b>{k}:</b> {edge.get(k)}")
        ev = edge.get("evidence") or {}
        if isinstance(ev, dict) and ev.get("snippet_or_summary"):
            e_title.append(f"<br><b>evidence:</b> {ev.get('snippet_or_summary')}")
        net.add_edge(str(edge.get("source")), str(edge.get("target")), label=str(edge.get("predicate") or ""), title="".join(e_title))

    net.heading = title
    net.show(str(html_path))
    return html_path


def make_hvplot_payload(graph_payload: Dict[str, Any]) -> Tuple[Any, Any]:
    import networkx as nx  # type: ignore
    import hvplot.networkx as hvnx  # type: ignore

    G = nx.DiGraph()
    for node in graph_payload.get("nodes") or []:
        G.add_node(str(node.get("id") or node.get("label") or ""), **node)
    for edge in graph_payload.get("edges") or []:
        G.add_edge(str(edge.get("source")), str(edge.get("target")), predicate=str(edge.get("predicate") or ""), **edge)
    pos = nx.spring_layout(G, seed=42)
    plot = hvnx.draw(G, pos=pos, with_labels=True, width=1100, height=760, node_size=18, arrowhead_length=0.02)
    return G, plot


# -----------------------------
# Bundle orchestration / CLI entrypoint
# -----------------------------


def build_task2_validation_bundle(
    trajectory_path: Path,
    *,
    out_dir: Path,
    include_auto_pipeline: bool = True,
    multimodal: bool = True,
    no_llm_hypotheses: bool = False,
    collection_text: Optional[str] = None,
    collection_mm: Optional[str] = None,
    max_chunks_for_triplets: int = 24,
    retrieval_k: int = 10,
    enable_reference_scout: bool = True,
) -> Task2Bundle:
    data = load_task1_yaml(trajectory_path)
    topic = str(data.get("topic") or trajectory_path.stem)
    domain_id = str(data.get("domain") or "science")

    bundle_dir = out_dir / _slugify(f"task2_{trajectory_path.stem}")
    bundle_dir.mkdir(parents=True, exist_ok=True)
    notebook_inputs = bundle_dir / "notebook_inputs"
    notebook_inputs.mkdir(parents=True, exist_ok=True)
    (notebook_inputs / trajectory_path.name).write_text(trajectory_path.read_text(encoding="utf-8"), encoding="utf-8")

    gold_dir = bundle_dir / "gold_graph"
    gold_graph = build_gold_graph_from_trajectory(data, gold_dir)

    auto_run_dir: Optional[Path] = None
    auto_graph: Optional[Dict[str, Any]] = None
    resolved_records: List[ResolvedPaper] = []
    if include_auto_pipeline:
        resolved_records = resolve_task1_papers(data)
        resolved_payload = [
            {
                "requested_id": r.requested_id,
                "requested_title": r.requested_title,
                "requested_year": r.requested_year,
                "resolution": r.resolution,
                "score": r.score,
                "metadata": r.metadata.model_dump(mode="json"),
            }
            for r in resolved_records
        ]
        (bundle_dir / "resolved_papers.json").write_text(json.dumps(resolved_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        auto_parent = bundle_dir / "auto_pipeline"
        auto_parent.mkdir(parents=True, exist_ok=True)
        auto_run_dir = run_pipeline(
            query=topic,
            domain_id=domain_id,
            selected_papers=[r.metadata for r in resolved_records],
            top_papers=max(1, len(resolved_records)),
            run_dir=auto_parent,
            include_multimodal=multimodal,
            use_llm_for_hypotheses=not no_llm_hypotheses,
            collection_text=collection_text,
            collection_mm=collection_mm,
            max_chunks_for_triplets=max_chunks_for_triplets,
            retrieval_k=retrieval_k,
        )
        auto_graph = export_auto_triplets(auto_run_dir, bundle_dir / "auto_graph")
        compare_gold_vs_auto(gold_graph, auto_graph, bundle_dir / "comparison_summary.json")

    if enable_reference_scout:
        build_reference_scout(data, bundle_dir / "reference_scout")

    manifest = {
        "trajectory_path": str(trajectory_path.as_posix()),
        "topic": topic,
        "domain": domain_id,
        "bundle_dir": str(bundle_dir.as_posix()),
        "gold_graph": str((gold_dir / "gold_graph.json").as_posix()),
        "gold_triplets_csv": str((gold_dir / "gold_triplets.csv").as_posix()),
        "gold_graph_html": str((gold_dir / "gold_graph.html").as_posix()),
        "auto_run_dir": str(auto_run_dir.as_posix()) if auto_run_dir else None,
        "auto_graph_json": str((bundle_dir / "auto_graph" / "auto_graph.json").as_posix()) if auto_run_dir else None,
        "auto_triplets_csv": str((bundle_dir / "auto_graph" / "auto_triplets.csv").as_posix()) if auto_run_dir else None,
        "auto_graph_html": str((bundle_dir / "auto_graph" / "auto_graph.html").as_posix()) if auto_run_dir else None,
        "comparison_summary": str((bundle_dir / "comparison_summary.json").as_posix()) if auto_run_dir else None,
        "reference_scout": str((bundle_dir / "reference_scout" / "reference_scout.json").as_posix()) if enable_reference_scout else None,
    }
    manifest_path = bundle_dir / "task2_bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return Task2Bundle(bundle_dir=bundle_dir, gold_dir=gold_dir, auto_run_dir=auto_run_dir, manifest_path=manifest_path, notebook_inputs_dir=notebook_inputs)
