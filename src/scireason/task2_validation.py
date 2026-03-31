from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple

import yaml  # type: ignore

from .task2_offline_review import build_task2_offline_review_package
from .pipeline.task2_validation import (
    build_reference_graph,
    prepare_task2_validation_bundle,
    resolve_papers_from_trajectory,
    suggest_link_candidates,
)


@dataclass
class BundleResult:
    bundle_dir: Path
    manifest_path: Path


def get_task2_review_state_paths(bundle_dir: str | Path) -> Dict[str, Path]:
    root = Path(bundle_dir) / "expert_validation" / "drafts"
    latest = root / "review_state_latest.json"
    return {"draft_dir": root, "latest": latest}


def save_task2_review_state(bundle_dir: str | Path, payload: Dict[str, Any], *, label: str = "manual") -> Path:
    paths = get_task2_review_state_paths(bundle_dir)
    draft_dir = paths["draft_dir"]
    latest = paths["latest"]
    draft_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in str(label or "manual")).strip("-") or "manual"
    versioned = draft_dir / f"review_state_{timestamp}_{safe_label}.json"

    body = dict(payload)
    body.setdefault("artifact_version", 1)
    body["saved_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    body["bundle_dir"] = str(Path(bundle_dir))

    encoded = json.dumps(body, ensure_ascii=False, indent=2)
    versioned.write_text(encoded, encoding="utf-8")
    latest.write_text(encoded, encoding="utf-8")
    return latest


def load_task2_review_state(bundle_dir: str | Path, path: str | Path | None = None) -> Dict[str, Any]:
    target = Path(path) if path else get_task2_review_state_paths(bundle_dir)["latest"]
    if not target.exists():
        return {}
    return json.loads(target.read_text(encoding="utf-8"))


def load_task1_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(doc, dict):
        raise ValueError("Task 1 YAML must contain a top-level object.")
    return doc


def _write_triplets_csv(json_path: Path, csv_path: Path) -> Path:
    rows = json.loads(json_path.read_text(encoding="utf-8"))
    try:
        import pandas as pd  # type: ignore

        pd.DataFrame(rows).to_csv(csv_path, index=False)
    except Exception:
        import csv

        rows = rows if isinstance(rows, list) else []
        fieldnames = sorted({k for row in rows if isinstance(row, dict) for k in row.keys()})
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
                for row in rows:
                    if isinstance(row, dict):
                        writer.writerow(row)
    return csv_path


def _balanced_graph_palette() -> Dict[str, str]:
    return {
        "step": "#4c78a8",
        "paper": "#7f8c8d",
        "term": "#72b7b2",
        "time": "#e0ac2b",
        "assertion": "#b279a2",
        "default": "#9aa5b1",
        "edge": "#94a3b8",
        "edge_temporal": "#7c8ba1",
        "edge_support": "#6ba292",
        "edge_warning": "#c28e6a",
    }


def _node_visual_style(attrs: Dict[str, Any]) -> Dict[str, Any]:
    palette = _balanced_graph_palette()
    raw_type = str(attrs.get("type") or attrs.get("node_type") or "").lower()
    label = str(attrs.get("label") or attrs.get("term") or attrs.get("id") or "")

    group = "default"
    if "trajectory" in raw_type or "step" in raw_type:
        group = "step"
    elif "paper" in raw_type or attrs.get("paper_id") or attrs.get("papers"):
        group = "paper"
    elif "time" in raw_type or any(k in attrs for k in ("yearly_doc_freq", "valid_from", "valid_to")):
        group = "time"
    elif "assertion" in raw_type:
        group = "assertion"
    elif attrs.get("term") or raw_type in {"term", "entity", "concept"}:
        group = "term"

    return {
        "group": group,
        "color": palette.get(group, palette["default"]),
        "label": label,
    }


def _edge_visual_style(attrs: Dict[str, Any]) -> Dict[str, Any]:
    palette = _balanced_graph_palette()
    predicate = str(attrs.get("predicate") or attrs.get("label") or "").lower()
    color = palette["edge"]
    if "time" in predicate or "valid" in predicate:
        color = palette["edge_temporal"]
    elif any(tok in predicate for tok in ("support", "influence", "relate", "correl")):
        color = palette["edge_support"]
    elif any(tok in predicate for tok in ("contrad", "conflict", "reject")):
        color = palette["edge_warning"]
    return {"color": color}


def _networkx_from_payload(payload: Dict[str, Any]):
    import networkx as nx

    directed = True
    edges = payload.get("edges") or []
    if any(not bool(e.get("directed", True)) for e in edges if isinstance(e, dict)):
        directed = False

    G = nx.DiGraph() if directed else nx.Graph()

    for node in payload.get("nodes", []) or []:
        if not isinstance(node, dict):
            continue
        node_id = node.get("id") or node.get("term") or node.get("label")
        if node_id is None:
            continue
        attrs = dict(node)
        attrs.setdefault("label", attrs.get("label") or attrs.get("term") or str(node_id))
        G.add_node(str(node_id), **attrs)

    for edge in edges:
        if not isinstance(edge, dict):
            continue
        src = edge.get("source")
        tgt = edge.get("target") or edge.get("object")
        if src is None or tgt is None:
            continue
        if str(src) not in G:
            G.add_node(str(src), label=str(src))
        if str(tgt) not in G:
            G.add_node(str(tgt), label=str(tgt))
        G.add_edge(str(src), str(tgt), **dict(edge))

    return G


def _graph_analytics(G) -> Dict[str, Any]:
    import networkx as nx

    if G.number_of_nodes() == 0:
        return {"nodes": {}, "summary": {"n_communities": 0, "n_cliques": 0, "max_clique_size": 0}}

    U = G.to_undirected() if hasattr(G, "to_undirected") else G
    degree = nx.degree_centrality(U)
    try:
        closeness = nx.closeness_centrality(U)
    except Exception:
        closeness = {node: 0.0 for node in G.nodes()}
    try:
        betweenness = nx.betweenness_centrality(U) if U.number_of_nodes() <= 250 else {node: 0.0 for node in G.nodes()}
    except Exception:
        betweenness = {node: 0.0 for node in G.nodes()}
    try:
        pagerank = nx.pagerank(G if G.number_of_edges() else U)
    except Exception:
        pagerank = {node: 0.0 for node in G.nodes()}

    try:
        communities = list(nx.community.louvain_communities(U, seed=42)) if U.number_of_edges() else [{node} for node in U.nodes()]
    except Exception:
        try:
            communities = list(nx.community.greedy_modularity_communities(U)) if U.number_of_edges() else [{node} for node in U.nodes()]
        except Exception:
            communities = [{node} for node in U.nodes()]
    community_by_node: Dict[str, int] = {}
    for idx, members in enumerate(communities, start=1):
        for node in members:
            community_by_node[str(node)] = idx

    cliques = []
    try:
        if U.number_of_nodes() <= 120:
            cliques = list(nx.find_cliques(U))
    except Exception:
        cliques = []
    clique_counts: Dict[str, int] = {str(node): 0 for node in U.nodes()}
    max_clique_size_by_node: Dict[str, int] = {str(node): 0 for node in U.nodes()}
    for clique in cliques:
        size = len(clique)
        for node in clique:
            key = str(node)
            clique_counts[key] = clique_counts.get(key, 0) + 1
            max_clique_size_by_node[key] = max(max_clique_size_by_node.get(key, 0), size)

    node_stats: Dict[str, Any] = {}
    for node in G.nodes():
        key = str(node)
        node_stats[key] = {
            "degree_centrality": float(degree.get(node, 0.0) or 0.0),
            "betweenness_centrality": float(betweenness.get(node, 0.0) or 0.0),
            "closeness_centrality": float(closeness.get(node, 0.0) or 0.0),
            "pagerank": float(pagerank.get(node, 0.0) or 0.0),
            "community_id": int(community_by_node.get(key, 0) or 0),
            "clique_count": int(clique_counts.get(key, 0) or 0),
            "max_clique_size": int(max_clique_size_by_node.get(key, 0) or 0),
        }

    return {
        "nodes": node_stats,
        "summary": {
            "n_communities": len(communities),
            "n_cliques": len(cliques),
            "max_clique_size": max((len(c) for c in cliques), default=0),
        },
    }


def make_hvplot_payload(payload: Dict[str, Any]) -> Tuple[Any, Any]:
    G = _networkx_from_payload(payload)
    try:
        import hvplot.networkx as hvnx  # noqa: F401
        import networkx as nx

        node_colors = []
        for node_id, attrs in G.nodes(data=True):
            style = _node_visual_style(dict(attrs))
            G.nodes[node_id]["viz_group"] = style["group"]
            G.nodes[node_id]["viz_color"] = style["color"]
            node_colors.append(style["color"])

        edge_colors = []
        for src, tgt, attrs in G.edges(data=True):
            style = _edge_visual_style(dict(attrs))
            G.edges[src, tgt]["viz_color"] = style["color"]
            edge_colors.append(style["color"])

        pos = nx.spring_layout(G, seed=7, k=0.9 / max(1, G.number_of_nodes() ** 0.5))
        plot = hvnx.draw(
            G,
            pos,
            node_size=14,
            node_color=node_colors,
            edge_color=edge_colors,
            with_labels=False,
            arrowhead_length=0.012,
            edge_line_width=1.2,
            width=950,
            height=650,
        ).opts(bgcolor="#ffffff")
        return G, plot
    except Exception:
        return G, None


def _write_graph_html(graph_json_path: Path, html_path: Path) -> Path:
    payload = json.loads(graph_json_path.read_text(encoding="utf-8"))
    G = _networkx_from_payload(payload)
    analytics = _graph_analytics(G)
    community_palette = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2", "#ff9da6", "#9d755d", "#bab0ab"]

    try:
        from pyvis.network import Network  # type: ignore

        net = Network(height="820px", width="100%", directed=bool(getattr(G, "is_directed", lambda: True)()), notebook=False, filter_menu=True)
        net.barnes_hut()
        try:
            net.show_buttons(filter_=["physics", "layout", "interaction"])
        except Exception:
            pass

        for node_id, attrs in G.nodes(data=True):
            label = attrs.get("label") or attrs.get("term") or str(node_id)
            style = _node_visual_style(dict(attrs))
            node_metrics = analytics.get("nodes", {}).get(str(node_id), {})
            community_id = int(node_metrics.get("community_id") or 0)
            community_color = community_palette[(community_id - 1) % len(community_palette)] if community_id else style["color"]
            node_size = 14 + 34 * max(float(node_metrics.get("degree_centrality") or 0.0), float(node_metrics.get("pagerank") or 0.0))
            title_parts = [f"label: {label}"]
            title_parts.extend(f"{k}: {v}" for k, v in attrs.items() if k != "label")
            title_parts.extend([
                f"community_id: {community_id}",
                f"degree_centrality: {float(node_metrics.get('degree_centrality') or 0.0):.4f}",
                f"betweenness_centrality: {float(node_metrics.get('betweenness_centrality') or 0.0):.4f}",
                f"closeness_centrality: {float(node_metrics.get('closeness_centrality') or 0.0):.4f}",
                f"pagerank: {float(node_metrics.get('pagerank') or 0.0):.4f}",
                f"clique_count: {int(node_metrics.get('clique_count') or 0)}",
                f"max_clique_size: {int(node_metrics.get('max_clique_size') or 0)}",
            ])
            net.add_node(
                str(node_id),
                label=str(label)[:80],
                title="\n".join(title_parts),
                color=community_color,
                group=style["group"],
                value=node_size,
                community_id=community_id,
                degree_centrality=float(node_metrics.get("degree_centrality") or 0.0),
                betweenness_centrality=float(node_metrics.get("betweenness_centrality") or 0.0),
                closeness_centrality=float(node_metrics.get("closeness_centrality") or 0.0),
                pagerank=float(node_metrics.get("pagerank") or 0.0),
                clique_count=int(node_metrics.get("clique_count") or 0),
                max_clique_size=int(node_metrics.get("max_clique_size") or 0),
            )

        for src, tgt, attrs in G.edges(data=True):
            label = attrs.get("predicate") or attrs.get("label") or ""
            title = "\n".join(f"{k}: {v}" for k, v in attrs.items())
            importance = float(attrs.get("importance_score") or 0.0)
            net.add_edge(str(src), str(tgt), label=str(label), title=title, width=1.0 + 4.0 * importance, importance_score=importance)

        net.write_html(str(html_path), notebook=False)
    except Exception:
        title = graph_json_path.stem
        summary = analytics.get("summary", {})
        html = [
            "<html><head><meta charset='utf-8'><title>%s</title></head><body>" % title,
            "<h2>%s</h2>" % title,
            "<p>pyvis is not installed, so this fallback HTML shows a compact graph summary with graph analytics.</p>",
            f"<p><b>Communities:</b> {summary.get('n_communities', 0)} · <b>Cliques:</b> {summary.get('n_cliques', 0)} · <b>Max clique size:</b> {summary.get('max_clique_size', 0)}</p>",
            "<h3>Nodes</h3><ul>",
        ]
        for node_id, attrs in G.nodes(data=True):
            label = attrs.get("label") or attrs.get("term") or str(node_id)
            node_metrics = analytics.get("nodes", {}).get(str(node_id), {})
            html.append(
                f"<li><b>{label}</b> <code>{node_id}</code> — community {int(node_metrics.get('community_id') or 0)}, "
                f"degree {float(node_metrics.get('degree_centrality') or 0.0):.3f}, pagerank {float(node_metrics.get('pagerank') or 0.0):.3f}, "
                f"max clique {int(node_metrics.get('max_clique_size') or 0)}</li>"
            )
        html.append("</ul><h3>Edges</h3><ul>")
        for src, tgt, attrs in G.edges(data=True):
            label = attrs.get("predicate") or attrs.get("label") or "related_to"
            importance = float(attrs.get("importance_score") or 0.0)
            html.append(f"<li><code>{src}</code> — <b>{label}</b> → <code>{tgt}</code> (importance={importance:.3f})</li>")
        html.append("</ul></body></html>")
        html_path.write_text("".join(html), encoding="utf-8")
    return html_path


def build_task2_validation_bundle(
    trajectory_path: str | Path,
    *,
    out_dir: str | Path,
    include_auto_pipeline: bool = True,
    multimodal: bool = True,
    enable_reference_scout: bool = True,
    run_vlm: bool = True,
    edge_mode: str = "auto",
    max_papers: int = 0,
    max_link_queries: int = 4,
    enable_remote_lookup: bool = False,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    g4f_model: str | None = None,
    local_model: str | None = None,
    vlm_backend: str | None = None,
    vlm_model_id: str | None = None,
    excluded_paper_ids: Optional[list[str]] = None,
    excluded_paper_titles: Optional[list[str]] = None,
    excluded_title_contains: Optional[list[str]] = None,
    min_importance: Optional[float] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> BundleResult:
    trajectory_path = Path(trajectory_path)
    out_dir = Path(out_dir)

    doc = load_task1_yaml(trajectory_path)
    run_name = str(doc.get("submission_id") or trajectory_path.stem)
    bundle_dir = out_dir / run_name

    if include_auto_pipeline:
        bundle_dir = prepare_task2_validation_bundle(
            trajectory_path,
            out_dir=out_dir,
            include_multimodal=multimodal,
            run_vlm=run_vlm,
            edge_mode=edge_mode,
            suggest_links=enable_reference_scout,
            max_papers=max_papers,
            max_link_queries=max_link_queries,
            enable_remote_lookup=enable_remote_lookup,
            llm_provider=llm_provider,
            llm_model=llm_model,
            g4f_model=g4f_model,
            local_model=local_model,
            vlm_backend=vlm_backend,
            vlm_model_id=vlm_model_id,
            excluded_paper_ids=excluded_paper_ids,
            excluded_paper_titles=excluded_paper_titles,
            excluded_title_contains=excluded_title_contains,
            min_importance=min_importance,
            progress_callback=progress_callback,
        )
    else:
        if progress_callback is not None:
            progress_callback({"stage": "manual", "current": 1, "total": 3, "percent": 33, "message": "Готовлю bundle без автоматического pipeline"})
        bundle_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(trajectory_path, bundle_dir / trajectory_path.name)

        reference_graph = build_reference_graph(doc)
        (bundle_dir / "reference_graph.json").write_text(
            json.dumps(reference_graph, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (bundle_dir / "reference_triplets.json").write_text(
            json.dumps(reference_graph.get("triplets") or [], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if enable_reference_scout:
            if progress_callback is not None:
                progress_callback({"stage": "scout", "current": 2, "total": 3, "percent": 67, "message": "Генерирую reference scout"})
            try:
                resolved = resolve_papers_from_trajectory(doc, enable_remote_lookup=enable_remote_lookup)
                suggestions = suggest_link_candidates(
                    doc,
                    known_papers=resolved,
                    max_queries=max_link_queries,
                    enable_remote_lookup=enable_remote_lookup,
                )
            except Exception:
                suggestions = []

            scout_dir = bundle_dir / "scout"
            scout_dir.mkdir(parents=True, exist_ok=True)
            (scout_dir / "suggested_links.json").write_text(
                json.dumps(suggestions, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    gold_graph_json = bundle_dir / "reference_graph.json"
    gold_triplets_json = bundle_dir / "reference_triplets.json"
    gold_triplets_csv = bundle_dir / "reference_triplets.csv"
    gold_graph_html = bundle_dir / "reference_graph.html"

    _write_triplets_csv(gold_triplets_json, gold_triplets_csv)
    _write_graph_html(gold_graph_json, gold_graph_html)

    auto_graph_json = bundle_dir / "automatic_graph" / "temporal_kg.json"
    auto_triplets_json = bundle_dir / "automatic_triplets.json"
    auto_triplets_csv = bundle_dir / "automatic_triplets.csv"
    auto_graph_html = bundle_dir / "automatic_graph.html"

    if auto_triplets_json.exists():
        _write_triplets_csv(auto_triplets_json, auto_triplets_csv)
    if auto_graph_json.exists():
        _write_graph_html(auto_graph_json, auto_graph_html)

    review_state_paths = get_task2_review_state_paths(bundle_dir)
    review_state_paths["draft_dir"].mkdir(parents=True, exist_ok=True)

    manifest = {
        "topic": str(doc.get("topic") or ""),
        "bundle_dir": str(bundle_dir),
        "gold_graph": str(gold_graph_json),
        "gold_graph_html": str(gold_graph_html),
        "gold_triplets_csv": str(gold_triplets_csv),
        "manifest_version": 6,
        "review_state_dir": str(review_state_paths["draft_dir"]),
        "review_state_latest": str(review_state_paths["latest"]),
    }

    if auto_graph_json.exists():
        manifest.update({
            "auto_run_dir": str(bundle_dir / "automatic_graph"),
            "auto_graph_json": str(auto_graph_json),
            "auto_graph_html": str(auto_graph_html),
            "auto_triplets_csv": str(auto_triplets_csv),
            "auto_triplets_json": str(auto_triplets_json),
        })

    comparison = bundle_dir / "comparison_summary.json"
    if comparison.exists():
        manifest["comparison_summary"] = str(comparison)

    scout = bundle_dir / "scout" / "suggested_links.json"
    if scout.exists():
        manifest["reference_scout"] = str(scout)

    runtime_manifest = bundle_dir / "manifest.json"
    if runtime_manifest.exists():
        try:
            runtime_payload = json.loads(runtime_manifest.read_text(encoding="utf-8"))
        except Exception:
            runtime_payload = {}
        for key in ("llm_effective_provider", "llm_effective_model", "vlm_effective_backend", "vlm_effective_model", "default_importance_threshold", "excluded_papers_applied", "task2_controls"):
            if runtime_payload.get(key) not in (None, ""):
                manifest[key] = runtime_payload.get(key)

    offline_review_html = build_task2_offline_review_package(manifest, doc)
    manifest["offline_review_html"] = str(offline_review_html)

    if progress_callback is not None:
        progress_callback({"stage": "manifest", "current": 3, "total": 3, "percent": 100, "message": "Сохраняю notebook manifest"})

    manifest_path = bundle_dir / "task2_notebook_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return BundleResult(bundle_dir=bundle_dir, manifest_path=manifest_path)
