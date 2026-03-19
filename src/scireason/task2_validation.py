from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml  # type: ignore

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


def make_hvplot_payload(payload: Dict[str, Any]) -> Tuple[Any, Any]:
    G = _networkx_from_payload(payload)
    try:
        import hvplot.networkx as hvnx  # noqa: F401
        import networkx as nx

        pos = nx.spring_layout(G, seed=7)
        plot = hvnx.draw(
            G,
            pos,
            node_size=10,
            with_labels=False,
            arrowhead_length=0.015,
            width=950,
            height=650,
        )
        return G, plot
    except Exception:
        return G, None


def _write_graph_html(graph_json_path: Path, html_path: Path) -> Path:
    payload = json.loads(graph_json_path.read_text(encoding="utf-8"))
    G = _networkx_from_payload(payload)

    try:
        from pyvis.network import Network  # type: ignore

        net = Network(height="750px", width="100%", directed=True, notebook=False)
        net.barnes_hut()

        for node_id, attrs in G.nodes(data=True):
            label = attrs.get("label") or attrs.get("term") or str(node_id)
            title = "\n".join(f"{k}: {v}" for k, v in attrs.items() if k != "label")
            net.add_node(str(node_id), label=str(label)[:80], title=title)

        for src, tgt, attrs in G.edges(data=True):
            label = attrs.get("predicate") or attrs.get("label") or ""
            title = "\n".join(f"{k}: {v}" for k, v in attrs.items())
            net.add_edge(str(src), str(tgt), label=str(label), title=title)

        net.write_html(str(html_path), notebook=False)
    except Exception:
        title = graph_json_path.stem
        html = [
            "<html><head><meta charset='utf-8'><title>%s</title></head><body>" % title,
            "<h2>%s</h2>" % title,
            "<p>pyvis is not installed, so this fallback HTML shows a compact graph summary.</p>",
            "<h3>Nodes</h3><ul>",
        ]
        for node_id, attrs in G.nodes(data=True):
            label = attrs.get("label") or attrs.get("term") or str(node_id)
            html.append(f"<li><b>{label}</b> <code>{node_id}</code></li>")
        html.append("</ul><h3>Edges</h3><ul>")
        for src, tgt, attrs in G.edges(data=True):
            label = attrs.get("predicate") or attrs.get("label") or "related_to"
            html.append(f"<li><code>{src}</code> — <b>{label}</b> → <code>{tgt}</code></li>")
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
        )
    else:
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
            try:
                resolved = resolve_papers_from_trajectory(doc)
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

    manifest = {
        "topic": str(doc.get("topic") or ""),
        "bundle_dir": str(bundle_dir),
        "gold_graph": str(gold_graph_json),
        "gold_graph_html": str(gold_graph_html),
        "gold_triplets_csv": str(gold_triplets_csv),
        "manifest_version": 4,
    }

    if auto_graph_json.exists():
        manifest.update({
            "auto_run_dir": str(bundle_dir / "automatic_graph"),
            "auto_graph_json": str(auto_graph_json),
            "auto_graph_html": str(auto_graph_html),
            "auto_triplets_csv": str(auto_triplets_csv),
        })

    comparison = bundle_dir / "comparison_summary.json"
    if comparison.exists():
        manifest["comparison_summary"] = str(comparison)

    scout = bundle_dir / "scout" / "suggested_links.json"
    if scout.exists():
        manifest["reference_scout"] = str(scout)

    manifest_path = bundle_dir / "task2_notebook_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return BundleResult(bundle_dir=bundle_dir, manifest_path=manifest_path)
