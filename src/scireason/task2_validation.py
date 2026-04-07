from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple

import yaml  # type: ignore

from .task2_offline_review import build_task2_offline_review_package
from .task2_graph_viz import build_interactive_graph_view, compute_graph_analytics, networkx_from_payload, write_graph_html, write_graph_html_variants
from .task2_filters import score_triplet_importance, topic_profile_from_doc, serialize_exclusion_spec, normalize_exclusion_spec
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


def make_hvplot_payload(payload: Dict[str, Any]) -> Tuple[Any, Any]:
    analytics = {}
    try:
        analytics = compute_graph_analytics(payload) if payload else {}
    except Exception:
        analytics = {}
    return build_interactive_graph_view(payload, analytics=analytics, title='Интерактивный граф')


def _score_triplet_rows(rows: list[dict[str, Any]], doc: Dict[str, Any], *, analytics: Dict[str, Any] | None = None) -> list[dict[str, Any]]:
    profile = topic_profile_from_doc(doc)
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item.update(score_triplet_importance(item, profile, graph_metrics=analytics))
        out.append(item)
    return out


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
    exclusion_spec: Dict[str, Any] | str | Path | None = None,
    importance_threshold: float = 0.0,
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
            exclusion_spec=exclusion_spec,
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
    gold_graph_html_light = bundle_dir / "reference_graph_light.html"
    gold_graph_analytics = bundle_dir / "reference_graph_analytics.json"

    gold_payload = json.loads(gold_graph_json.read_text(encoding="utf-8")) if gold_graph_json.exists() else {}
    gold_analytics = compute_graph_analytics(gold_payload) if gold_payload else {}
    if gold_triplets_json.exists():
        gold_rows = json.loads(gold_triplets_json.read_text(encoding="utf-8"))
        gold_rows = gold_rows if isinstance(gold_rows, list) else []
        gold_rows = _score_triplet_rows(gold_rows, doc, analytics=gold_analytics)
        gold_triplets_json.write_text(json.dumps(gold_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_triplets_csv(gold_triplets_json, gold_triplets_csv)
    write_graph_html_variants(gold_graph_json, gold_graph_html, analytics_path=gold_graph_analytics, light_html_path=gold_graph_html_light)

    auto_graph_json = bundle_dir / "automatic_graph" / "temporal_kg.json"
    auto_triplets_json = bundle_dir / "automatic_triplets.json"
    auto_triplets_csv = bundle_dir / "automatic_triplets.csv"
    auto_graph_html = bundle_dir / "automatic_graph.html"
    auto_graph_html_light = bundle_dir / "automatic_graph_light.html"
    auto_graph_analytics = bundle_dir / "automatic_graph_analytics.json"

    auto_analytics = {}
    if auto_graph_json.exists():
        auto_payload = json.loads(auto_graph_json.read_text(encoding="utf-8"))
        auto_analytics = compute_graph_analytics(auto_payload) if auto_payload else {}
        write_graph_html_variants(auto_graph_json, auto_graph_html, analytics_path=auto_graph_analytics, light_html_path=auto_graph_html_light)
    if auto_triplets_json.exists():
        auto_rows = json.loads(auto_triplets_json.read_text(encoding="utf-8"))
        auto_rows = auto_rows if isinstance(auto_rows, list) else []
        auto_rows = _score_triplet_rows(auto_rows, doc, analytics=auto_analytics)
        auto_triplets_json.write_text(json.dumps(auto_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_triplets_csv(auto_triplets_json, auto_triplets_csv)

    review_state_paths = get_task2_review_state_paths(bundle_dir)
    review_state_paths["draft_dir"].mkdir(parents=True, exist_ok=True)

    manifest = {
        "topic": str(doc.get("topic") or ""),
        "bundle_dir": str(bundle_dir),
        "gold_graph": str(gold_graph_json),
        "gold_graph_html": str(gold_graph_html_light),
        "gold_graph_html_light": str(gold_graph_html_light),
        "gold_graph_html_full": str(gold_graph_html),
        "gold_triplets_csv": str(gold_triplets_csv),
        "manifest_version": 7,
        "review_state_dir": str(review_state_paths["draft_dir"]),
        "review_state_latest": str(review_state_paths["latest"]),
        "gold_graph_analytics": str(gold_graph_analytics),
        "filter_defaults": {
            "importance_threshold": max(0.0, min(1.0, float(importance_threshold or 0.0))),
            "cooccurrence_filter_mode": "all",
            "weak_cooccurrence_max_importance": 0.45,
            "exclusion_rules": serialize_exclusion_spec(normalize_exclusion_spec(exclusion_spec)),
        },
    }

    if auto_graph_json.exists():
        manifest.update({
            "auto_run_dir": str(bundle_dir / "automatic_graph"),
            "auto_graph_json": str(auto_graph_json),
            "auto_graph_html": str(auto_graph_html_light),
            "auto_graph_html_light": str(auto_graph_html_light),
            "auto_graph_html_full": str(auto_graph_html),
            "auto_triplets_csv": str(auto_triplets_csv),
            "auto_graph_analytics": str(auto_graph_analytics),
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
        for key in ("llm_effective_provider", "llm_effective_model", "vlm_effective_backend", "vlm_effective_model"):
            if runtime_payload.get(key) not in (None, ""):
                manifest[key] = runtime_payload.get(key)

    offline_review_html = build_task2_offline_review_package(manifest, doc)
    manifest["offline_review_html"] = str(offline_review_html)

    if progress_callback is not None:
        progress_callback({"stage": "manifest", "current": 3, "total": 3, "percent": 100, "message": "Сохраняю notebook manifest"})

    manifest_path = bundle_dir / "task2_notebook_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return BundleResult(bundle_dir=bundle_dir, manifest_path=manifest_path)
