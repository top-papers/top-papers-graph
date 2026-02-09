#!/usr/bin/env python3
"""Export expert artifacts into training datasets (JSONL).

This is a **lightweight** dataset compiler meant for the 200-300-expert workflow:

Inputs (version controlled, small files)
--------------------------------------
* data/experts/trajectories/**/*.yaml
* data/experts/graph_reviews/**/*.json
* data/experts/hypothesis_reviews/**/*.json

Outputs (derived)
-----------------
* data/derived/training/sft_trajectories.jsonl
* data/derived/training/sft_graph_reviews.jsonl
* data/derived/training/sft_hypothesis_reviews.jsonl

The exported format is deliberately compatible with common chat fine-tuning tooling:
each line contains a `messages` list of {role, content}.

Note: This script does NOT attempt to build preference pairs (DPO) yet; that can be added once
the review process produces explicit *alternatives* (accepted vs rejected outputs).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAJ_DIR = REPO_ROOT / "data" / "experts" / "trajectories"
DEFAULT_GRAPH_REV_DIR = REPO_ROOT / "data" / "experts" / "graph_reviews"
DEFAULT_HYP_REV_DIR = REPO_ROOT / "data" / "experts" / "hypothesis_reviews"


@dataclass
class ExportStats:
    trajectories: int = 0
    graph_review_items: int = 0
    hypothesis_reviews: int = 0


def _read_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def export_trajectories(traj_dir: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for p in sorted(traj_dir.glob("**/*.y*ml")):
        doc = _read_yaml(p)
        topic = str(doc.get("topic") or "").strip()
        domain = str(doc.get("domain") or "Science").strip()

        papers = doc.get("papers") or []
        steps = doc.get("steps") or []
        if not isinstance(steps, list) or not steps:
            continue

        paper_list = "\n".join(
            [
                f"- {x.get('id')} ({x.get('year','?')}): {x.get('title','')}"
                for x in papers
                if isinstance(x, dict)
            ]
        )

        for i, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                continue

            ev = step.get("evidence") or {}
            cond = step.get("conditions") or {}

            user = (
                f"Topic: {topic}\n"
                f"Domain: {domain}\n\n"
                f"Papers:\n{paper_list}\n\n"
                f"Step {i}:\n"
                f"Claim: {step.get('claim','')}\n\n"
                f"Evidence: source={ev.get('source')} page={ev.get('page')} type={ev.get('type')}\n"
                f"Evidence snippet: {ev.get('snippet_or_summary','')}\n\n"
                f"Conditions/context: {json.dumps(cond, ensure_ascii=False)}\n\n"
                "Explain the inference and propose the next question."
            )
            assistant = (
                f"Inference: {step.get('inference','')}\n"
                f"Next question: {step.get('next_question','')}\n"
            )

            out.append(
                {
                    "id": f"trajectory:{p.stem}:{i}",
                    "source_file": str(p.relative_to(REPO_ROOT)),
                    "domain": domain,
                    "topic": topic,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a careful scientific assistant. Do not hallucinate. "
                                "Keep track of conditions and cite evidence snippets."
                            ),
                        },
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": assistant},
                    ],
                }
            )

    return out


def export_graph_reviews(graph_dir: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in sorted(graph_dir.glob("**/*.json")):
        doc = _read_json(p)
        domain = str(doc.get("domain") or "Science")
        paper_id = str(doc.get("paper_id") or "unknown")
        items = doc.get("assertions") or []
        if not isinstance(items, list) or not items:
            continue

        for i, a in enumerate(items, start=1):
            if not isinstance(a, dict):
                continue
            ev = a.get("evidence") or {}
            user = (
                f"Domain: {domain}\nPaper: {paper_id}\n\n"
                f"Assertion:\n"
                f"- subject: {a.get('subject')}\n"
                f"- predicate: {a.get('predicate')}\n"
                f"- object: {a.get('object')}\n"
                f"- time_interval/conditions: {a.get('time_interval')}\n\n"
                f"Evidence snippet: {ev.get('snippet_or_summary','')}\n\n"
                "Provide a review verdict and rationale."
            )
            assistant = json.dumps(
                {
                    "verdict": a.get("verdict"),
                    "rationale": a.get("rationale"),
                    "corrections": {
                        "corrected_subject": a.get("corrected_subject"),
                        "corrected_predicate": a.get("corrected_predicate"),
                        "corrected_object": a.get("corrected_object"),
                        "corrected_time_interval": a.get("corrected_time_interval"),
                    },
                },
                ensure_ascii=False,
            )

            out.append(
                {
                    "id": f"graph_review:{p.stem}:{i}",
                    "source_file": str(p.relative_to(REPO_ROOT)),
                    "domain": domain,
                    "paper_id": paper_id,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a strict scientific reviewer. Do not hallucinate.",
                        },
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": assistant},
                    ],
                }
            )
    return out


def export_hypothesis_reviews(hyp_dir: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in sorted(hyp_dir.glob("**/*.json")):
        doc = _read_json(p)
        domain = str(doc.get("domain") or "Science")
        hyp = doc.get("hypothesis") or {}
        scores = doc.get("scores") or {}

        user = (
            f"Domain: {domain}\n\n"
            f"Hypothesis:\n{json.dumps(hyp, ensure_ascii=False, indent=2)}\n\n"
            f"Time/conditions scope: {doc.get('time_scope')}\n\n"
            "Review the hypothesis. Provide accept/reject, score breakdown, and required experiments."
        )
        assistant = json.dumps(
            {
                "accept": doc.get("accept"),
                "scores": scores,
                "major_issues": doc.get("major_issues"),
                "required_experiments": doc.get("required_experiments"),
                "notes": doc.get("notes"),
            },
            ensure_ascii=False,
        )

        out.append(
            {
                "id": f"hypothesis_review:{p.stem}",
                "source_file": str(p.relative_to(REPO_ROOT)),
                "domain": domain,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a skeptical scientific reviewer. Be precise and evidence-driven.",
                    },
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ],
            }
        )
    return out


def main(
    traj_dir: Path = DEFAULT_TRAJ_DIR,
    graph_dir: Path = DEFAULT_GRAPH_REV_DIR,
    hyp_dir: Path = DEFAULT_HYP_REV_DIR,
    out_dir: Path = REPO_ROOT / "data" / "derived" / "training",
) -> ExportStats:
    stats = ExportStats()

    traj_rows = export_trajectories(traj_dir)
    stats.trajectories = _write_jsonl(out_dir / "sft_trajectories.jsonl", traj_rows)

    graph_rows = export_graph_reviews(graph_dir)
    stats.graph_review_items = _write_jsonl(out_dir / "sft_graph_reviews.jsonl", graph_rows)

    hyp_rows = export_hypothesis_reviews(hyp_dir)
    stats.hypothesis_reviews = _write_jsonl(out_dir / "sft_hypothesis_reviews.jsonl", hyp_rows)

    print("✅ Export finished")
    print(f"- trajectories: {stats.trajectories}")
    print(f"- graph review items: {stats.graph_review_items}")
    print(f"- hypothesis reviews: {stats.hypothesis_reviews}")
    print(f"Output dir: {out_dir}")
    return stats


if __name__ == "__main__":
    raise SystemExit(main())
