#!/usr/bin/env python3
"""
Validate expert artifacts (YAML/JSON) for required fields:
- evidence present
- conditions present
- time_scope / time_interval present (where applicable)

Exit code:
- 0 if all good
- 1 if any problems found

Designed to be used in CI on pull requests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import yaml  # type: ignore
except Exception:
    print("ERROR: Missing dependency PyYAML. Install with `pip install pyyaml`.", file=sys.stderr)
    raise

REPO_ROOT = Path(__file__).resolve().parents[2]

TRAJ_DIR = REPO_ROOT / "data" / "experts" / "trajectories"
GRAPH_REV_DIR = REPO_ROOT / "data" / "experts" / "graph_reviews"
HYP_REV_DIR = REPO_ROOT / "data" / "experts" / "hypothesis_reviews"
MM_REV_DIR = REPO_ROOT / "data" / "experts" / "mm_reviews"
TEMP_CORR_DIR = REPO_ROOT / "data" / "experts" / "temporal_corrections"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _is_empty(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and not v.strip():
        return True
    if isinstance(v, (list, dict)) and len(v) == 0:
        return True
    return False


def _required_condition_keys(domain: str) -> List[str]:
    """Domain-specific condition requirements for trajectories.

    Why: different scientific domains have different "boundary conditions" that must accompany a claim.
    The list should be defined in the domain YAML (artifact_validation.trajectory_required_conditions).
    """
    d = (domain or "").strip().lower()

    # Try load from domain config: configs/domains/<domain>.yaml
    candidate_ids: List[str] = []
    if d:
        candidate_ids.append(d)
        # common normalization: keep only letters/numbers/underscore
        norm = "".join(ch if ch.isalnum() else "_" for ch in d)
        if norm != d:
            candidate_ids.append(norm)

    for cid in candidate_ids:
        cfg_path = REPO_ROOT / "configs" / "domains" / f"{cid}.yaml"
        if not cfg_path.exists():
            continue
        try:
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            av = cfg.get("artifact_validation") or {}
            req = av.get("trajectory_required_conditions") or []
            if isinstance(req, list):
                return [str(x) for x in req]
        except Exception:
            pass

    return []


def validate_trajectory(path: Path) -> List[str]:
    errs: List[str] = []
    doc = _load_yaml(path)

    steps = doc.get("steps", [])
    if not isinstance(steps, list) or len(steps) == 0:
        return ["steps[] must be a non-empty list"]

    for i, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            errs.append(f"step {i}: must be object")
            continue

        evidence = step.get("evidence", {})
        conditions = step.get("conditions", {})

        if _is_empty(step.get("claim")):
            errs.append(f"step {i}: missing claim")

        if not isinstance(evidence, dict) or _is_empty(evidence):
            errs.append(f"step {i}: missing evidence")
        else:
            if _is_empty(evidence.get("source")):
                errs.append(f"step {i}: evidence.source required")
            if _is_empty(evidence.get("page")):
                errs.append(f"step {i}: evidence.page required")
            if _is_empty(evidence.get("snippet_or_summary")):
                errs.append(f"step {i}: evidence.snippet_or_summary required")
            etype = str(evidence.get("type", "")).strip().lower()
            if etype in {"figure", "table"} and _is_empty(evidence.get("figure_or_table")):
                errs.append(f"step {i}: evidence.figure_or_table required for {etype}")

        if not isinstance(conditions, dict) or _is_empty(conditions):
            errs.append(f"step {i}: missing conditions")
        else:
            # Domain-specific required condition keys (configurable)
            domain = str(doc.get("domain") or "").strip().lower()
            required = _required_condition_keys(domain)
            for k in required:
                if _is_empty(conditions.get(k)):
                    errs.append(f"step {i}: conditions.{k} required (use 'unknown' if not stated)")

    return errs


def validate_graph_review(path: Path) -> List[str]:
    errs: List[str] = []
    doc = _load_json(path)

    assertions = doc.get("assertions", [])
    if not isinstance(assertions, list) or len(assertions) == 0:
        return ["assertions[] must be a non-empty list"]

    for i, a in enumerate(assertions, start=1):
        if not isinstance(a, dict):
            errs.append(f"assertion {i}: must be object")
            continue

        for k in ("subject", "predicate", "object"):
            if _is_empty(a.get(k)):
                errs.append(f"assertion {i}: missing {k}")

        if _is_empty(a.get("time_interval")):
            errs.append(f"assertion {i}: missing time_interval (use 'unknown' if not stated)")

        ev = a.get("evidence", {})
        if not isinstance(ev, dict) or _is_empty(ev):
            errs.append(f"assertion {i}: missing evidence")
        else:
            if _is_empty(ev.get("page")):
                errs.append(f"assertion {i}: evidence.page required")
            if _is_empty(ev.get("snippet_or_summary")):
                errs.append(f"assertion {i}: evidence.snippet_or_summary required")

        verdict = str(a.get("verdict", "")).strip()
        if verdict not in {"accepted", "rejected", "needs_time_fix", "needs_evidence_fix", "added"}:
            errs.append(f"assertion {i}: invalid verdict '{verdict}'")

        if _is_empty(a.get("rationale")):
            errs.append(f"assertion {i}: missing rationale")

    return errs


def validate_hypothesis_review(path: Path) -> List[str]:
    errs: List[str] = []
    doc = _load_json(path)

    if _is_empty(doc.get("time_scope")):
        errs.append("missing time_scope (conditions/time applicability)")

    scores = doc.get("scores")
    if not isinstance(scores, dict):
        errs.append("missing scores object")
    else:
        for k in ("novelty", "soundness", "testability"):
            if k not in scores:
                errs.append(f"missing scores.{k}")

    if "accept" not in doc:
        errs.append("missing accept (true/false)")
    if _is_empty(doc.get("major_issues")):
        errs.append("missing major_issues[] (at least 1, or ['none'] if truly clean)")
    if _is_empty(doc.get("required_experiments")):
        errs.append("missing required_experiments (how to test)")

    return errs


def validate_mm_review(path: Path) -> List[str]:
    errs: List[str] = []
    doc = _load_json(path)

    items = doc.get("items", [])
    if not isinstance(items, list) or len(items) == 0:
        return ["items[] must be a non-empty list"]

    for i, it in enumerate(items, start=1):
        if not isinstance(it, dict):
            errs.append(f"item {i}: must be object")
            continue

        if it.get("page") is None:
            errs.append(f"item {i}: missing page")
        if _is_empty(it.get("verdict")):
            errs.append(f"item {i}: missing verdict")
        if _is_empty(it.get("rationale")):
            errs.append(f"item {i}: missing rationale")

        # At least one modality field should be present
        if _is_empty(it.get("vlm_caption")) and _is_empty(it.get("tables_md")) and _is_empty(it.get("equations_md")):
            errs.append(f"item {i}: provide at least one of vlm_caption/tables_md/equations_md")

        v = str(it.get("verdict", "")).strip()
        if v not in {"accepted", "needs_fix", "rejected"}:
            errs.append(f"item {i}: invalid verdict '{v}'")

    return errs


def validate_temporal_correction(path: Path) -> List[str]:
    errs: List[str] = []
    doc = _load_json(path)

    corrections = doc.get("corrections", [])
    if not isinstance(corrections, list) or len(corrections) == 0:
        return ["corrections[] must be a non-empty list"]

    for i, c in enumerate(corrections, start=1):
        if not isinstance(c, dict):
            errs.append(f"correction {i}: must be object")
            continue
        if _is_empty(c.get("assertion_id")):
            errs.append(f"correction {i}: missing assertion_id")
        if _is_empty(c.get("rationale")):
            errs.append(f"correction {i}: missing rationale")

        ct = c.get("corrected_time")
        if not isinstance(ct, dict) or _is_empty(ct):
            errs.append(f"correction {i}: missing corrected_time object")
        else:
            if _is_empty(ct.get("start")):
                errs.append(f"correction {i}: corrected_time.start required")
            if _is_empty(ct.get("granularity")):
                errs.append(f"correction {i}: corrected_time.granularity required")

    return errs


def _collect_files() -> List[Tuple[str, Path]]:
    files: List[Tuple[str, Path]] = []
    for p in sorted(TRAJ_DIR.glob("**/*.y*ml")):
        files.append(("trajectory", p))
    for p in sorted(GRAPH_REV_DIR.glob("**/*.json")):
        files.append(("graph_review", p))
    for p in sorted(HYP_REV_DIR.glob("**/*.json")):
        files.append(("hypothesis_review", p))
    for p in sorted(MM_REV_DIR.glob("**/*.json")):
        files.append(("mm_review", p))
    for p in sorted(TEMP_CORR_DIR.glob("**/*.json")):
        files.append(("temporal_correction", p))
    return files


def main() -> int:
    problems: List[str] = []

    for kind, path in _collect_files():
        try:
            if kind == "trajectory":
                errs = validate_trajectory(path)
            elif kind == "graph_review":
                errs = validate_graph_review(path)
            elif kind == "hypothesis_review":
                errs = validate_hypothesis_review(path)
            elif kind == "mm_review":
                errs = validate_mm_review(path)
            else:
                errs = validate_temporal_correction(path)
        except Exception as e:
            problems.append(f"{kind} {path}: failed to parse ({e})")
            continue

        for e in errs:
            problems.append(f"{kind} {path}: {e}")

    if problems:
        print("❌ Expert artifacts validation failed:\n")
        for p in problems:
            print(" - " + p)
        print(f"\nTotal problems: {len(problems)}")
        return 1

    print("✅ Expert artifacts validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
