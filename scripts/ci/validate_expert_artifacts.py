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
import re
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


TIME_TOKEN_RE = re.compile(r"^(?:\d{4}|\d{4}-\d{2}|\d{4}-\d{2}-\d{2}|unknown|\+inf|-inf)$")


def _is_time_token(v: Any) -> bool:
    if v is None:
        return False
    return bool(TIME_TOKEN_RE.fullmatch(str(v).strip()))


def _has_structured_temporal_fields(a: Dict[str, Any]) -> bool:
    return any(not _is_empty(a.get(k)) for k in ("start_date", "end_date", "valid_from", "valid_to"))


def _validate_temporal_fields(a: Dict[str, Any], prefix: str) -> List[str]:
    errs: List[str] = []
    if _has_structured_temporal_fields(a):
        # Evidence interval is the primary temporal axis in graph reviews.
        if _is_empty(a.get("start_date")):
            errs.append(f"{prefix}: missing start_date (use 'unknown' or '-inf' if needed)")
        elif not _is_time_token(a.get("start_date")):
            errs.append(f"{prefix}: invalid start_date '{a.get('start_date')}'")

        if _is_empty(a.get("end_date")):
            errs.append(f"{prefix}: missing end_date (use publication date, 'unknown' or '+inf' if needed)")
        elif not _is_time_token(a.get("end_date")):
            errs.append(f"{prefix}: invalid end_date '{a.get('end_date')}'")

        for k in ("valid_from", "valid_to"):
            if not _is_empty(a.get(k)) and not _is_time_token(a.get(k)):
                errs.append(f"{prefix}: invalid {k} '{a.get(k)}'")
    return errs


def _resolve_domain_config(domain_value: str) -> Dict[str, Any]:
    """Resolve domain config from configs/domains/*.yaml.

    Supports both:
    - legacy domain id (e.g. "science")
    - Wikidata QID stored in trajectory artifact (e.g. "Q336")
    """
    dv = (domain_value or "").strip()
    if not dv:
        return {}

    qid = dv.upper() if re.fullmatch(r"Q\d+", dv.upper()) else ""

    domain_dir = REPO_ROOT / "configs" / "domains"
    for cfg_path in sorted(domain_dir.glob("*.yaml")):
        try:
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue

        # QID match
        if qid:
            if str(cfg.get("wikidata_qid") or "").strip().upper() == qid:
                return cfg
            continue

        # legacy match
        domain_id = str(cfg.get("domain_id") or "").strip().lower()
        stem = cfg_path.stem.strip().lower()
        dv_norm = dv.strip().lower()
        if dv_norm in {domain_id, stem}:
            return cfg

    return {}


def _required_condition_keys(domain: str) -> List[str]:
    """Domain-specific condition requirements for trajectories.

    For artifact v2/v3 the trajectory may store `domain` as a Wikidata QID (e.g. Q336).
    We map it back to configs/domains/*.yaml via `wikidata_qid`.
    """
    cfg = _resolve_domain_config(domain)
    av = (cfg.get("artifact_validation") or {}) if isinstance(cfg, dict) else {}
    req = av.get("trajectory_required_conditions") or []
    if isinstance(req, list):
        return [str(x) for x in req]
    return []


def _is_qid_like(value: Any) -> bool:
    return bool(re.fullmatch(r"Q\d+", str(value or "").strip(), flags=re.I))


def _is_qid_dict(value: Any) -> bool:
    return isinstance(value, dict) and _is_qid_like(value.get("id"))


def _validate_qid_dict(value: Any, prefix: str) -> List[str]:
    errs: List[str] = []
    if value in (None, "", {}):
        return errs
    if not isinstance(value, dict):
        errs.append(f"{prefix}: must be object with fields id/label")
        return errs
    if not _is_qid_like(value.get("id")):
        errs.append(f"{prefix}: id must be a Wikidata QID")
    return errs


def _validate_step_discovery_context(step: Dict[str, Any], step_idx: int) -> List[str]:
    errs: List[str] = []
    ctx = step.get("discovery_context")
    if ctx in (None, "", {}):
        return errs
    if not isinstance(ctx, dict):
        return [f"step {step_idx}: discovery_context must be an object"]
    geography = ctx.get("geography") or {}
    if geography not in (None, "") and not isinstance(geography, dict):
        errs.append(f"step {step_idx}: discovery_context.geography must be an object")
        geography = {}
    country = geography.get("country") if isinstance(geography, dict) else None
    city = geography.get("city") if isinstance(geography, dict) else None
    errs.extend(_validate_qid_dict(country, f"step {step_idx}: geography.country"))
    errs.extend(_validate_qid_dict(city, f"step {step_idx}: geography.city"))
    if city not in (None, "", {}) and country in (None, "", {}):
        errs.append(f"step {step_idx}: geography.country is required when geography.city is set")
    branches = ctx.get("science_branches") or []
    if branches not in (None, "") and not isinstance(branches, list):
        errs.append(f"step {step_idx}: discovery_context.science_branches must be a list")
        branches = []
    for b_idx, branch in enumerate(branches, start=1):
        errs.extend(_validate_qid_dict(branch, f"step {step_idx}: science_branches[{b_idx}]"))
    simultaneous = ctx.get("simultaneous_discovery")
    if simultaneous not in (None, True, False):
        errs.append(f"step {step_idx}: simultaneous_discovery must be boolean")
    return errs


def _validate_edges_v3(edges: Any, n_steps: int) -> List[str]:
    errs: List[str] = []
    if edges is None or edges == []:
        return errs
    if not isinstance(edges, list):
        return ["edges must be a list"]
    seen: set[tuple[int, int, str, str]] = set()
    for k, e in enumerate(edges, start=1):
        if isinstance(e, (list, tuple)) and len(e) == 2:
            try:
                a_i = int(e[0]); b_i = int(e[1])
            except Exception:
                errs.append(f"edge {k}: from/to must be integers")
                continue
            predicate = "leads_to"
            directionality = "directed"
        elif isinstance(e, dict):
            try:
                a_i = int(e.get("from_step_id"))
                b_i = int(e.get("to_step_id"))
            except Exception:
                errs.append(f"edge {k}: from_step_id/to_step_id must be integers")
                continue
            predicate = str(e.get("predicate") or "").strip() or "leads_to"
            directionality = str(e.get("directionality") or "directed").strip().lower()
            if directionality not in {"directed", "bidirectional", "simultaneous"}:
                errs.append(f"edge {k}: invalid directionality '{directionality}'")
            if e.get("simultaneous_discovery") not in (None, True, False):
                errs.append(f"edge {k}: simultaneous_discovery must be boolean")
        else:
            errs.append(f"edge {k}: must be [from, to] or object with from_step_id/to_step_id")
            continue
        if a_i == b_i:
            errs.append(f"edge {k}: self-loop {a_i}->{b_i} is not allowed")
            continue
        if not (1 <= a_i <= n_steps) or not (1 <= b_i <= n_steps):
            errs.append(f"edge {k}: out of range (steps are 1..{n_steps})")
            continue
        key = (a_i, b_i, predicate, directionality)
        if key in seen:
            errs.append(f"edge {k}: duplicate edge {a_i}->{b_i} ({predicate}, {directionality})")
        else:
            seen.add(key)
    return errs



def validate_trajectory(path: Path) -> List[str]:
    errs: List[str] = []
    doc = _load_yaml(path)

    artifact_version = int(doc.get("artifact_version") or 1)
    domain = str(doc.get("domain") or "").strip()

    if _is_empty(domain):
        errs.append("domain required")

    steps = doc.get("steps", [])
    if not isinstance(steps, list) or len(steps) == 0:
        errs.append("steps[] must be a non-empty list")
        return errs

    n_steps = len(steps)
    required_keys = _required_condition_keys(domain) if domain else []

    for i, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            errs.append(f"step {i}: must be object")
            continue

        if _is_empty(step.get("claim")):
            errs.append(f"step {i}: missing claim")

        # conditions
        conditions = step.get("conditions", {})
        if not isinstance(conditions, dict) or _is_empty(conditions):
            errs.append(f"step {i}: missing conditions")
        else:
            for k in required_keys:
                if _is_empty(conditions.get(k)):
                    errs.append(f"step {i}: conditions.{k} required for domain={domain}")

        # inference / next_question (recommended to be non-empty)
        if _is_empty(step.get("inference")):
            errs.append(f"step {i}: missing inference")
        if _is_empty(step.get("next_question")):
            errs.append(f"step {i}: missing next_question")

        if artifact_version >= 3:
            errs.extend(_validate_step_discovery_context(step, i))

        if artifact_version >= 2:
            # v2: sources[]
            sources = step.get("sources", [])
            if not isinstance(sources, list) or len(sources) == 0:
                errs.append(f"step {i}: missing sources[] (must be non-empty list)")
            else:
                for j, src in enumerate(sources, start=1):
                    if not isinstance(src, dict):
                        errs.append(f"step {i} source {j}: must be object")
                        continue
                    stype = str(src.get("type") or "").strip().lower()
                    if stype == "figure":
                        stype = "image"
                    if stype not in {"text", "image", "table"}:
                        errs.append(f"step {i} source {j}: invalid type='{src.get('type')}' (use text/image/table)")
                    if _is_empty(src.get("source")):
                        errs.append(f"step {i} source {j}: source required")
                    if _is_empty(src.get("snippet_or_summary")):
                        errs.append(f"step {i} source {j}: snippet_or_summary required")
        else:
            # v1 legacy: evidence{}
            evidence = step.get("evidence", {})
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

    # edges (optional): legacy [from, to] pairs or rich v3 edge objects
    edges = doc.get("edges", [])
    errs.extend(_validate_edges_v3(edges, n_steps))

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

        has_legacy_interval = not _is_empty(a.get("time_interval"))
        if has_legacy_interval:
            # Legacy free-form field remains supported for backward compatibility.
            pass
        elif not _has_structured_temporal_fields(a):
            errs.append(
                f"assertion {i}: missing temporal information (provide time_interval or start_date/end_date; use 'unknown' if not stated)"
            )

        errs.extend(_validate_temporal_fields(a, f"assertion {i}"))

        ev = a.get("evidence", {})
        if not isinstance(ev, dict) or _is_empty(ev):
            errs.append(f"assertion {i}: missing evidence")
        else:
            if _is_empty(ev.get("snippet_or_summary")):
                errs.append(f"assertion {i}: evidence.snippet_or_summary required")
            has_locator = not _is_empty(ev.get("page")) or not _is_empty(ev.get("figure_or_table")) or not _is_empty(ev.get("paper_id")) or not _is_empty(ev.get("source"))
            if not has_locator:
                errs.append(f"assertion {i}: evidence should include page, figure_or_table, paper_id, or source")

        verdict = str(a.get("verdict", "")).strip()
        if verdict not in {"accepted", "rejected", "needs_time_fix", "needs_evidence_fix", "added", "uncertain"}:
            errs.append(f"assertion {i}: invalid verdict '{verdict}'")

        if str(a.get("expert_verdict") or a.get("verdict") or "").strip() == "needs_time_fix":
            if _is_empty(a.get("corrected_start_date")) and _is_empty(a.get("corrected_end_date")) and _is_empty(a.get("corrected_valid_from")) and _is_empty(a.get("corrected_valid_to")) and _is_empty(a.get("corrected_time_source")):
                errs.append(f"assertion {i}: needs_time_fix requires at least one corrected temporal field")

        if not _is_empty(a.get("hypothesis_relevance")) and str(a.get("hypothesis_relevance")) not in {"0", "1", "2"}:
            errs.append(f"assertion {i}: hypothesis_relevance must be 0/1/2")
        if not _is_empty(a.get("testability_signal")) and str(a.get("testability_signal")) not in {"0", "1", "2"}:
            errs.append(f"assertion {i}: testability_signal must be 0/1/2")

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
            if _is_empty(ct.get("granularity")):
                errs.append(f"correction {i}: corrected_time.granularity required")
            if _is_empty(ct.get("start")) and _is_empty(ct.get("end")):
                errs.append(f"correction {i}: corrected_time must include start and/or end")
            for key in ("start", "end"):
                if not _is_empty(ct.get(key)) and not _is_time_token(ct.get(key)):
                    errs.append(f"correction {i}: invalid corrected_time.{key} '{ct.get(key)}'")

        ot = c.get("original_time")
        if isinstance(ot, dict) and not _is_empty(ot):
            for key in ("start", "end"):
                if not _is_empty(ot.get(key)) and not _is_time_token(ot.get(key)):
                    errs.append(f"correction {i}: invalid original_time.{key} '{ot.get(key)}'")

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
