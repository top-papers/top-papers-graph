"""Normalize Task 2 expert validation bundles to a single unified format.

Input: a directory containing one or more bundle subdirectories. A bundle is
recognised by any of the following:

* ``edge_reviews.json``                                (legacy finished review)
* ``review_templates/graph_review_prefill.json``       (pre-fill template)
* ``task2_notebook_manifest.json``                     (newer manifest-driven flow)

When ``task2_notebook_manifest.json`` is present it is the source of truth and
declares the canonical file layout for the bundle::

    gold_graph          -> reference_graph.json  (triplets: gold assertions)
    auto_graph_json     -> automatic_graph/temporal_kg.json
    review_state_latest -> expert_validation/drafts/review_state_latest.json
    ...

The manifest-first path lets us recover gold assertions that older bundles
shipped only via ``edge_reviews.json``. Legacy bundles still work via the
``edge_reviews.json`` / prefill fallback.

Output: for every bundle, a folder ``<submission_id>/`` with two files
``gold.json`` and ``auto.json`` (split by ``graph_kind``). Each assertion has:

- canonical ``start_date`` / ``end_date`` (legacy ``valid_from`` / ``valid_to`` /
  ``time_interval`` / ``time_source`` fields are dropped),
- parsed structured ``evidence`` block,
- canonical ``paper_ids`` list,
- cleanly nested ``expert`` sub-object with verdict and all review signals.

Usage::

    scidatapipe-normalize-task2 data/raw/incoming_task2 data/processed/normalized_task2
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Iterable

from scireason.scidatapipe_bridge.vendor.common.utils.io import read_json, validate, write_json
from scireason.scidatapipe_bridge.vendor.common.utils.paper_ids import PaperRef, resolve
from scireason.scidatapipe_bridge.vendor.common.utils.schemas import TASK2_ASSERTION_SCHEMA

logger = logging.getLogger("scidatapipe.normalize_task2")


INF_TOKENS = {"inf", "+inf", "infinity", "+infinity", "-inf", "-infinity", "nan", "none"}

# Interval encodings observed in the wild:
#   "evidence:2010..2010|valid:2010..+inf"
#   "valid:2020..2020"
INTERVAL_RE = re.compile(
    r"(evidence|valid)\s*:\s*([^|]+?)\s*\.\.\s*([^|]+?)(?:\||$)",
    re.IGNORECASE,
)

EXPERT_FIELDS = (
    "verdict",
    "rationale",
    "corrected_start_date",
    "corrected_end_date",
    "corrected_valid_from",
    "corrected_valid_to",
    "corrected_time_source",
    "correction_comment",
    "semantic_correctness",
    "evidence_sufficiency",
    "scope_match",
    "system_match",
    "environment_match",
    "protocol_match",
    "scope_overgeneralized",
    "corrected_scope_note",
    "hypothesis_role",
    "hypothesis_relevance",
    "testability_signal",
    "causal_status",
    "severity",
    "evidence_before_cutoff",
    "leakage_risk",
    "time_type",
    "time_granularity",
    "time_confidence",
    "mm_verdict",
    "mm_rationale",
    "time_source_note",
)



def _source_fingerprint(path: Path) -> str:
    """Stable short fingerprint for collision-safe bundle ids."""
    h = hashlib.sha1()
    if path.is_file():
        try:
            h.update(path.read_bytes())
        except Exception:
            pass
    elif path.is_dir():
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            try:
                rel = child.relative_to(path).as_posix()
            except Exception:
                rel = child.name
            h.update(rel.encode("utf-8", errors="ignore"))
            try:
                h.update(child.read_bytes())
            except Exception:
                h.update(str(child.resolve()).encode("utf-8", errors="ignore"))
    h.update(str(path.resolve()).encode("utf-8", errors="ignore"))
    return h.hexdigest()[:10]


def _source_marker_path(bundle_out_dir: Path) -> Path:
    return bundle_out_dir / ".source_path"


def _same_source(marker: Path, bundle_dir: Path) -> bool:
    try:
        return marker.read_text(encoding="utf-8").strip() == str(bundle_dir.resolve())
    except Exception:
        return False


def _disambiguate_submission_id(submission_id: str, bundle_dir: Path, output_dir: Path) -> str:
    base = submission_id or bundle_dir.name or "unknown_bundle"
    suffix = _source_fingerprint(bundle_dir)
    candidate = f"{base}__input_{suffix}"
    i = 2
    while (output_dir / candidate).exists():
        marker = _source_marker_path(output_dir / candidate)
        if _same_source(marker, bundle_dir):
            return candidate
        candidate = f"{base}__input_{suffix}_{i}"
        i += 1
    return candidate

def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _parse_token(raw: Any) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in INF_TOKENS:
        return None
    if re.fullmatch(r"-?\d+\.\d+", s):
        s = s.split(".", 1)[0]
    return s


def _parse_int_like(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _parse_stringified_container(raw: Any) -> Any:
    """Best-effort conversion of a ``str``-encoded Python literal to a value.

    Falls back to the original string when the blob is not a Python literal.
    """
    if not isinstance(raw, str):
        return raw
    s = raw.strip()
    if not s:
        return s
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return raw


def _parse_evidence(raw: Any) -> dict[str, Any]:
    """Extract structured evidence from the many legacy shapes of ``evidence``.

    Supports:
      - dict (``graph_review_prefill.json`` style),
      - JSON/python-literal string (``edge_reviews.json`` style),
      - empty / missing.
    """
    if isinstance(raw, dict):
        parsed = raw
    elif isinstance(raw, str):
        value = _parse_stringified_container(raw)
        parsed = value if isinstance(value, dict) else {"snippet_or_summary": raw}
    else:
        parsed = {}

    snippet = _clean(parsed.get("snippet_or_summary"))
    page = _parse_int_like(parsed.get("page"))
    figure_or_table = _clean(parsed.get("figure_or_table"))
    paper_id = _clean(parsed.get("paper_id"))
    image_path = _clean(parsed.get("image_path"))

    return {
        "text": snippet,
        "page": page,
        "figure_or_table": figure_or_table,
        "paper_id": paper_id,
        "image_path": image_path,
    }


def _parse_interval(raw: Any) -> tuple[str | None, str | None]:
    """Return ``(start, end)`` using the ``valid`` segment if available."""
    if not isinstance(raw, str) or not raw.strip():
        return None, None
    matches = {m.group(1).lower(): (m.group(2), m.group(3)) for m in INTERVAL_RE.finditer(raw)}
    for kind in ("valid", "evidence"):
        pair = matches.get(kind)
        if pair:
            return _parse_token(pair[0]), _parse_token(pair[1])
    return None, None


def _resolve_paper_ids(raw: Any) -> list[str]:
    if raw in (None, ""):
        return []
    if isinstance(raw, str):
        raw = _parse_stringified_container(raw)
    items: list[str] = []
    if isinstance(raw, (list, tuple)):
        for entry in raw:
            if isinstance(entry, (list, tuple)):
                items.extend(str(x) for x in entry if x)
            elif entry:
                items.append(str(entry))
    elif isinstance(raw, str):
        items.append(raw)
    resolved: list[str] = []
    seen: set[str] = set()
    for item in items:
        ref = resolve(item)
        if ref.id and ref.id not in seen:
            seen.add(ref.id)
            resolved.append(ref.id)
    return resolved


def _derive_dates(record: dict[str, Any]) -> tuple[str | None, str | None]:
    """Priority ladder for canonical ``start_date`` / ``end_date``."""
    corrected_start = _clean(record.get("corrected_start_date")) or _clean(
        record.get("expert_corrected_start_date")
    )
    corrected_end = _clean(record.get("corrected_end_date")) or _clean(
        record.get("expert_corrected_end_date")
    )
    if corrected_start or corrected_end:
        return (
            _parse_token(corrected_start) or None,
            _parse_token(corrected_end) or _parse_token(corrected_start) or None,
        )

    start = _parse_token(record.get("start_date"))
    end = _parse_token(record.get("end_date"))
    if start or end:
        return start, end or start

    valid_from = _parse_token(record.get("valid_from"))
    valid_to = _parse_token(record.get("valid_to"))
    if valid_from or valid_to:
        return valid_from, valid_to or valid_from

    interval_start, interval_end = _parse_interval(record.get("time_interval"))
    if interval_start or interval_end:
        return interval_start, interval_end or interval_start

    return None, None


def _pick_expert_block(record: dict[str, Any]) -> dict[str, Any]:
    """Collect expert signals from flat ``expert_*`` and ``default_review_state``.

    The expert fields live in three places historically:
      - flat on the record with ``expert_`` prefix,
      - flat on the record without prefix (``verdict``/``rationale``/etc.),
      - nested under ``default_review_state`` (newer bundles).
    """
    expert: dict[str, Any] = {}
    default_state = record.get("default_review_state") if isinstance(record.get("default_review_state"), dict) else {}
    for field in EXPERT_FIELDS:
        value = (
            record.get(f"expert_{field}")
            if record.get(f"expert_{field}") not in (None, "", False)
            else record.get(field)
        )
        if value in (None, "") and default_state:
            value = default_state.get(field)
        if value is None:
            value = ""
        if isinstance(value, bool):
            expert[field] = value
        else:
            expert[field] = _clean(value) if not isinstance(value, (list, dict)) else value
    return expert


def _apply_temporal_corrections(
    assertions: list[dict[str, Any]],
    corrections_raw: dict[str, Any] | None,
) -> None:
    """Merge ``temporal_corrections.json`` entries into matching assertions."""
    if not corrections_raw:
        return
    correction_list = corrections_raw.get("corrections") if isinstance(corrections_raw, dict) else None
    if not isinstance(correction_list, list):
        return
    by_uid: dict[str, dict[str, Any]] = {}
    by_assertion: dict[str, dict[str, Any]] = {}
    for item in correction_list:
        if not isinstance(item, dict):
            continue
        uid = _clean(item.get("edge_uid"))
        aid = _clean(item.get("assertion_id"))
        if uid:
            by_uid[uid] = item
        if aid:
            by_assertion[aid] = item

    for assertion in assertions:
        uid = _clean(assertion.get("_edge_uid"))
        aid = assertion["assertion_id"]
        patch = by_uid.get(uid) or by_assertion.get(aid)
        if not patch:
            continue
        expert = assertion["expert"]
        # Non-empty corrections take precedence over the ladder already computed.
        for key in (
            "corrected_start_date",
            "corrected_end_date",
            "corrected_valid_from",
            "corrected_valid_to",
            "corrected_time_source",
        ):
            new_value = _clean(patch.get(key))
            if new_value:
                expert[key] = new_value
        comment = _clean(patch.get("comment") or patch.get("rationale"))
        if comment and not expert.get("correction_comment"):
            expert["correction_comment"] = comment
        # Re-derive dates if corrections changed.
        merged_record = {
            "corrected_start_date": expert.get("corrected_start_date"),
            "corrected_end_date": expert.get("corrected_end_date"),
            "start_date": assertion["start_date"],
            "end_date": assertion["end_date"],
        }
        assertion["start_date"], assertion["end_date"] = _derive_dates(merged_record)


def _normalise_assertion(raw: dict[str, Any]) -> dict[str, Any]:
    graph_kind = _clean(raw.get("graph_kind")).lower()
    if graph_kind not in {"gold", "auto"}:
        graph_kind = "gold" if _clean(raw.get("assertion_id")).startswith("manual") else "auto"

    start, end = _derive_dates(raw)
    evidence = _parse_evidence(
        raw.get("evidence")
        or raw.get("evidence_payload_full")
        or raw.get("evidence_text")
    )

    # ``paper_ids`` in Matvey-style bundles may be a list containing a single
    # stringified list. Flatten defensively.
    paper_ids = _resolve_paper_ids(
        raw.get("paper_ids")
        or raw.get("papers_text")
        or raw.get("papers")
    )
    if evidence["paper_id"] and not paper_ids:
        ref = resolve(evidence["paper_id"])
        if ref.id:
            paper_ids.append(ref.id)

    try:
        importance = float(raw.get("importance_score"))
    except (TypeError, ValueError):
        importance = None

    assertion = {
        "assertion_id": _clean(raw.get("assertion_id")),
        "graph_kind": graph_kind,
        "subject": _clean(raw.get("subject")),
        "predicate": _clean(raw.get("predicate")),
        "object": _clean(raw.get("object")),
        "start_date": start,
        "end_date": end,
        "evidence": evidence,
        "paper_ids": paper_ids,
        "importance_score": importance,
        "expert": _pick_expert_block(raw),
        # Preserved for downstream correction merging; stripped before output.
        "_edge_uid": _clean(raw.get("edge_uid")),
    }
    return assertion


def _submission_id(edge_reviews: dict[str, Any], bundle_dir: Path) -> str:
    candidates = [
        edge_reviews.get("trajectory_submission_id"),
        edge_reviews.get("submission_id"),
    ]
    for candidate in candidates:
        cleaned = _clean(candidate)
        if cleaned:
            return cleaned
    # Fall back to the manifest's ``bundle_dir`` tail when available.
    manifest = _load_manifest(bundle_dir)
    if manifest:
        tail = _clean(manifest.get("bundle_dir")).rstrip("/").rsplit("/", 1)[-1]
        if tail:
            return tail
    return bundle_dir.name


def _find_review_file(bundle_dir: Path) -> Path | None:
    """Find the most authoritative review file in a bundle.

    Preference order:
      1. ``edge_reviews.json`` — expert-completed review.
      2. ``review_templates/graph_review_prefill.json`` — pre-fill template
         used by experts; contains all assertions but typically empty
         verdicts.
    """
    for candidate in (
        bundle_dir / "edge_reviews.json",
        bundle_dir / "review_templates" / "graph_review_prefill.json",
    ):
        if candidate.exists():
            return candidate
    return None


def _load_manifest(bundle_dir: Path) -> dict | None:
    """Return ``task2_notebook_manifest.json`` content if present."""
    path = bundle_dir / "task2_notebook_manifest.json"
    if not path.exists():
        return None
    try:
        data = read_json(path)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _resolve_manifest_path(bundle_dir: Path, value: str | None) -> Path | None:
    """Resolve a path declared in the manifest relative to ``bundle_dir``.

    Manifest entries look like::

        "runs/task2_validation/<submission>/reference_graph.json"

    i.e. they are rooted at the producing repo, not at the bundle that the
    expert actually shipped. We therefore try progressively shorter suffixes
    of the declared path against ``bundle_dir`` until something exists.
    """
    if not value:
        return None
    candidate = bundle_dir / value
    if candidate.exists():
        return candidate
    parts = Path(value).parts
    for i in range(len(parts)):
        candidate = bundle_dir.joinpath(*parts[i:])
        if candidate.exists():
            return candidate
    return None


def _load_gold_from_manifest(
    bundle_dir: Path, manifest: dict | None
) -> list[dict[str, Any]]:
    """Return gold assertions from ``reference_graph.json`` declared in manifest."""
    if not manifest:
        return []
    path = _resolve_manifest_path(bundle_dir, manifest.get("gold_graph"))
    if path is None:
        return []
    try:
        data = read_json(path)
    except (OSError, ValueError):
        return []
    if not isinstance(data, dict):
        return []
    triplets = data.get("triplets") or []
    result: list[dict[str, Any]] = []
    for raw in triplets:
        if not isinstance(raw, dict):
            continue
        # Reference graph triplets are always the gold canonical set.
        raw = {**raw, "graph_kind": "gold"}
        result.append(_normalise_assertion(raw))
    return result


def _load_review_state_latest(
    bundle_dir: Path, manifest: dict | None
) -> dict[str, dict[str, Any]]:
    """Return ``assertion_id -> review-state patch`` from the latest draft."""
    if not manifest:
        return {}
    path = _resolve_manifest_path(bundle_dir, manifest.get("review_state_latest"))
    if path is None:
        return {}
    try:
        data = read_json(path)
    except (OSError, ValueError):
        return {}
    # Structure isn't fully stabilised in the wild; accept several shapes.
    out: dict[str, dict[str, Any]] = {}
    if isinstance(data, dict):
        candidates = (
            data.get("assertions")
            or data.get("review_state")
            or data.get("items")
            or data.get("edges")
            or []
        )
    else:
        candidates = []
    if isinstance(candidates, list):
        for item in candidates:
            if not isinstance(item, dict):
                continue
            aid = _clean(item.get("assertion_id") or item.get("edge_uid"))
            if aid:
                out[aid] = item
    elif isinstance(candidates, dict):
        for aid, item in candidates.items():
            if isinstance(item, dict):
                out[str(aid)] = item
    return out


def _apply_review_state(
    assertions: list[dict[str, Any]],
    state_map: dict[str, dict[str, Any]],
) -> None:
    """Overlay verdict + corrections from ``review_state_latest`` onto assertions."""
    if not state_map:
        return
    for assertion in assertions:
        patch = state_map.get(assertion["assertion_id"])
        if not patch:
            continue
        expert = assertion["expert"]
        for field in EXPERT_FIELDS:
            value = patch.get(field)
            if value in (None, ""):
                value = patch.get(f"expert_{field}")
            if value in (None, ""):
                continue
            if isinstance(value, (bool, list, dict)):
                expert[field] = value
            else:
                expert[field] = _clean(value)
        merged = {
            "corrected_start_date": expert.get("corrected_start_date"),
            "corrected_end_date": expert.get("corrected_end_date"),
            "start_date": assertion["start_date"],
            "end_date": assertion["end_date"],
        }
        assertion["start_date"], assertion["end_date"] = _derive_dates(merged)


def _merge_gold_into(
    assertions: list[dict[str, Any]], gold_from_manifest: list[dict[str, Any]]
) -> tuple[int, int]:
    """Merge manifest gold triplets into ``assertions`` in-place.

    Returns ``(added, overwritten)``. Manifest is authoritative: existing
    entries with the same ``assertion_id`` are replaced.
    """
    if not gold_from_manifest:
        return 0, 0
    by_id = {a["assertion_id"]: a for a in assertions}
    added = overwritten = 0
    for gold in gold_from_manifest:
        aid = gold["assertion_id"]
        if not aid:
            continue
        if aid in by_id:
            by_id[aid].update(
                {k: v for k, v in gold.items() if k not in {"_edge_uid"}}
            )
            overwritten += 1
        else:
            assertions.append(gold)
            by_id[aid] = gold
            added += 1
    return added, overwritten


def _has_bundle_marker(bundle_dir: Path) -> bool:
    return (
        _find_review_file(bundle_dir) is not None
        or (bundle_dir / "task2_notebook_manifest.json").exists()
    )


def normalize_bundle(
    bundle_dir: Path,
    output_dir: Path,
    *,
    force: bool = False,
) -> tuple[Path, Path, bool] | None:
    """Normalize one bundle. Returns ``(gold_path, auto_path, skipped)``.

    When ``force`` is False and both output files already exist we skip
    re-parsing, so the pipeline is idempotent across incremental runs.
    """
    manifest = _load_manifest(bundle_dir)
    edge_reviews_path = _find_review_file(bundle_dir)

    edge_reviews: dict[str, Any] = {}
    raw_assertions: list[Any] = []
    if edge_reviews_path is not None:
        loaded = read_json(edge_reviews_path)
        if isinstance(loaded, dict):
            edge_reviews = loaded
            raw = loaded.get("assertions")
            if isinstance(raw, list):
                raw_assertions = raw

    if not raw_assertions and not manifest:
        logger.warning(
            "skip %s: no review file and no manifest", bundle_dir.name
        )
        return None

    submission_id = _submission_id(edge_reviews, bundle_dir)
    original_submission_id = submission_id

    existing_dir = output_dir / submission_id
    existing_gold = existing_dir / "gold.json"
    existing_auto = existing_dir / "auto.json"
    existing_marker = _source_marker_path(existing_dir)
    if not force and existing_gold.exists() and existing_auto.exists():
        if _same_source(existing_marker, bundle_dir):
            return existing_gold, existing_auto, True
        submission_id = _disambiguate_submission_id(submission_id, bundle_dir, output_dir)
        logger.warning(
            "submission_id collision for bundle '%s' from %s; writing as '%s'",
            original_submission_id,
            bundle_dir,
            submission_id,
        )

    assertions = [
        _normalise_assertion(raw)
        for raw in raw_assertions
        if isinstance(raw, dict)
    ]

    gold_from_manifest = _load_gold_from_manifest(bundle_dir, manifest)
    added, overwritten = _merge_gold_into(assertions, gold_from_manifest)
    if added or overwritten:
        logger.info(
            "%s: manifest gold merge +%d / overwrote %d",
            bundle_dir.name,
            added,
            overwritten,
        )

    review_state = _load_review_state_latest(bundle_dir, manifest)
    if review_state:
        _apply_review_state(assertions, review_state)
        logger.info(
            "%s: overlaid %d review_state_latest entries",
            bundle_dir.name,
            len(review_state),
        )

    corrections_path = bundle_dir / "temporal_corrections.json"
    corrections = read_json(corrections_path) if corrections_path.exists() else None
    _apply_temporal_corrections(assertions, corrections)

    # Drop private fields before writing to disk.
    for assertion in assertions:
        assertion.pop("_edge_uid", None)

    gold = [a for a in assertions if a["graph_kind"] == "gold"]
    auto = [a for a in assertions if a["graph_kind"] == "auto"]

    base = {
        "submission_id": submission_id,
        "original_submission_id": original_submission_id if original_submission_id != submission_id else "",
        "trajectory_submission_id": _clean(edge_reviews.get("trajectory_submission_id")),
        "domain": _clean(edge_reviews.get("domain")),
        "topic": _clean(edge_reviews.get("topic")),
        "cutoff_year": _parse_int_like(edge_reviews.get("cutoff_year")),
        "reviewer_id": _clean(edge_reviews.get("reviewer_id")),
        "timestamp": _clean(edge_reviews.get("timestamp")),
    }

    out_dir = output_dir / submission_id
    gold_path = out_dir / "gold.json"
    auto_path = out_dir / "auto.json"
    write_json(gold_path, {**base, "assertions": gold})
    write_json(auto_path, {**base, "assertions": auto})
    _source_marker_path(out_dir).write_text(str(bundle_dir.resolve()), encoding="utf-8")

    for path in (gold_path, auto_path):
        for err in validate(read_json(path), TASK2_ASSERTION_SCHEMA):
            logger.warning("%s schema issue: %s", path.name, err)

    logger.info(
        "%s: gold=%d auto=%d -> %s",
        bundle_dir.name,
        len(gold),
        len(auto),
        out_dir,
    )
    return gold_path, auto_path, False


def _iter_bundles(root: Path) -> Iterable[Path]:
    if _has_bundle_marker(root):
        yield root
        return
    for child in sorted(root.iterdir()):
        if child.is_dir() and _has_bundle_marker(child):
            yield child


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize Task 2 expert bundles")
    parser.add_argument("input_dir", type=Path, help="Directory containing bundle subfolders")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-normalize bundles whose output folder already exists",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.input_dir.exists():
        logger.error("input directory missing: %s", args.input_dir)
        return 1

    created = skipped = 0
    for bundle in _iter_bundles(args.input_dir):
        result = normalize_bundle(bundle, args.output_dir, force=args.force)
        if result is None:
            continue
        _, _, was_skipped = result
        if was_skipped:
            logger.info("skip %s (already normalized)", bundle.name)
            skipped += 1
        else:
            created += 1
    logger.info("normalized %d (skipped %d) Task 2 bundles", created, skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
