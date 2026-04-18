"""Normalize legacy Task 1 trajectory YAMLs to the frozen v4 schema.

Input: any YAML with ``artifact_version: 2`` or ``3`` (including the
experimental v3+time-intervals variant).

Output: YAML with ``artifact_version: 4`` where:
- paper identifiers are canonicalised (arxiv / doi / wiki / url),
- steps carry ``start_date`` / ``end_date`` / ``time_source`` explicitly,
- edges are dicts (``[[1,2]]`` tuples are upgraded),
- every step has ``importance`` + ``discovery_context`` + sanitised
  ``conditions`` + per-source ``has_figure_ref`` / ``figure_kind`` /
  ``figure_number`` flags.

Usage::

    python -m scripts.normalize_task1 data/incoming_task1 data/normalized_task1
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import yaml

from scireason.scidatapipe_bridge.vendor.common.utils.io import validate
from scireason.scidatapipe_bridge.vendor.common.utils.paper_ids import (
    PaperRef,
    extract_figure_locator,
    has_figure_ref,
    resolve,
)
from scireason.scidatapipe_bridge.vendor.common.utils.schemas import TASK1_SCHEMA_V4

logger = logging.getLogger("scidatapipe.normalize_task1")

WIKIDATA_CACHE_PATH = Path("data/processed/cache/wikidata.json")
WIKIDATA_USER_AGENT = "SciHistPipeline/1.0 (research)"

INF_TOKENS = {"inf", "+inf", "infinity", "+infinity", "-inf", "-infinity", "nan", "none"}

VALID_IMPORTANCE = {"ключевая", "не ключевая", "фоновая"}


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _coerce_page(value: Any) -> int | None:
    if value in (None, "", "unknown"):
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _coerce_year(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _parse_temporal_token(raw: Any) -> str | None:
    """Parse a single temporal token such as ``"2013"``, ``2013``, ``"+inf"``.

    Returns ``None`` for infinity / NaN markers; strips ``.0`` suffixes from
    numeric strings that originated from ``Infinity`` in source JSON.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in INF_TOKENS:
        return None
    if re.fullmatch(r"-?\d+\.\d+", s):
        s = s.split(".", 1)[0]
    return s


def _load_wikidata_cache() -> dict[str, str]:
    try:
        return json.loads(WIKIDATA_CACHE_PATH.read_text())
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def _save_wikidata_cache(cache: dict[str, str]) -> None:
    WIKIDATA_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    WIKIDATA_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2))


def _lookup_domain_label(qid: str, cache: dict[str, str]) -> str:
    """Look up a human-readable label for a Wikidata QID.

    Uses the on-disk cache first; falls back to the public API if available.
    Returns an empty string on any error (including no network access).
    """
    if not qid or not re.fullmatch(r"Q\d+", qid):
        return ""
    if qid in cache:
        return cache[qid]
    try:
        import requests  # type: ignore
    except ImportError:
        return ""
    try:
        response = requests.get(
            "https://www.wikidata.org/wiki/Special:EntityData/" + qid + ".json",
            headers={"User-Agent": WIKIDATA_USER_AGENT},
            timeout=8,
        )
        response.raise_for_status()
        payload = response.json()
        entity = payload.get("entities", {}).get(qid, {})
        labels = entity.get("labels", {})
        label = (
            labels.get("ru", {}).get("value")
            or labels.get("en", {}).get("value")
            or ""
        )
    except Exception as exc:  # pragma: no cover - network flakiness
        logger.debug("Wikidata lookup failed for %s: %s", qid, exc)
        label = ""
    cache[qid] = label
    return label


def _normalise_paper(raw_paper: dict[str, Any]) -> dict[str, Any]:
    raw_id = _clean(raw_paper.get("id"))
    ref = resolve(raw_id) if raw_id else PaperRef(id="", paper_type="url", raw="")
    return {
        "id": ref.id,
        "paper_type": ref.paper_type,
        "arxiv_id": ref.arxiv_id,
        "version": ref.version,
        "year": _coerce_year(raw_paper.get("year")),
        "title": _clean(raw_paper.get("title")),
        "resolved": True,
        "raw": ref.raw or raw_id,
    }


def _synthetic_paper(ref: PaperRef) -> dict[str, Any]:
    return {
        "id": ref.id,
        "paper_type": ref.paper_type,
        "arxiv_id": ref.arxiv_id,
        "version": ref.version,
        "year": None,
        "title": "",
        "resolved": False,
        "raw": ref.raw,
    }


def _normalise_source(src: dict[str, Any]) -> tuple[dict[str, Any], PaperRef]:
    type_raw = _clean(src.get("type")).lower()
    if type_raw == "figure":
        type_raw = "image"
    if type_raw not in {"text", "image", "table"}:
        type_raw = "text"

    raw_source = _clean(src.get("source"))
    ref = resolve(raw_source) if raw_source else PaperRef(id="", paper_type="url", raw="")
    locator = _clean(src.get("locator"))
    fig = extract_figure_locator(locator)

    normalised = {
        "type": type_raw,
        "source": raw_source,
        "paper_ref_id": ref.id,
        "page": _coerce_page(src.get("page")),
        "locator": locator,
        "snippet_or_summary": _clean(src.get("snippet_or_summary")),
        "has_figure_ref": has_figure_ref(locator) or type_raw in {"image", "table"},
        "figure_kind": fig[0] if fig else ("figure" if type_raw == "image" else ("table" if type_raw == "table" else "")),
        "figure_number": fig[1] if fig else None,
    }
    return normalised, ref


def _normalise_conditions(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        raw = {}
    result = {}
    for key in ("system", "environment", "protocol", "notes"):
        value = _clean(raw.get(key))
        # Legacy YAMLs commonly fill these with the literal string "unknown".
        if value.lower() == "unknown":
            value = ""
        result[key] = value
    return result


def _normalise_discovery_context(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    geography = raw.get("geography") if isinstance(raw.get("geography"), dict) else {}
    country = geography.get("country") if isinstance(geography.get("country"), dict) else {}
    city = geography.get("city") if isinstance(geography.get("city"), dict) else {}
    branches = raw.get("science_branches") if isinstance(raw.get("science_branches"), list) else []
    return {
        "simultaneous_discovery": bool(raw.get("simultaneous_discovery")),
        "geography": {
            "country": {"id": _clean(country.get("id")), "label": _clean(country.get("label"))},
            "city": {"id": _clean(city.get("id")), "label": _clean(city.get("label"))},
        },
        "science_branches": [
            {
                "id": _clean(b.get("id")) if isinstance(b, dict) else "",
                "label": _clean(b.get("label")) if isinstance(b, dict) else "",
            }
            for b in branches
            if isinstance(b, dict)
        ],
    }


def _normalise_edges(raw: Any, step_count: int) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    result: list[dict[str, Any]] = []
    for edge in raw:
        from_id: Any
        to_id: Any
        predicate = "leads_to"
        directionality = "directed"
        direction_label = ""
        simultaneous = False
        if isinstance(edge, dict):
            from_id = edge.get("from_step_id", edge.get("from"))
            to_id = edge.get("to_step_id", edge.get("to"))
            predicate = _clean(edge.get("predicate")) or "leads_to"
            directionality = _clean(edge.get("directionality")) or "directed"
            direction_label = _clean(edge.get("direction_label"))
            simultaneous = bool(edge.get("simultaneous_discovery"))
        elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
            from_id, to_id = edge[0], edge[1]
        else:
            continue
        try:
            from_step_id = int(from_id)
            to_step_id = int(to_id)
        except (TypeError, ValueError):
            continue
        if not (1 <= from_step_id <= step_count and 1 <= to_step_id <= step_count):
            continue
        if directionality not in {"directed", "bidirectional", "simultaneous"}:
            directionality = "directed"
        result.append(
            {
                "from_step_id": from_step_id,
                "to_step_id": to_step_id,
                "predicate": predicate,
                "directionality": directionality,
                "direction_label": direction_label,
                "simultaneous_discovery": simultaneous,
            }
        )
    return result


def _derive_step_dates(
    step: dict[str, Any],
    source_refs: list[PaperRef],
    paper_year_by_id: dict[str, int],
    cutoff_year: int | None,
) -> tuple[str | None, str | None, str]:
    """Fill step-level ``start_date`` / ``end_date`` / ``time_source``.

    Priority ladder:
      1) legacy ``valid_from`` / ``valid_to``
      2) explicit ``start_date`` / ``end_date`` on the legacy step
      3) explicit ``year``
      4) min/max year across the step's source papers
    """
    legacy_from = _parse_temporal_token(step.get("valid_from"))
    legacy_to = _parse_temporal_token(step.get("valid_to"))
    if legacy_from or legacy_to:
        return legacy_from, legacy_to or legacy_from, _clean(step.get("time_source")) or "legacy_valid_from_to"

    explicit_start = _parse_temporal_token(step.get("start_date"))
    explicit_end = _parse_temporal_token(step.get("end_date"))
    if explicit_start or explicit_end:
        return (
            explicit_start,
            explicit_end or explicit_start,
            _clean(step.get("time_source")) or "explicit",
        )

    year = _parse_temporal_token(step.get("year"))
    if year:
        return year, year, _clean(step.get("time_source")) or "step_year"

    years: list[int] = []
    for ref in source_refs:
        year_value = paper_year_by_id.get(ref.id)
        if isinstance(year_value, int):
            years.append(year_value)
    if years:
        return str(min(years)), str(max(years)), "paper_year_fallback"

    if cutoff_year:
        return str(cutoff_year), str(cutoff_year), "cutoff_year_fallback"
    return None, None, "unknown"


def _expert_block(raw_expert: Any) -> dict[str, str]:
    if not isinstance(raw_expert, dict):
        raw_expert = {}
    keys = (
        "last_name",
        "first_name",
        "patronymic",
        "full_name",
        "latin_full_name",
        "latin_slug",
    )
    expert = {k: _clean(raw_expert.get(k)) for k in keys}
    if not expert["full_name"]:
        parts = [expert["last_name"], expert["first_name"], expert["patronymic"]]
        expert["full_name"] = " ".join(p for p in parts if p)
    return expert


def _canonical_submission_id(doc: dict[str, Any], expert: dict[str, str]) -> tuple[str, str]:
    """Reconcile ``submission_id`` and ``artifact_hash`` into a canonical pair.

    Returns ``(submission_id, artifact_hash)``.
    """
    submission_raw = _clean(doc.get("submission_id"))
    artifact_hash = _clean(doc.get("artifact_hash"))
    latin_slug = expert.get("latin_slug") or ""

    if artifact_hash and latin_slug:
        canonical = f"{latin_slug}__{artifact_hash}"
    elif submission_raw:
        canonical = submission_raw
        # Attempt to recover an artifact_hash from the suffix pattern.
        m = re.search(r"__([0-9a-f]{6,})$", submission_raw)
        if m and not artifact_hash:
            artifact_hash = m.group(1)
    else:
        canonical = latin_slug or "unknown_submission"

    if submission_raw and submission_raw != canonical:
        logger.warning(
            "submission_id mismatch: '%s' → canonical '%s'", submission_raw, canonical
        )
    return canonical, artifact_hash


def _normalise_doc(raw: dict[str, Any], wiki_cache: dict[str, str]) -> dict[str, Any]:
    expert = _expert_block(raw.get("expert"))
    submission_id, artifact_hash = _canonical_submission_id(raw, expert)

    papers_raw = raw.get("papers") if isinstance(raw.get("papers"), list) else []
    papers: list[dict[str, Any]] = []
    paper_year_by_id: dict[str, int] = {}
    seen_paper_ids: set[str] = set()
    for paper in papers_raw:
        if not isinstance(paper, dict):
            continue
        normalised = _normalise_paper(paper)
        if normalised["id"] and normalised["id"] not in seen_paper_ids:
            seen_paper_ids.add(normalised["id"])
            papers.append(normalised)
            if isinstance(normalised["year"], int):
                paper_year_by_id[normalised["id"]] = normalised["year"]

    steps_raw = raw.get("steps") if isinstance(raw.get("steps"), list) else []
    steps: list[dict[str, Any]] = []
    cutoff_year = _coerce_year(raw.get("cutoff_year"))
    for index, step in enumerate(steps_raw, start=1):
        if not isinstance(step, dict):
            continue

        sources: list[dict[str, Any]] = []
        source_refs: list[PaperRef] = []

        # Legacy v2 YAMLs sometimes use a single ``evidence`` dict instead of ``sources``.
        src_iter = step.get("sources")
        if not isinstance(src_iter, list) or not src_iter:
            legacy_evidence = step.get("evidence")
            if isinstance(legacy_evidence, dict) and legacy_evidence:
                src_iter = [legacy_evidence]
            else:
                src_iter = []

        for src in src_iter:
            if not isinstance(src, dict):
                continue
            normalised, ref = _normalise_source(src)
            sources.append(normalised)
            if ref.id and ref.id not in seen_paper_ids:
                seen_paper_ids.add(ref.id)
                papers.append(_synthetic_paper(ref))
                logger.warning(
                    "step %d references paper '%s' not listed in papers[]; appended as unresolved",
                    index,
                    ref.raw or ref.id,
                )
            if ref.id:
                source_refs.append(ref)

        if not sources:
            # Schema requires at least one source; insert a placeholder so the
            # validator complains downstream rather than silently dropping the step.
            sources.append(
                {
                    "type": "text",
                    "source": "",
                    "paper_ref_id": "",
                    "page": None,
                    "locator": "",
                    "snippet_or_summary": "",
                    "has_figure_ref": False,
                    "figure_kind": "",
                    "figure_number": None,
                }
            )

        start_date, end_date, time_source = _derive_step_dates(
            step, source_refs, paper_year_by_id, cutoff_year
        )

        importance = _clean(step.get("importance"))
        if importance not in VALID_IMPORTANCE:
            importance = "ключевая"

        steps.append(
            {
                "step_id": int(step.get("step_id") or index),
                "claim": _clean(step.get("claim")),
                "importance": importance,
                "start_date": start_date,
                "end_date": end_date,
                "time_source": time_source,
                "conditions": _normalise_conditions(step.get("conditions")),
                "sources": sources,
                "discovery_context": _normalise_discovery_context(step.get("discovery_context")),
                "inference": _clean(step.get("inference")),
                "next_question": "" if _clean(step.get("next_question")) in {"-", "—", "none", "Вопросов больше нет"} else _clean(step.get("next_question")),
            }
        )

    # Renumber step_ids so edges stay consistent after skipping invalid entries.
    id_remap: dict[int, int] = {}
    for new_idx, step in enumerate(steps, start=1):
        id_remap[step["step_id"]] = new_idx
        step["step_id"] = new_idx

    edges: list[dict[str, Any]] = []
    for edge in _normalise_edges(raw.get("edges"), len(steps)):
        edge["from_step_id"] = id_remap.get(edge["from_step_id"], edge["from_step_id"])
        edge["to_step_id"] = id_remap.get(edge["to_step_id"], edge["to_step_id"])
        if 1 <= edge["from_step_id"] <= len(steps) and 1 <= edge["to_step_id"] <= len(steps):
            edges.append(edge)

    domain = _clean(raw.get("domain"))
    domain_label = _clean(raw.get("domain_label")) or _lookup_domain_label(domain, wiki_cache)

    return {
        "artifact_version": 4,
        "topic": _clean(raw.get("topic")),
        "domain": domain,
        "domain_label": domain_label,
        "cutoff_year": cutoff_year,
        "submission_id": submission_id,
        "artifact_hash": artifact_hash,
        "generated_at": _clean(raw.get("generated_at")),
        "expert": expert,
        "papers": papers,
        "steps": steps,
        "edges": edges,
    }


def _peek_submission_id(raw: dict[str, Any], input_path: Path) -> str:
    """Return submission_id without running the full normalizer.

    Mirrors the logic in ``_normalise_doc`` / ``_submission_id`` so we can
    detect an already-processed submission cheaply and skip it.
    """
    candidates = [
        raw.get("submission_id"),
        raw.get("trajectory_submission_id"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return input_path.stem


def normalize_file(
    input_path: Path,
    output_dir: Path,
    wiki_cache: dict[str, str],
    *,
    force: bool = False,
) -> tuple[Path, bool]:
    """Normalize one YAML. Returns ``(out_path, skipped)``.

    When ``force`` is False and the destination YAML already exists, we
    skip re-normalization so the pipeline can be re-run cheaply after new
    raw files arrive.
    """
    raw = yaml.safe_load(input_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{input_path}: top-level YAML must be a mapping")

    submission_id_hint = _peek_submission_id(raw, input_path)
    existing = output_dir / submission_id_hint / f"{submission_id_hint}.yaml"
    if not force and existing.exists():
        return existing, True

    doc = _normalise_doc(raw, wiki_cache)
    errors = validate(doc, TASK1_SCHEMA_V4)
    for err in errors:
        logger.warning("%s schema issue: %s", input_path.name, err)

    submission_dir = output_dir / doc["submission_id"]
    submission_dir.mkdir(parents=True, exist_ok=True)
    out_path = submission_dir / f"{doc['submission_id']}.yaml"
    out_path.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False))
    return out_path, False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize legacy Task 1 YAMLs to v4")
    parser.add_argument("input_dir", type=Path, help="Directory with raw YAML files")
    parser.add_argument("output_dir", type=Path, help="Destination directory")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-normalize submissions whose output folder already exists",
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

    wiki_cache = _load_wikidata_cache()
    created = skipped = 0
    for yaml_path in sorted(args.input_dir.rglob("*.yaml")):
        try:
            out, was_skipped = normalize_file(
                yaml_path, args.output_dir, wiki_cache, force=args.force
            )
        except Exception as exc:  # pragma: no cover - surfaced to user
            logger.error("failed to normalize %s: %s", yaml_path, exc)
            continue
        if was_skipped:
            logger.info("skip %s (already present at %s)", yaml_path.name, out)
            skipped += 1
        else:
            logger.info("%s -> %s", yaml_path.name, out.name)
            created += 1
    _save_wikidata_cache(wiki_cache)
    logger.info("normalized %d (skipped %d) Task 1 YAML files", created, skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
