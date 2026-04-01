from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse, unquote

import yaml  # type: ignore
from rich.console import Console

from ..domain import DomainConfig, load_domain_config
from ..ingest.acquire import AcquireResult, acquire_pdf
from ..ingest.mm_pipeline import ingest_pdf_multimodal_auto
from ..config import settings
from ..ingest.pipeline import ingest_pdf_auto
from ..llm import temporary_llm_selection
from ..mm.vlm import temporary_vlm_selection
from ..papers.schema import ExternalIds, PaperMetadata, PaperSource
from ..papers.service import get_paper_by_doi, search_papers
from ..temporal.temporal_kg_builder import PaperRecord, TemporalKnowledgeGraph, build_temporal_kg, load_papers_from_processed
from ..task2_filters import entity_matches_exclusion, normalize_exclusion_spec, serialize_exclusion_spec, topic_profile_from_doc, score_triplet_importance


console = Console()


DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
ARXIV_RE = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/|arxiv:)?([a-z\-]+/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE)
OPENALEX_RE = re.compile(r"(?:openalex\.org/)?(W\d+)", re.IGNORECASE)
PMID_RE = re.compile(r"(?:pubmed(?:\.ncbi\.nlm\.nih\.gov)?/)?(\d{5,10})", re.IGNORECASE)
PMCID_RE = re.compile(r"(PMC\d+)", re.IGNORECASE)
DATE_TOKEN_RE = re.compile(r"^(?:\d{4}|\d{4}-\d{2}|\d{4}-\d{2}-\d{2}|unknown|\+inf|-inf)$")


def _emit_progress(
    progress_callback: Optional[Callable[[Dict[str, Any]], None]],
    *,
    stage: str,
    current: int,
    total: int,
    message: str,
    **extra: Any,
) -> None:
    payload = {
        "stage": stage,
        "current": current,
        "total": total,
        "message": message,
        "percent": 0 if total <= 0 else int(round((current / total) * 100)),
    }
    payload.update(extra)
    console.print(f"[blue][Task2 {current}/{total}][/blue] {message}")
    if progress_callback is not None:
        progress_callback(payload)


DEFAULT_SEARCH_SOURCES = [
    PaperSource.semantic_scholar,
    PaperSource.openalex,
    PaperSource.crossref,
    PaperSource.pubmed,
    PaperSource.europe_pmc,
    PaperSource.arxiv,
    PaperSource.biorxiv,
]


def acquire_pdfs(resolved: Sequence[PaperMetadata], *, raw_dir: Path, meta_dir: Path) -> List[AcquireResult]:
    """Acquire PDFs for a resolved paper list.

    Kept as a small wrapper so tests and notebook code can monkeypatch the whole
    acquisition step without reaching into per-paper internals.
    """
    return [acquire_pdf(meta, raw_dir=raw_dir, meta_dir=meta_dir) for meta in resolved]


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9\-\s_]+", "", (text or "").strip().lower())
    s = re.sub(r"\s+", "-", s).strip("-")
    return s[:80] or "task2"


def _utc_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _norm_ws(text: str) -> str:
    return " ".join((text or "").split())


def _norm_title(text: str) -> str:
    s = _norm_ws(text).lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]+", "", s, flags=re.UNICODE)
    return s.strip()


def _looks_like_url(text: str) -> bool:
    try:
        u = urlparse((text or "").strip())
    except Exception:
        return False
    return bool(u.scheme and u.netloc)


def _normalized_url_match_key(text: str, *, strip_pdf_suffix: bool = False) -> str:
    raw = str(text or "").strip()
    if not raw or not _looks_like_url(raw):
        return ""
    try:
        u = urlparse(raw)
    except Exception:
        return ""
    path = unquote(u.path or "").strip().rstrip("/")
    if strip_pdf_suffix and path.lower().endswith(".pdf"):
        path = path[:-4]
    return f"{u.scheme.lower()}://{u.netloc.lower()}{path}"


def _entry_identity_keys(entry: Dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    raw_id = str(entry.get("id") or "").strip()
    title = str(entry.get("title") or "").strip()

    doi = _extract_doi(raw_id)
    if doi:
        keys.add(f"doi:{doi.lower()}")

    arxiv = _extract_arxiv(raw_id)
    if arxiv:
        keys.add(f"arxiv:{arxiv.lower()}")

    pmcid = _extract_pmcid(raw_id)
    if pmcid:
        keys.add(f"pmcid:{pmcid.lower()}")

    pmid = _extract_pmid(raw_id) if not _looks_like_url(raw_id) else None
    if pmid:
        keys.add(f"pmid:{pmid}")

    if _looks_like_url(raw_id):
        keys.add(_normalized_url_match_key(raw_id, strip_pdf_suffix=False))
        keys.add(_normalized_url_match_key(raw_id, strip_pdf_suffix=True))

    if title:
        keys.add(f"title:{_norm_title(title)}")

    return {k for k in keys if k}


def _primary_entry_identity_key(entry: Dict[str, Any]) -> str:
    raw_id = str(entry.get("id") or "").strip()
    title = str(entry.get("title") or "").strip()

    doi = _extract_doi(raw_id)
    if doi:
        return f"doi:{doi.lower()}"

    arxiv = _extract_arxiv(raw_id)
    if arxiv:
        return f"arxiv:{arxiv.lower()}"

    openalex = _extract_openalex(raw_id)
    if openalex:
        return f"openalex:{openalex.lower()}"

    pmcid = _extract_pmcid(raw_id)
    if pmcid:
        return f"pmcid:{pmcid.lower()}"

    pmid = _extract_pmid(raw_id) if not _looks_like_url(raw_id) else None
    if pmid:
        return f"pmid:{pmid}"

    if _looks_like_url(raw_id):
        key = _normalized_url_match_key(raw_id, strip_pdf_suffix=True)
        if key:
            return key

    if title:
        return f"title:{_norm_title(title)}"

    return f"raw:{raw_id}"


def _source_entry_to_dict(src: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    source_ref = str(src.get("source") or "").strip()
    if not source_ref:
        return None

    title = str(src.get("title") or "").strip()
    year = src.get("year")
    try:
        year = int(year) if year not in (None, "") else None
    except Exception:
        year = None

    if not (
        _looks_like_url(source_ref)
        or _extract_doi(source_ref)
        or _extract_arxiv(source_ref)
        or _extract_openalex(source_ref)
        or _extract_pmid(source_ref)
        or _extract_pmcid(source_ref)
    ):
        return None

    return {
        "id": source_ref,
        "title": title,
        "year": year,
        "_derived_from_step_source": True,
    }


def _iter_trajectory_entries(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def add(entry: Dict[str, Any]) -> None:
        if not isinstance(entry, dict):
            return
        key = _primary_entry_identity_key(entry)
        if key in seen:
            return
        seen.add(key)
        entries.append(entry)

    for entry in doc.get("papers", []) or []:
        add(dict(entry))

    for step in doc.get("steps", []) or []:
        if not isinstance(step, dict):
            continue
        for src in step.get("sources", []) or []:
            if not isinstance(src, dict):
                continue
            synthesized = _source_entry_to_dict(src)
            if synthesized is not None:
                add(synthesized)

    return entries


def _trajectory_acquire_hints_for_entry(doc: Dict[str, Any], entry: Dict[str, Any]) -> List[str]:
    entry_keys = _entry_identity_keys(entry)
    if not entry_keys:
        return []

    hints: List[str] = []
    for step in doc.get("steps", []) or []:
        if not isinstance(step, dict):
            continue
        for src in step.get("sources", []) or []:
            if not isinstance(src, dict):
                continue
            source_ref = str(src.get("source") or "").strip()
            if not source_ref:
                continue

            src_entry = _source_entry_to_dict(src) or {"id": source_ref, "title": str(src.get("title") or "").strip()}
            source_keys = _entry_identity_keys(src_entry)
            if entry_keys & source_keys and source_ref not in hints:
                hints.append(source_ref)

    return hints


def _trajectory_pdf_hints_for_entry(doc: Dict[str, Any], entry: Dict[str, Any]) -> List[str]:
    return [hint for hint in _trajectory_acquire_hints_for_entry(doc, entry) if str(hint).lower().endswith(".pdf")]


def _extract_doi(text: str) -> Optional[str]:
    s = unquote((text or "").strip())
    m = DOI_RE.search(s)
    if not m:
        return None
    doi = m.group(0).rstrip(").,;]")
    return doi


def _extract_arxiv(text: str) -> Optional[str]:
    s = unquote((text or "").strip())
    m = ARXIV_RE.search(s)
    if not m:
        return None
    return m.group(1)


def _extract_openalex(text: str) -> Optional[str]:
    s = unquote((text or "").strip())
    m = OPENALEX_RE.search(s)
    if not m:
        return None
    return m.group(1).upper()


def _extract_pmid(text: str) -> Optional[str]:
    s = unquote((text or "").strip())
    m = PMID_RE.search(s)
    if not m:
        return None
    return m.group(1)


def _extract_pmcid(text: str) -> Optional[str]:
    s = unquote((text or "").strip())
    m = PMCID_RE.search(s)
    if not m:
        return None
    return m.group(1).upper()


def _paper_year(meta: PaperMetadata) -> Optional[int]:
    if meta.year is not None:
        try:
            return int(meta.year)
        except Exception:
            return None
    if meta.published_date is not None:
        try:
            return int(meta.published_date.year)
        except Exception:
            return None
    return None


def _score_title_match(a: str, b: str) -> float:
    na = _norm_title(a)
    nb = _norm_title(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


def _pick_best_candidate(entry: Dict[str, Any], candidates: Sequence[PaperMetadata]) -> Optional[PaperMetadata]:
    title = str(entry.get("title") or "")
    year_raw = entry.get("year")
    try:
        entry_year = int(year_raw) if year_raw not in (None, "") else None
    except Exception:
        entry_year = None
    raw_id = str(entry.get("id") or "")
    doi_hint = _extract_doi(raw_id)
    arxiv_hint = _extract_arxiv(raw_id)
    openalex_hint = _extract_openalex(raw_id)

    scored: List[Tuple[float, PaperMetadata]] = []
    for cand in candidates:
        score = 0.0
        if title:
            score += 4.0 * _score_title_match(title, cand.title)
        cy = _paper_year(cand)
        if entry_year is not None and cy is not None:
            score += max(0.0, 1.0 - 0.25 * abs(entry_year - cy))
        if doi_hint and cand.ids and cand.ids.doi and cand.ids.doi.lower() == doi_hint.lower():
            score += 5.0
        if arxiv_hint and cand.ids and cand.ids.arxiv and cand.ids.arxiv.lower() == arxiv_hint.lower():
            score += 4.0
        if openalex_hint and cand.ids and cand.ids.openalex and cand.ids.openalex.upper() == openalex_hint.upper():
            score += 4.0
        if raw_id and cand.id.lower() == raw_id.lower():
            score += 5.0
        if cand.pdf_url:
            score += 0.25
        if cand.abstract:
            score += 0.1
        scored.append((score, cand))

    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best = scored[0]
    if best_score < 1.75:
        return None
    return best


def _fallback_paper(entry: Dict[str, Any]) -> PaperMetadata:
    raw_id = str(entry.get("id") or "").strip()
    title = str(entry.get("title") or raw_id or "Untitled paper").strip()
    try:
        year = int(entry.get("year")) if entry.get("year") not in (None, "") else None
    except Exception:
        year = None

    canonical_id = raw_id
    if not canonical_id:
        canonical_id = f"manual:{_slugify(title)}"
    elif _extract_doi(canonical_id):
        canonical_id = f"doi:{_extract_doi(canonical_id)}"
    elif _extract_arxiv(canonical_id):
        canonical_id = f"arxiv:{_extract_arxiv(canonical_id)}"
    elif _extract_openalex(canonical_id):
        canonical_id = f"openalex:{_extract_openalex(canonical_id)}"
    elif _extract_pmcid(canonical_id):
        canonical_id = f"pmc:{_extract_pmcid(canonical_id)}"
    elif _extract_pmid(canonical_id) and not _looks_like_url(canonical_id):
        canonical_id = f"pmid:{_extract_pmid(canonical_id)}"
    elif _looks_like_url(canonical_id):
        canonical_id = canonical_id
    else:
        canonical_id = f"manual:{_slugify(canonical_id)}"

    doi = _extract_doi(raw_id)
    pmid = _extract_pmid(raw_id) if not _looks_like_url(raw_id) else None
    pmcid = _extract_pmcid(raw_id)
    ids = ExternalIds(
        doi=doi,
        arxiv=_extract_arxiv(raw_id),
        openalex=_extract_openalex(raw_id),
        pmid=pmid,
        pmcid=pmcid,
    )

    fallback_url: Optional[str] = None
    if _looks_like_url(raw_id):
        fallback_url = raw_id
    elif doi:
        fallback_url = f"https://doi.org/{doi}"
    elif pmcid:
        fallback_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
    elif pmid:
        fallback_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

    return PaperMetadata(
        id=canonical_id,
        source=PaperSource.unknown,
        title=title,
        abstract=None,
        year=year,
        url=fallback_url,
        pdf_url=None,
        ids=ids,
    )


def _resolve_entry_by_exact_identifier(entry: Dict[str, Any], *, enable_remote_lookup: bool = True) -> Optional[PaperMetadata]:
    raw_id = str(entry.get("id") or "").strip()
    if not raw_id:
        return None
    if not enable_remote_lookup:
        return None

    doi = _extract_doi(raw_id)
    if doi:
        try:
            meta = get_paper_by_doi(doi)
            if meta is not None:
                return meta
        except Exception:
            pass

    title = str(entry.get("title") or "").strip()
    try:
        candidates = search_papers(title or raw_id, limit=8, sources=DEFAULT_SEARCH_SOURCES)
    except Exception:
        candidates = []

    best = _pick_best_candidate(entry, candidates)
    if best is not None:
        return best

    return None


def resolve_papers_from_trajectory(
    doc: Dict[str, Any],
    *,
    search_limit: int = 8,
    enable_remote_lookup: bool = False,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[PaperMetadata]:
    resolved: List[PaperMetadata] = []
    seen_ids: set[str] = set()
    entries = _iter_trajectory_entries(doc)
    total_entries = len(entries)

    for index, entry in enumerate(entries, start=1):
        if progress_callback is not None:
            title_hint = str(entry.get("title") or entry.get("id") or f"paper {index}")
            progress_callback({
                "stage": "resolve",
                "current": index,
                "total": total_entries or 1,
                "message": f"Резолв публикаций: {index}/{total_entries or 1} — {title_hint[:80]}",
                "title": title_hint,
            })
        if not isinstance(entry, dict):
            continue
        meta = _resolve_entry_by_exact_identifier(entry, enable_remote_lookup=enable_remote_lookup)
        if meta is None and enable_remote_lookup:
            title = str(entry.get("title") or "").strip()
            if title:
                try:
                    candidates = search_papers(title, limit=search_limit, sources=DEFAULT_SEARCH_SOURCES)
                except Exception:
                    candidates = []
                meta = _pick_best_candidate(entry, candidates)
        if meta is None:
            meta = _fallback_paper(entry)

        # Fill missing fields from YAML entry when resolver found only partial metadata.
        if not meta.title and entry.get("title"):
            meta.title = str(entry.get("title") or "")
        if meta.year is None and entry.get("year") not in (None, ""):
            try:
                meta.year = int(entry.get("year"))
            except Exception:
                pass
        if not meta.url and _looks_like_url(str(entry.get("id") or "")):
            meta.url = str(entry.get("id") or "")

        acquire_hints = _trajectory_acquire_hints_for_entry(doc, entry)
        if acquire_hints:
            raw_payload = dict(meta.raw or {})
            existing_hints = raw_payload.get("acquire_hints") or []
            if isinstance(existing_hints, (str, bytes)):
                existing_hints = [existing_hints]
            merged_hints: List[str] = []
            for hint in list(existing_hints) + acquire_hints:
                hint = str(hint or "").strip()
                if hint and hint not in merged_hints:
                    merged_hints.append(hint)
            raw_payload["acquire_hints"] = merged_hints
            meta.raw = raw_payload

        if not meta.pdf_url:
            pdf_hints = _trajectory_pdf_hints_for_entry(doc, entry)
            if pdf_hints:
                meta.pdf_url = pdf_hints[0]

        if not meta.url and acquire_hints:
            landing_hints = [h for h in acquire_hints if _looks_like_url(h) and not str(h).lower().endswith(".pdf")]
            if landing_hints:
                meta.url = landing_hints[0]

        if meta.id in seen_ids:
            continue
        seen_ids.add(meta.id)
        resolved.append(meta)

    return resolved


def _source_label(src: Dict[str, Any], idx: int) -> str:
    s = str(src.get("source") or f"source_{idx}")
    loc = str(src.get("locator") or src.get("page") or "").strip()
    return f"{s} {loc}".strip()


def _paper_lookup(doc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for p in doc.get("papers", []) or []:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("id") or "").strip()
        if pid:
            out[pid] = p
            doi = _extract_doi(pid)
            if doi:
                out[doi] = p
                out[f"doi:{doi}"] = p
        title = str(p.get("title") or "").strip()
        if title:
            out[_norm_title(title)] = p
    return out


def _year_from_source(source_ref: str, papers_by_key: Dict[str, Dict[str, Any]]) -> Optional[int]:
    ref = (source_ref or "").strip()
    candidates = [ref]
    doi = _extract_doi(ref)
    if doi:
        candidates.extend([doi, f"doi:{doi}"])
    arxiv = _extract_arxiv(ref)
    if arxiv:
        candidates.extend([arxiv, f"arxiv:{arxiv}"])
    oa = _extract_openalex(ref)
    if oa:
        candidates.extend([oa, f"openalex:{oa}"])
    if _norm_title(ref):
        candidates.append(_norm_title(ref))

    for key in candidates:
        paper = papers_by_key.get(key)
        if not isinstance(paper, dict):
            continue
        try:
            y = int(paper.get("year")) if paper.get("year") not in (None, "") else None
        except Exception:
            y = None
        if y is not None:
            return y
    return None


def _step_time_window(step: Dict[str, Any], papers_by_key: Dict[str, Dict[str, Any]]) -> Tuple[str, str]:
    years: List[int] = []
    for src in step.get("sources", []) or []:
        if not isinstance(src, dict):
            continue
        y = _year_from_source(str(src.get("source") or ""), papers_by_key)
        if y is not None:
            years.append(y)
    if not years:
        return ("unknown", "unknown")
    return (str(min(years)), str(max(years)))


def _coerce_time_token(value: Any, *, default: str = "unknown") -> str:
    if value is None:
        return default
    s = str(value).strip()
    if not s:
        return default
    if DATE_TOKEN_RE.match(s):
        return s
    return default


def _qid_node_id(prefix: str, payload: Dict[str, Any]) -> str:
    qid = str(payload.get("id") or "").strip()
    if qid:
        return f"{prefix}:{qid}"
    label = _slugify(str(payload.get("label") or prefix))
    return f"{prefix}:{label}"


def _add_discovery_context_nodes(
    *,
    step_node_id: str,
    step: Dict[str, Any],
    manual_nodes: List[Dict[str, Any]],
    manual_edges: List[Dict[str, Any]],
    manual_triplets: List[Dict[str, Any]],
    start_date: str,
    end_date: str,
) -> None:
    ctx = step.get("discovery_context") if isinstance(step.get("discovery_context"), dict) else {}
    if not ctx:
        return
    geography = ctx.get("geography") if isinstance(ctx.get("geography"), dict) else {}
    country = geography.get("country") if isinstance(geography, dict) and isinstance(geography.get("country"), dict) else None
    city = geography.get("city") if isinstance(geography, dict) and isinstance(geography.get("city"), dict) else None
    branches = ctx.get("science_branches") if isinstance(ctx.get("science_branches"), list) else []

    if country:
        country_node = {
            "id": _qid_node_id("country", country),
            "type": "country",
            "label": str(country.get("label") or country.get("id") or "Country"),
            "wikidata_id": str(country.get("id") or ""),
            "start_date": start_date,
            "end_date": end_date,
        }
        if country_node["id"] not in {n.get("id") for n in manual_nodes}:
            manual_nodes.append(country_node)
        manual_edges.append({
            "source": step_node_id,
            "target": country_node["id"],
            "predicate": "origin_country",
            "type": "discovery_geography",
            "start_date": start_date,
            "end_date": end_date,
        })
        manual_triplets.append({
            "assertion_id": f"{step_node_id}:country",
            "subject": step_node_id,
            "predicate": "origin_country",
            "object": country_node["id"],
            "start_date": start_date,
            "end_date": end_date,
            "valid_from": start_date,
            "valid_to": end_date,
            "time_source": "metadata",
            "time_interval": f"evidence:{start_date}..{end_date}|valid:{start_date}..{end_date}",
            "evidence": {"snippet_or_summary": "Discovery context country from Task 1 trajectory."},
            "verdict": "accepted",
            "rationale": "Discovery geography country reconstructed from Task 1 trajectory YAML.",
        })

    if city:
        city_node = {
            "id": _qid_node_id("city", city),
            "type": "city",
            "label": str(city.get("label") or city.get("id") or "City"),
            "wikidata_id": str(city.get("id") or ""),
            "start_date": start_date,
            "end_date": end_date,
        }
        if city_node["id"] not in {n.get("id") for n in manual_nodes}:
            manual_nodes.append(city_node)
        manual_edges.append({
            "source": step_node_id,
            "target": city_node["id"],
            "predicate": "origin_city",
            "type": "discovery_geography",
            "start_date": start_date,
            "end_date": end_date,
        })
        if country:
            manual_edges.append({
                "source": city_node["id"],
                "target": _qid_node_id("country", country),
                "predicate": "located_in_country",
                "type": "geography_hierarchy",
                "start_date": start_date,
                "end_date": end_date,
            })

    for idx, branch in enumerate(branches, start=1):
        if not isinstance(branch, dict):
            continue
        branch_node = {
            "id": _qid_node_id("branch", branch),
            "type": "science_branch",
            "label": str(branch.get("label") or branch.get("id") or f"branch_{idx}"),
            "wikidata_id": str(branch.get("id") or ""),
            "start_date": start_date,
            "end_date": end_date,
        }
        if branch_node["id"] not in {n.get("id") for n in manual_nodes}:
            manual_nodes.append(branch_node)
        manual_edges.append({
            "source": step_node_id,
            "target": branch_node["id"],
            "predicate": "science_branch",
            "type": "discovery_branch",
            "start_date": start_date,
            "end_date": end_date,
        })


def _edge_record_variants(edge: Dict[str, Any], *, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    src_step = int(edge.get("from_step_id"))
    dst_step = int(edge.get("to_step_id"))
    src_node = f"step:{src_step}"
    dst_node = f"step:{dst_step}"
    predicate = str(edge.get("predicate") or "leads_to")
    directionality = str(edge.get("directionality") or "directed").strip().lower()
    direction_label = str(edge.get("direction_label") or "").strip()
    simultaneous = bool(edge.get("simultaneous_discovery"))
    base = {
        "predicate": predicate,
        "type": "reasoning_transition",
        "directionality": directionality,
        "direction_label": direction_label,
        "simultaneous_discovery": simultaneous,
        "start_date": start_date,
        "end_date": end_date,
    }
    variants = [dict(base, source=src_node, target=dst_node)]
    if directionality in {"bidirectional", "simultaneous"}:
        variants.append(dict(base, source=dst_node, target=src_node))
    return variants



def _canonical_time_interval(rec: Dict[str, Any]) -> str:
    start = _coerce_time_token(rec.get("start_date"), default="unknown")
    end = _coerce_time_token(rec.get("end_date"), default="unknown")
    valid_from = _coerce_time_token(rec.get("valid_from"), default=start)
    valid_to = _coerce_time_token(rec.get("valid_to"), default="+inf")
    legacy = str(rec.get("time_interval") or "").strip()
    if legacy:
        return legacy
    return f"evidence:{start}..{end}|valid:{valid_from}..{valid_to}"


def build_reference_graph(doc: Dict[str, Any]) -> Dict[str, Any]:
    papers_by_key = _paper_lookup(doc)
    steps = [s for s in (doc.get("steps") or []) if isinstance(s, dict)]
    manual_nodes: List[Dict[str, Any]] = []
    manual_edges: List[Dict[str, Any]] = []
    manual_triplets: List[Dict[str, Any]] = []

    step_ids: List[int] = []
    for idx, step in enumerate(steps, start=1):
        sid = int(step.get("step_id") or idx)
        step_ids.append(sid)
        start_date, end_date = _step_time_window(step, papers_by_key)
        node_id = f"step:{sid}"
        conditions = step.get("conditions") if isinstance(step.get("conditions"), dict) else {}
        discovery_context = step.get("discovery_context") if isinstance(step.get("discovery_context"), dict) else {}
        manual_nodes.append(
            {
                "id": node_id,
                "type": "trajectory_step",
                "label": str(step.get("claim") or f"Шаг {sid}"),
                "step_id": sid,
                "claim": str(step.get("claim") or ""),
                "inference": str(step.get("inference") or ""),
                "next_question": str(step.get("next_question") or ""),
                "conditions": conditions,
                "discovery_context": discovery_context,
                "simultaneous_discovery": bool(discovery_context.get("simultaneous_discovery")) if discovery_context else False,
                "start_date": start_date,
                "end_date": end_date,
                "valid_from": start_date,
                "valid_to": "+inf" if start_date != "unknown" else "unknown",
                "time_source": "metadata",
            }
        )
        _add_discovery_context_nodes(
            step_node_id=node_id,
            step=step,
            manual_nodes=manual_nodes,
            manual_edges=manual_edges,
            manual_triplets=manual_triplets,
            start_date=start_date,
            end_date=end_date,
        )
        manual_triplets.append(
            {
                "assertion_id": f"manual-step-{sid}",
                "subject": node_id,
                "predicate": "states",
                "object": str(step.get("claim") or ""),
                "start_date": start_date,
                "end_date": end_date,
                "valid_from": start_date,
                "valid_to": "+inf" if start_date != "unknown" else "unknown",
                "time_source": "metadata",
                "time_interval": f"evidence:{start_date}..{end_date}|valid:{start_date}..{'+inf' if start_date != 'unknown' else 'unknown'}",
                "evidence": {
                    "page": None,
                    "figure_or_table": None,
                    "snippet_or_summary": str(step.get("inference") or step.get("claim") or ""),
                },
                "verdict": "accepted",
                "rationale": "Reference step reconstructed from Task 1 trajectory YAML.",
            }
        )

        for src_idx, src in enumerate(step.get("sources", []) or [], start=1):
            if not isinstance(src, dict):
                continue
            source_id = f"step:{sid}:source:{src_idx:02d}"
            stype = str(src.get("type") or "text")
            source_label = _source_label(src, src_idx)
            page_value = src.get("page")
            try:
                page_value = int(page_value) if page_value not in (None, "") else None
            except Exception:
                page_value = None
            manual_nodes.append(
                {
                    "id": source_id,
                    "type": "evidence_source",
                    "label": source_label,
                    "source_type": stype,
                    "source_ref": str(src.get("source") or ""),
                    "locator": str(src.get("locator") or ""),
                    "page": page_value,
                    "snippet_or_summary": str(src.get("snippet_or_summary") or ""),
                    "start_date": start_date,
                    "end_date": end_date,
                    "valid_from": start_date,
                    "valid_to": end_date,
                    "time_source": "metadata",
                }
            )
            manual_edges.append(
                {
                    "source": source_id,
                    "target": node_id,
                    "predicate": "supports",
                    "type": "evidence_support",
                    "start_date": start_date,
                    "end_date": end_date,
                }
            )
            manual_triplets.append(
                {
                    "assertion_id": f"manual-source-{sid}-{src_idx}",
                    "subject": str(src.get("source") or source_id),
                    "predicate": "supports_step",
                    "object": node_id,
                    "start_date": start_date,
                    "end_date": end_date,
                    "valid_from": start_date,
                    "valid_to": end_date,
                    "time_source": "metadata",
                    "time_interval": f"evidence:{start_date}..{end_date}|valid:{start_date}..{end_date}",
                    "evidence": {
                        "page": page_value,
                        "figure_or_table": str(src.get("locator") or "") or None,
                        "snippet_or_summary": str(src.get("snippet_or_summary") or ""),
                    },
                    "verdict": "accepted",
                    "rationale": "Evidence link reconstructed from Task 1 trajectory YAML.",
                }
            )

    raw_edges = doc.get("edges") or []
    if not raw_edges and len(step_ids) > 1:
        raw_edges = [{"from_step_id": step_ids[i], "to_step_id": step_ids[i + 1], "predicate": "leads_to", "directionality": "directed"} for i in range(len(step_ids) - 1)]

    step_map = {int(step.get("step_id") or idx): step for idx, step in enumerate(steps, start=1)}
    for edge_idx, edge in enumerate(raw_edges, start=1):
        edge_obj: Dict[str, Any]
        if isinstance(edge, (list, tuple)) and len(edge) == 2:
            edge_obj = {"from_step_id": edge[0], "to_step_id": edge[1], "predicate": "leads_to", "directionality": "directed"}
        elif isinstance(edge, dict):
            edge_obj = dict(edge)
        else:
            continue
        try:
            src_step = int(edge_obj.get("from_step_id"))
            dst_step = int(edge_obj.get("to_step_id"))
        except Exception:
            continue
        src_step_obj = step_map.get(src_step, {})
        start_date, end_date = _step_time_window(src_step_obj, papers_by_key) if src_step_obj else ("unknown", "unknown")
        for variant_idx, edge_variant in enumerate(_edge_record_variants(edge_obj, start_date=start_date, end_date=end_date), start=1):
            manual_edges.append(edge_variant)
            manual_triplets.append(
                {
                    "assertion_id": f"manual-edge-{edge_idx}-{variant_idx}",
                    "subject": edge_variant["source"],
                    "predicate": edge_variant.get("predicate") or "leads_to",
                    "object": edge_variant["target"],
                    "start_date": start_date,
                    "end_date": end_date,
                    "valid_from": start_date,
                    "valid_to": "+inf" if start_date != "unknown" else "unknown",
                    "time_source": "metadata",
                    "time_interval": f"evidence:{start_date}..{end_date}|valid:{start_date}..{'+inf' if start_date != 'unknown' else 'unknown'}",
                    "directionality": edge_variant.get("directionality"),
                    "direction_label": edge_variant.get("direction_label"),
                    "simultaneous_discovery": bool(edge_variant.get("simultaneous_discovery")),
                    "evidence": {
                        "page": None,
                        "figure_or_table": None,
                        "snippet_or_summary": str(src_step_obj.get("next_question") or src_step_obj.get("inference") or ""),
                    },
                    "verdict": "accepted",
                    "rationale": "Reasoning transition reconstructed from Task 1 trajectory YAML.",
                }
            )

    return {
        "meta": {
            "kind": "reference_reasoning_graph",
            "artifact_version": int(doc.get("artifact_version") or 1),
            "topic": str(doc.get("topic") or ""),
            "domain": str(doc.get("domain") or ""),
            "submission_id": str(doc.get("submission_id") or ""),
            "generated_at": _utc_now(),
            "n_steps": len(steps),
            "n_papers": len(doc.get("papers") or []),
        },
        "nodes": manual_nodes,
        "edges": manual_edges,
        "triplets": manual_triplets,
    }


def _paperrecord_from_metadata(meta: PaperMetadata) -> PaperRecord:
    text = (meta.abstract or "").strip() or meta.title
    return PaperRecord(
        paper_id=meta.id,
        title=meta.title,
        year=_paper_year(meta),
        text=text,
        url=meta.url or "",
        source=str(meta.source.value if hasattr(meta.source, "value") else meta.source),
        evidence_units=[],
    )


def _load_domain_from_trajectory(doc: Dict[str, Any]) -> DomainConfig:
    value = str(doc.get("domain") or "science").strip()
    return load_domain_config(value or None)


def _flatten_automatic_graph(kg: TemporalKnowledgeGraph) -> List[Dict[str, Any]]:
    def _time_sort_key(value: str) -> str:
        raw = str(value or "").strip()
        if not raw:
            return "9999-99-99"
        parts = raw.split("-")
        if len(parts) == 1:
            return f"{parts[0]}-99-99"
        if len(parts) == 2:
            return f"{parts[0]}-{parts[1]}-99"
        return raw

    rows: List[Dict[str, Any]] = []
    kg_json = kg.to_json_dict()
    for idx, edge in enumerate(kg_json.get("edges", []) or [], start=1):
        intervals = [item for item in (edge.get("time_intervals") or []) if isinstance(item, dict)]
        extracted_intervals = [item for item in intervals if str(item.get("source") or "") == "extracted"]
        selected_intervals = extracted_intervals or intervals

        if selected_intervals:
            starts = [str(item.get("start") or "").strip() for item in selected_intervals if str(item.get("start") or "").strip()]
            ends = [str(item.get("end") or item.get("start") or "").strip() for item in selected_intervals if str(item.get("end") or item.get("start") or "").strip()]
            start_date = min(starts, key=_time_sort_key) if starts else "unknown"
            end_date = max(ends, key=_time_sort_key) if ends else start_date
            time_source = "triplet_extractor" if extracted_intervals else "paper_year_fallback"
        else:
            years = []
            try:
                years = sorted(int(y) for y in (edge.get("yearly_count") or {}).keys())
            except Exception:
                years = []
            start_date = str(years[0]) if years else "unknown"
            end_date = str(years[-1]) if years else "unknown"
            time_source = "metadata"

        quotes = edge.get("evidence_quotes") or []
        first_quote = quotes[0] if quotes else {}
        rows.append(
            {
                "assertion_id": f"auto-{idx:05d}",
                "subject": str(edge.get("source") or ""),
                "predicate": str(edge.get("predicate") or ""),
                "object": str(edge.get("target") or ""),
                "start_date": start_date,
                "end_date": end_date,
                "valid_from": start_date,
                "valid_to": end_date if end_date != "unknown" else ("+inf" if start_date != "unknown" else "unknown"),
                "time_source": time_source,
                "time_interval": f"evidence:{start_date}..{end_date}|valid:{start_date}..{(end_date if end_date != 'unknown' else ('+inf' if start_date != 'unknown' else 'unknown'))}",
                "time_candidates": selected_intervals[:10],
                "score": edge.get("score"),
                "mean_confidence": edge.get("mean_confidence"),
                "papers": edge.get("papers") or [],
                "evidence": {
                    "page": first_quote.get("page"),
                    "figure_or_table": None,
                    "snippet_or_summary": str(first_quote.get("quote") or ""),
                    "paper_id": str(first_quote.get("paper_id") or ""),
                    "source_kind": str(first_quote.get("source_kind") or ""),
                    "image_path": str(first_quote.get("image_path") or ""),
                },
            }
        )
    return rows


def _prefill_graph_review(doc: Dict[str, Any], automatic_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    reviewer_id = ""
    expert = doc.get("expert") if isinstance(doc.get("expert"), dict) else {}
    if expert:
        reviewer_id = str(expert.get("latin_slug") or expert.get("full_name") or "")

    assertions = []
    cutoff_year = str(doc.get("cutoff_year") or "")
    for row in automatic_rows:
        item = dict(row)
        item.update({
            "verdict": "",
            "rationale": "",
            "time_source_note": "",
            "semantic_correctness": "",
            "evidence_sufficiency": "",
            "scope_match": "",
            "system_match": "",
            "environment_match": "",
            "protocol_match": "",
            "scope_overgeneralized": False,
            "corrected_scope_note": "",
            "hypothesis_role": "background",
            "hypothesis_relevance": "1",
            "testability_signal": "1",
            "causal_status": "descriptive",
            "severity": "warning",
            "evidence_before_cutoff": "",
            "leakage_risk": "possible",
            "time_type": "publication_time",
            "time_granularity": "unknown",
            "time_confidence": "medium",
            "mm_verdict": "",
            "mm_rationale": "",
            "cutoff_year": cutoff_year,
        })
        assertions.append(item)

    return {
        "artifact_version": 5,
        "domain": str(doc.get("domain") or ""),
        "topic": str(doc.get("topic") or ""),
        "trajectory_submission_id": str(doc.get("submission_id") or ""),
        "cutoff_year": cutoff_year,
        "reviewer_id": reviewer_id,
        "timestamp": _utc_now(),
        "assertions": assertions,
    }


def _empty_temporal_corrections(doc: Dict[str, Any]) -> Dict[str, Any]:
    reviewer_id = ""
    expert = doc.get("expert") if isinstance(doc.get("expert"), dict) else {}
    if expert:
        reviewer_id = str(expert.get("latin_slug") or expert.get("full_name") or "")

    return {
        "artifact_version": 2,
        "domain": str(doc.get("domain") or ""),
        "paper_id": "",
        "reviewer_id": reviewer_id,
        "trajectory_submission_id": str(doc.get("submission_id") or ""),
        "corrections": [],
    }


def _compare_graphs(reference_graph: Dict[str, Any], automatic_rows: Sequence[Dict[str, Any]], resolved_papers: Sequence[PaperMetadata]) -> Dict[str, Any]:
    ref_step_labels = {str(n.get("label") or "") for n in reference_graph.get("nodes", []) if n.get("type") == "trajectory_step"}
    auto_terms = {str(r.get("subject") or "") for r in automatic_rows} | {str(r.get("object") or "") for r in automatic_rows}
    paper_ids = {p.id for p in resolved_papers}
    auto_papers = set()
    for row in automatic_rows:
        auto_papers.update(str(x) for x in (row.get("papers") or []))

    return {
        "reference_steps": len([n for n in reference_graph.get("nodes", []) if n.get("type") == "trajectory_step"]),
        "reference_edges": len([e for e in reference_graph.get("edges", []) if e.get("predicate") == "leads_to"]),
        "automatic_edges": len(list(automatic_rows)),
        "automatic_unique_terms": len(auto_terms),
        "resolved_papers": len(list(resolved_papers)),
        "papers_reaching_automatic_graph": len(auto_papers & paper_ids),
        "trajectory_claim_examples": sorted(list(ref_step_labels))[:10],
        "automatic_term_examples": sorted(list(auto_terms))[:20],
        "cutoff_year": str(reference_graph.get("metadata", {}).get("cutoff_year") or ""),
        "review_focus": "Prefer edges that can seed temporally valid, condition-aware hypotheses.",
    }




def _apply_exclusion_filter_to_resolved(resolved: Sequence[PaperMetadata], exclusion_spec: Any) -> Tuple[List[PaperMetadata], List[Dict[str, Any]]]:
    rules = normalize_exclusion_spec(exclusion_spec)
    if not any([rules['paper_ids'], rules['titles'], rules['source_refs'], rules['match_substrings'], rules['url_substrings'], rules['max_year'] is not None]):
        return list(resolved), []

    kept: List[PaperMetadata] = []
    excluded: List[Dict[str, Any]] = []
    for meta in resolved:
        if entity_matches_exclusion(meta, rules):
            excluded.append({
                'paper_id': meta.id,
                'title': meta.title,
                'year': _paper_year(meta),
                'url': meta.url,
                'pdf_url': meta.pdf_url,
            })
        else:
            kept.append(meta)
    return kept, excluded


def _annotate_triplets_with_importance(rows: Sequence[Dict[str, Any]], doc: Dict[str, Any], *, graph_metrics: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    profile = topic_profile_from_doc(doc)
    enriched: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item.update(score_triplet_importance(item, profile, graph_metrics=graph_metrics))
        enriched.append(item)
    return enriched
def suggest_link_candidates(
    doc: Dict[str, Any],
    *,
    known_papers: Sequence[PaperMetadata],
    max_queries: int = 4,
    per_query: int = 8,
    enable_remote_lookup: bool = False,
) -> List[Dict[str, Any]]:
    if not enable_remote_lookup:
        return []

    queries: List[str] = []
    topic = str(doc.get("topic") or "").strip()
    if topic:
        queries.append(topic)

    for step in doc.get("steps", []) or []:
        if not isinstance(step, dict):
            continue
        nq = str(step.get("next_question") or "").strip()
        if nq and nq not in queries:
            queries.append(nq)
        inf = str(step.get("inference") or "").strip()
        if inf and len(queries) < max_queries and inf not in queries:
            queries.append(inf)
        if len(queries) >= max_queries:
            break

    queries = queries[:max_queries]
    known_titles = {_norm_title(p.title) for p in known_papers}
    known_ids = {p.id for p in known_papers}
    suggestions: Dict[str, Dict[str, Any]] = {}

    for q in queries:
        try:
            found = search_papers(q, limit=per_query, sources=DEFAULT_SEARCH_SOURCES)
        except Exception:
            found = []
        for cand in found:
            if cand.id in known_ids or _norm_title(cand.title) in known_titles:
                continue
            score = 2.0 * _score_title_match(q, cand.title)
            if cand.abstract:
                score += 0.25
            if cand.pdf_url:
                score += 0.25
            if cand.citation_count:
                score += min(float(cand.citation_count) / 500.0, 0.5)
            existing = suggestions.get(cand.id)
            payload = {
                "paper_id": cand.id,
                "title": cand.title,
                "year": _paper_year(cand),
                "url": cand.url,
                "pdf_url": cand.pdf_url,
                "source": str(cand.source.value if hasattr(cand.source, "value") else cand.source),
                "trigger_queries": [q],
                "score": round(score, 4),
            }
            if existing is None:
                suggestions[cand.id] = payload
            else:
                existing["score"] = max(float(existing.get("score") or 0.0), payload["score"])
                tq = list(existing.get("trigger_queries") or [])
                if q not in tq:
                    tq.append(q)
                existing["trigger_queries"] = tq

    ranked = sorted(suggestions.values(), key=lambda x: float(x.get("score") or 0.0), reverse=True)
    return ranked[:20]


def prepare_task2_validation_bundle(
    trajectory_yaml: Path,
    *,
    out_dir: Path,
    include_multimodal: bool = True,
    run_vlm: bool = True,
    edge_mode: str = "auto",
    suggest_links: bool = True,
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
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Path:
    with temporary_llm_selection(
        llm_provider=llm_provider,
        llm_model=llm_model,
        g4f_model=g4f_model,
        local_model=local_model,
    ):
        with temporary_vlm_selection(vlm_backend=vlm_backend, vlm_model_id=vlm_model_id):
            total_stages = 8 + (1 if suggest_links else 0)
            _emit_progress(progress_callback, stage="load", current=1, total=total_stages, message="Читаю YAML и готовлю рабочую директорию")

            doc = yaml.safe_load(trajectory_yaml.read_text(encoding="utf-8")) or {}
            if not isinstance(doc, dict):
                raise ValueError("Trajectory YAML must contain a top-level object.")

            run_name = str(doc.get("submission_id") or _slugify(str(doc.get("topic") or trajectory_yaml.stem)))
            out = out_dir / run_name
            out.mkdir(parents=True, exist_ok=True)

            shutil.copy2(trajectory_yaml, out / trajectory_yaml.name)

            normalized_exclusion_spec = normalize_exclusion_spec(exclusion_spec)
            exclusion_payload = serialize_exclusion_spec(normalized_exclusion_spec)
            (out / "exclusion_rules.json").write_text(json.dumps(exclusion_payload, ensure_ascii=False, indent=2), encoding="utf-8")

            _emit_progress(progress_callback, stage="reference", current=2, total=total_stages, message="Строю reference graph из YAML")
            reference_graph = build_reference_graph(doc)
            (out / "reference_graph.json").write_text(json.dumps(reference_graph, ensure_ascii=False, indent=2), encoding="utf-8")
            (out / "reference_triplets.json").write_text(json.dumps(reference_graph.get("triplets") or [], ensure_ascii=False, indent=2), encoding="utf-8")

            domain_cfg = _load_domain_from_trajectory(doc)

            _emit_progress(progress_callback, stage="resolve", current=3, total=total_stages, message="Резолвлю публикации и идентификаторы")
            resolved = resolve_papers_from_trajectory(doc, enable_remote_lookup=enable_remote_lookup, progress_callback=progress_callback)
            if max_papers and max_papers > 0:
                resolved = resolved[:max_papers]

            resolved, excluded_papers = _apply_exclusion_filter_to_resolved(resolved, normalized_exclusion_spec)
            (out / "excluded_papers.json").write_text(
                json.dumps(excluded_papers, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            (out / "papers_resolved.json").write_text(
                json.dumps([p.model_dump(mode="json") for p in resolved], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            acquire_dir = out / "automatic_graph"
            raw_dir = acquire_dir / "raw_pdfs"
            meta_dir = acquire_dir / "raw_meta"
            processed_dir = acquire_dir / "processed_papers"
            processed_dir.mkdir(parents=True, exist_ok=True)

            _emit_progress(progress_callback, stage="acquire", current=4, total=total_stages, message=f"Скачиваю PDF и сохраняю метаданные: 0/{len(resolved)}")
            total_resolved = len(resolved)
            for idx, meta in enumerate(resolved, start=1):
                title_hint = meta.title or meta.id
                _emit_progress(
                    progress_callback,
                    stage="acquire",
                    current=4,
                    total=total_stages,
                    message=f"Скачиваю PDF и сохраняю метаданные: {idx}/{total_resolved} — {title_hint[:80]}",
                    item_current=idx,
                    item_total=total_resolved,
                    paper_id=meta.id,
                    paper_title=title_hint,
                )
            acq: List[AcquireResult] = list(acquire_pdfs(resolved, raw_dir=raw_dir, meta_dir=meta_dir))
            (out / "acquire_results.json").write_text(
                json.dumps(
                    [asdict(a) | {"pdf_path": str(a.pdf_path) if a.pdf_path else None, "meta_path": str(a.meta_path)} for a in acq],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            ingested_ids: List[str] = []
            total_to_ingest = sum(1 for item in acq if item.pdf_path)
            _emit_progress(progress_callback, stage="ingest", current=5, total=total_stages, message=f"Парсю PDF и строю multimodal представление: 0/{total_to_ingest}")
            ingest_index = 0
            for meta, a in zip(resolved, acq):
                if not a.pdf_path:
                    continue
                ingest_index += 1
                meta_json = meta.model_dump(mode="json")
                title_hint = meta.title or meta.id
                _emit_progress(
                    progress_callback,
                    stage="ingest",
                    current=5,
                    total=total_stages,
                    message=f"Парсю PDF и строю multimodal представление: {ingest_index}/{total_to_ingest} — {title_hint[:80]}",
                    item_current=ingest_index,
                    item_total=total_to_ingest,
                    paper_id=meta.id,
                    paper_title=title_hint,
                )
                try:
                    if include_multimodal:
                        ingest_pdf_multimodal_auto(
                            a.pdf_path,
                            meta_json,
                            processed_dir,
                            run_vlm=run_vlm,
                            progress_callback=(
                                (lambda payload, *, meta=meta, index=ingest_index, total=total_to_ingest: _emit_progress(
                                    progress_callback,
                                    stage="pages",
                                    current=5,
                                    total=total_stages,
                                    message=f"{meta.title or meta.id}: {payload.get('message') or ''}",
                                    item_current=index,
                                    item_total=total,
                                    paper_id=meta.id,
                                    paper_title=meta.title or meta.id,
                                    page_current=payload.get("current"),
                                    page_total=payload.get("total"),
                                )) if progress_callback is not None else None
                            ),
                        )
                    else:
                        ingest_pdf_auto(a.pdf_path, meta_json, processed_dir)
                    ingested_ids.append(meta.id)
                except Exception as e:
                    console.print(f"[yellow]Ingest failed for {meta.id}: {e}[/yellow]")

            _emit_progress(progress_callback, stage="records", current=6, total=total_stages, message="Собираю записи публикаций для temporal KG")
            processed_records = load_papers_from_processed(processed_dir)
            processed_by_id = {p.paper_id: p for p in processed_records}
            paper_records: List[PaperRecord] = []
            for meta in resolved:
                if meta.id in processed_by_id:
                    paper_records.append(processed_by_id[meta.id])
                else:
                    paper_records.append(_paperrecord_from_metadata(meta))

            (out / "paper_records.json").write_text(
                json.dumps([asdict(r) for r in paper_records], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            _emit_progress(progress_callback, stage="kg", current=7, total=total_stages, message="Строю temporal knowledge graph")
            kg = build_temporal_kg(
                paper_records,
                domain=domain_cfg,
                query=str(doc.get("topic") or ""),
                edge_mode=edge_mode,  # type: ignore[arg-type]
                expert_overrides_path=None,
                llm_provider=str(settings.llm_provider or "") or None,
                llm_model=str(settings.llm_model or "") or None,
            )
            kg.dump_json(out / "automatic_graph" / "temporal_kg.json")

            automatic_rows = _flatten_automatic_graph(kg)
            automatic_rows = _annotate_triplets_with_importance(automatic_rows, doc)
            (out / "automatic_triplets.json").write_text(json.dumps(automatic_rows, ensure_ascii=False, indent=2), encoding="utf-8")

            prefill_review = _prefill_graph_review(doc, automatic_rows)
            review_dir = out / "review_templates"
            review_dir.mkdir(parents=True, exist_ok=True)
            (review_dir / "graph_review_prefill.json").write_text(json.dumps(prefill_review, ensure_ascii=False, indent=2), encoding="utf-8")
            (review_dir / "temporal_corrections_template.json").write_text(json.dumps(_empty_temporal_corrections(doc), ensure_ascii=False, indent=2), encoding="utf-8")

            _emit_progress(progress_callback, stage="compare", current=8, total=total_stages, message="Сравниваю automatic graph с reference graph")
            comparison = _compare_graphs(reference_graph, automatic_rows, resolved)
            (out / "comparison_summary.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

            if suggest_links:
                _emit_progress(progress_callback, stage="scout", current=9, total=total_stages, message="Генерирую reference scout и кандидатов на дополнительные ссылки")
                suggestions = suggest_link_candidates(
                    doc,
                    known_papers=resolved,
                    max_queries=max_link_queries,
                    enable_remote_lookup=enable_remote_lookup,
                )
                scout_dir = out / "scout"
                scout_dir.mkdir(parents=True, exist_ok=True)
                (scout_dir / "suggested_links.json").write_text(json.dumps(suggestions, ensure_ascii=False, indent=2), encoding="utf-8")

            review_state_dir = out / "expert_validation" / "drafts"
            review_state_dir.mkdir(parents=True, exist_ok=True)

            manifest = {
                "generated_at": _utc_now(),
                "trajectory_file": trajectory_yaml.name,
                "submission_id": str(doc.get("submission_id") or ""),
                "topic": str(doc.get("topic") or ""),
                "domain": str(doc.get("domain") or ""),
                "resolved_papers": len(resolved),
                "ingested_pdfs": len(ingested_ids),
                "automatic_edges": len(automatic_rows),
                "reference_steps": len([n for n in reference_graph.get("nodes", []) if n.get("type") == "trajectory_step"]),
                "remote_lookup_enabled": bool(enable_remote_lookup),
                "excluded_papers": len(excluded_papers),
                "llm_effective_provider": str(settings.llm_provider or ""),
                "llm_effective_model": str(settings.llm_model or ""),
                "vlm_effective_backend": str(getattr(settings, "vlm_backend", "") or ""),
                "vlm_effective_model": str(getattr(settings, "vlm_model_id", "") or ""),
                "review_state_dir": str(review_state_dir),
                "review_state_latest": str(review_state_dir / "review_state_latest.json"),
                "artifacts": {
                    "reference_graph": "reference_graph.json",
                    "reference_triplets": "reference_triplets.json",
                    "automatic_graph": "automatic_graph/temporal_kg.json",
                    "automatic_triplets": "automatic_triplets.json",
                    "papers_resolved": "papers_resolved.json",
                    "comparison_summary": "comparison_summary.json",
                    "review_prefill": "review_templates/graph_review_prefill.json",
                    "excluded_papers": "excluded_papers.json",
                    "exclusion_rules": "exclusion_rules.json",
                    "review_design_note": "Task 2 review now captures semantic, evidence, temporal, scope and hypothesis-readiness signals.",
                },
                "filter_defaults": {
                    "importance_threshold": 0.0,
                    "exclusion_rules": exclusion_payload,
                },
            }
            _emit_progress(progress_callback, stage="finalize", current=total_stages, total=total_stages, message="Bundle собран, сохраняю manifest")
            (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            return out
