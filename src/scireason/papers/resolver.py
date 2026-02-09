from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict

from scireason.config import settings
from scireason.net import get_json, get_text

from scireason.connectors import pubmed as pubmed_connector
from scireason.connectors import openalex as openalex_connector
from scireason.connectors import europe_pmc as europe_pmc_connector
from scireason.connectors import arxiv as arxiv_connector


class ResolvedIds(BaseModel):
    model_config = ConfigDict(extra="ignore")

    input: str

    doi: Optional[str] = None
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    arxiv: Optional[str] = None
    openalex: Optional[str] = None
    semantic_scholar: Optional[str] = None

    notes: list[str] = []


def _parse_id(x: str) -> Tuple[str, str]:
    s = (x or "").strip()
    if ":" in s:
        p, v = s.split(":", 1)
        p = p.strip().lower()
        v = v.strip()
        if p in {"doi", "pmid", "pmcid", "pmc", "arxiv", "openalex", "s2"}:
            return ("pmcid" if p in {"pmc", "pmcid"} else p, v)

    # heuristics
    if s.lower().startswith("10.") and "/" in s:
        return ("doi", s)
    if s.isdigit():
        return ("pmid", s)
    if s.upper().startswith("PMC") and s[3:].isdigit():
        return ("pmcid", s.upper())
    if s.lower().startswith("w") and s[1:].isdigit():
        return ("openalex", s)
    # arXiv ids are varied; best-effort
    if s.replace(".", "").replace("v", "").replace("/", "").replace("-", "").isalnum():
        return ("arxiv", s)
    return ("unknown", s)


def _ncbi_idconv(ids: str) -> Dict[str, Any]:
    """PMC ID Converter (works only for items in PMC)."""
    params: Dict[str, Any] = {"ids": ids, "format": "json"}
    if settings.ncbi_tool:
        params["tool"] = settings.ncbi_tool
    if settings.ncbi_email or settings.contact_email:
        params["email"] = settings.ncbi_email or settings.contact_email
    if settings.ncbi_api_key:
        params["api_key"] = settings.ncbi_api_key
    return get_json("https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/", params=params, ttl_seconds=30 * 24 * 3600)


def resolve_ids(identifier: str) -> ResolvedIds:
    kind, val = _parse_id(identifier)
    out = ResolvedIds(input=identifier, notes=[])

    if kind == "doi":
        out.doi = val
        # Try NCBI idconv (only for PMC)
        try:
            j = _ncbi_idconv(val)
            recs = j.get("records") if isinstance(j, dict) else None
            if isinstance(recs, list) and recs:
                r0 = recs[0] if isinstance(recs[0], dict) else {}
                out.pmid = r0.get("pmid") or out.pmid
                out.pmcid = r0.get("pmcid") or out.pmcid
                out.notes.append("ncbi_idconv")
        except Exception:
            pass

        # Europe PMC fallback (often provides PMID even if not in PMC)
        try:
            res = europe_pmc_connector.search(f"DOI:{val}", page_size=5, page=1, user_agent=settings.user_agent)
            if res:
                r0 = res[0]
                out.pmid = r0.get("pmid") or out.pmid
                out.pmcid = r0.get("pmcid") or out.pmcid
                out.notes.append("europe_pmc")
        except Exception:
            pass

        # OpenAlex resolution
        try:
            doi_url = f"https://doi.org/{val}"
            # OpenAlex supports /works/<doi_url>
            work = openalex_connector.get_work(f"https://api.openalex.org/works/{quote(doi_url, safe='')}", mailto=settings.openalex_mailto or settings.contact_email, api_key=settings.openalex_api_key, user_agent=settings.user_agent)
            if isinstance(work, dict):
                ids = work.get("ids") if isinstance(work.get("ids"), dict) else {}
                out.openalex = (ids.get("openalex") or work.get("id") or out.openalex)
                if isinstance(out.openalex, str) and out.openalex.startswith("https://openalex.org/"):
                    out.openalex = out.openalex.split("/")[-1]
                pmid = ids.get("pmid")
                if isinstance(pmid, str) and "pubmed.ncbi.nlm.nih.gov" in pmid:
                    out.pmid = pmid.rstrip("/").split("/")[-1]
                arxiv = ids.get("arxiv")
                if isinstance(arxiv, str) and "arxiv.org" in arxiv:
                    out.arxiv = arxiv.rstrip("/").split("/")[-1]
                out.notes.append("openalex")
        except Exception:
            pass

        return out

    if kind == "pmid":
        out.pmid = val
        # PubMed esummary can include DOI
        try:
            summary = pubmed_connector.esummary([val], api_key=settings.ncbi_api_key, tool=settings.ncbi_tool, email=settings.ncbi_email or settings.contact_email, user_agent=settings.user_agent)
            result = summary.get("result", {}) if isinstance(summary, dict) else {}
            item = result.get(val) or result.get(str(val)) or {}
            if isinstance(item, dict):
                doi = pubmed_connector._extract_doi_from_esummary_item(item)  # type: ignore[attr-defined]
                if doi:
                    out.doi = doi
                out.notes.append("pubmed_esummary")
        except Exception:
            pass

        # NCBI idconv can give PMCID if in PMC
        try:
            j = _ncbi_idconv(val)
            recs = j.get("records") if isinstance(j, dict) else None
            if isinstance(recs, list) and recs:
                r0 = recs[0] if isinstance(recs[0], dict) else {}
                out.pmcid = r0.get("pmcid") or out.pmcid
                out.doi = r0.get("doi") or out.doi
                out.notes.append("ncbi_idconv")
        except Exception:
            pass

        # OpenAlex if doi is known
        if out.doi:
            more = resolve_ids(f"doi:{out.doi}")
            out.openalex = out.openalex or more.openalex
            out.arxiv = out.arxiv or more.arxiv
            out.pmcid = out.pmcid or more.pmcid
            out.notes.extend([n for n in more.notes if n not in out.notes])

        return out

    if kind == "pmcid":
        out.pmcid = val.upper()
        # idconv best
        try:
            j = _ncbi_idconv(out.pmcid)
            recs = j.get("records") if isinstance(j, dict) else None
            if isinstance(recs, list) and recs:
                r0 = recs[0] if isinstance(recs[0], dict) else {}
                out.pmid = r0.get("pmid") or out.pmid
                out.doi = r0.get("doi") or out.doi
                out.notes.append("ncbi_idconv")
        except Exception:
            pass
        return out

    if kind == "openalex":
        out.openalex = val
        try:
            work = openalex_connector.get_work(f"https://api.openalex.org/works/{val}", mailto=settings.openalex_mailto or settings.contact_email, api_key=settings.openalex_api_key, user_agent=settings.user_agent)
            if isinstance(work, dict):
                ids = work.get("ids") if isinstance(work.get("ids"), dict) else {}
                doi = (ids.get("doi") or work.get("doi") or "").replace("https://doi.org/", "").strip() or None
                out.doi = doi or out.doi
                pmid = ids.get("pmid")
                if isinstance(pmid, str) and "pubmed.ncbi.nlm.nih.gov" in pmid:
                    out.pmid = pmid.rstrip("/").split("/")[-1]
                arxiv = ids.get("arxiv")
                if isinstance(arxiv, str) and "arxiv.org" in arxiv:
                    out.arxiv = arxiv.rstrip("/").split("/")[-1]
                out.notes.append("openalex")
        except Exception:
            pass
        return out

    if kind == "arxiv":
        out.arxiv = val
        # arXiv API lookup by id_list
        try:
            recs = arxiv_connector.get_by_id(val, user_agent=settings.user_agent)
            if recs:
                r0 = recs[0]
                doi = (r0.get("doi") or "").strip() or None
                out.doi = doi
                out.notes.append("arxiv_api")
        except Exception:
            pass

        if out.doi:
            more = resolve_ids(f"doi:{out.doi}")
            out.openalex = out.openalex or more.openalex
            out.pmid = out.pmid or more.pmid
            out.pmcid = out.pmcid or more.pmcid
            out.notes.extend([n for n in more.notes if n not in out.notes])

        return out

    out.notes.append("no_resolution")
    return out
