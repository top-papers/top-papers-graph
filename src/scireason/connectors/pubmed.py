from __future__ import annotations

from typing import Any, Dict, List, Optional

import xml.etree.ElementTree as ET

from scireason.net import get_json, get_text

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_DB = "pubmed"


def _headers(user_agent: Optional[str] = None) -> Dict[str, str]:
    h: Dict[str, str] = {}
    if user_agent:
        h["User-Agent"] = user_agent
    return h


def esearch(
    query: str,
    *,
    retstart: int = 0,
    retmax: int = 25,
    api_key: Optional[str] = None,
    tool: Optional[str] = None,
    email: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> List[str]:
    """Run PubMed ESearch and return a list of PMIDs."""
    params: Dict[str, Any] = {
        "db": PUBMED_DB,
        "term": query,
        "retmode": "json",
        "retstart": retstart,
        "retmax": retmax,
    }
    if api_key:
        params["api_key"] = api_key
    if tool:
        params["tool"] = tool
    if email:
        params["email"] = email

    data = get_json(f"{EUTILS_BASE}/esearch.fcgi", params=params, headers=_headers(user_agent), ttl_seconds=24 * 3600)
    return (data.get("esearchresult", {}) if isinstance(data, dict) else {}).get("idlist", []) or []


def esummary(
    pmids: List[str],
    *,
    api_key: Optional[str] = None,
    tool: Optional[str] = None,
    email: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> Dict[str, Any]:
    """Run PubMed ESummary for a list of PMIDs and return raw JSON."""
    if not pmids:
        return {}

    params: Dict[str, Any] = {
        "db": PUBMED_DB,
        "id": ",".join(pmids),
        "retmode": "json",
    }
    if api_key:
        params["api_key"] = api_key
    if tool:
        params["tool"] = tool
    if email:
        params["email"] = email

    data = get_json(f"{EUTILS_BASE}/esummary.fcgi", params=params, headers=_headers(user_agent), ttl_seconds=24 * 3600)
    return data if isinstance(data, dict) else {}


def fetch_abstracts(
    pmids: List[str],
    *,
    api_key: Optional[str] = None,
    tool: Optional[str] = None,
    email: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> Dict[str, str]:
    """Fetch abstracts for PMIDs using EFetch (XML)."""
    if not pmids:
        return {}

    params: Dict[str, Any] = {
        "db": PUBMED_DB,
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    if api_key:
        params["api_key"] = api_key
    if tool:
        params["tool"] = tool
    if email:
        params["email"] = email

    xml_text = get_text(f"{EUTILS_BASE}/efetch.fcgi", params=params, headers=_headers(user_agent), ttl_seconds=24 * 3600)

    root = ET.fromstring(xml_text)
    out: Dict[str, str] = {}

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//MedlineCitation/PMID")
        pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else ""
        if not pmid:
            continue

        abstract_parts: List[str] = []
        for at in article.findall(".//Article/Abstract/AbstractText"):
            if at.text:
                abstract_parts.append(at.text.strip())
        out[pmid] = "\n\n".join([p for p in abstract_parts if p])

    for pmid in pmids:
        out.setdefault(str(pmid), "")

    return out


def _extract_doi_from_esummary_item(item: Dict[str, Any]) -> Optional[str]:
    for rec in item.get("articleids", []) or []:
        try:
            if str(rec.get("idtype", "")).lower() == "doi":
                v = str(rec.get("value", "")).strip()
                return v or None
        except Exception:
            continue
    return None


def search(
    query: str,
    *,
    retmax: int = 25,
    retstart: int = 0,
    api_key: Optional[str] = None,
    tool: Optional[str] = None,
    email: Optional[str] = None,
    user_agent: Optional[str] = None,
    with_abstract: bool = False,
) -> List[Dict[str, Any]]:
    """Search PubMed using NCBI E-utilities.

    Returns a list of normalized dicts.
    """
    pmids = esearch(
        query,
        retstart=retstart,
        retmax=retmax,
        api_key=api_key,
        tool=tool,
        email=email,
        user_agent=user_agent,
    )
    if not pmids:
        return []

    summary = esummary(pmids, api_key=api_key, tool=tool, email=email, user_agent=user_agent)
    result = summary.get("result", {}) if isinstance(summary, dict) else {}

    abstracts: Dict[str, str] = {}
    if with_abstract:
        abstracts = fetch_abstracts(pmids, api_key=api_key, tool=tool, email=email, user_agent=user_agent)

    out: List[Dict[str, Any]] = []
    for pmid in pmids:
        item = result.get(str(pmid), {}) if isinstance(result, dict) else {}
        title = str(item.get("title", "") or "").strip()
        pubdate = str(item.get("pubdate", "") or "").strip()

        doi = _extract_doi_from_esummary_item(item)

        authors = item.get("authors", []) or []
        author_names: List[str] = []
        for a in authors:
            n = (a.get("name") if isinstance(a, dict) else None) or ""
            n = str(n).strip()
            if n:
                author_names.append(n)

        rec: Dict[str, Any] = {
            "id": f"pmid:{pmid}",
            "pmid": str(pmid),
            "title": title,
            "authors": author_names,
            "published": pubdate,
            "doi": doi,
            "source": "pubmed",
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        }

        if pubdate and len(pubdate) >= 4 and pubdate[:4].isdigit():
            rec["year"] = int(pubdate[:4])

        if with_abstract:
            rec["abstract"] = abstracts.get(str(pmid), "")

        out.append(rec)

    return out
