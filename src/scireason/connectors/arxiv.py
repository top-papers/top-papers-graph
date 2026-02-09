from __future__ import annotations

from typing import Any, Dict, List, Optional
import xml.etree.ElementTree as ET

from scireason.net import get_text

ARXIV_API = "https://export.arxiv.org/api/query"

ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def _headers(user_agent: Optional[str] = None) -> Dict[str, str]:
    h: Dict[str, str] = {}
    if user_agent:
        h["User-Agent"] = user_agent
    return h


def _parse_feed(xml_text: str) -> List[Dict[str, Any]]:
    root = ET.fromstring(xml_text)
    out: List[Dict[str, Any]] = []

    for entry in root.findall("atom:entry", namespaces=ATOM_NS):
        id_url = (entry.findtext("atom:id", default="", namespaces=ATOM_NS) or "").strip()
        arxiv_id = id_url.rstrip("/").split("/")[-1]

        title = (entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").strip().replace("\n", " ")
        summary = (entry.findtext("atom:summary", default="", namespaces=ATOM_NS) or "").strip()

        published = (entry.findtext("atom:published", default="", namespaces=ATOM_NS) or "").strip()
        updated = (entry.findtext("atom:updated", default="", namespaces=ATOM_NS) or "").strip()

        authors: List[str] = []
        for a in entry.findall("atom:author", namespaces=ATOM_NS):
            name = (a.findtext("atom:name", default="", namespaces=ATOM_NS) or "").strip()
            if name:
                authors.append(name)

        doi = (entry.findtext("arxiv:doi", default="", namespaces=ATOM_NS) or "").strip() or None

        pdf_url = None
        for link in entry.findall("atom:link", namespaces=ATOM_NS):
            if link.attrib.get("type") == "application/pdf" or link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href")
                break

        rec: Dict[str, Any] = {
            "arxiv_id": arxiv_id,
            "id": id_url,
            "url": id_url,
            "title": title,
            "summary": summary,
            "published": published,
            "updated": updated,
            "authors": authors,
            "doi": doi,
            "pdf_url": pdf_url,
            "source": "arxiv",
            "raw": {
                "id": id_url,
                "title": title,
                "summary": summary,
                "published": published,
                "updated": updated,
                "authors": authors,
                "doi": doi,
                "pdf_url": pdf_url,
            },
        }

        if published and len(published) >= 4 and published[:4].isdigit():
            rec["year"] = int(published[:4])

        out.append(rec)

    return out


def search(query: str, start: int = 0, max_results: int = 25, *, user_agent: Optional[str] = None) -> List[Dict[str, Any]]:
    """Search arXiv using the Atom API.

    NOTE: arXiv recommends a ~3 second delay between calls.
    Our shared HTTP client enforces this via per-host rate limiting.
    """
    params = {"search_query": query, "start": start, "max_results": max_results}
    xml_text = get_text(ARXIV_API, params=params, headers=_headers(user_agent), ttl_seconds=12 * 3600)
    return _parse_feed(xml_text)


def get_by_id(arxiv_id: str, *, user_agent: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch a record by arXiv ID via id_list."""
    params = {"id_list": arxiv_id.strip()}
    xml_text = get_text(ARXIV_API, params=params, headers=_headers(user_agent), ttl_seconds=30 * 24 * 3600)
    return _parse_feed(xml_text)
