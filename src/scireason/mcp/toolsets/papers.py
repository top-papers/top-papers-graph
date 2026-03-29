from __future__ import annotations

from typing import Any, Dict, List, Optional

from scireason.papers import PaperSource, get_paper_by_doi, resolve_ids, search_papers

from ..decorators import scireason_mcp_tool


@scireason_mcp_tool(toolset="papers")
def search_papers_tool(
    query: str,
    limit: int = 10,
    sources: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Search scientific papers across multiple sources and return normalized metadata."""
    limit = max(1, min(int(limit), 50))
    srcs = None
    if sources:
        srcs = []
        for source_name in sources:
            try:
                srcs.append(PaperSource(source_name))
            except Exception:
                continue
    papers = search_papers(query, limit=limit, sources=srcs)
    return [paper.model_dump(mode="json") for paper in papers]


@scireason_mcp_tool(toolset="papers")
def resolve_ids_tool(identifier: str) -> Dict[str, Any]:
    """Resolve between DOI, PMID, arXivID, and OpenAlexID using the normalized paper layer."""
    return resolve_ids(identifier).model_dump(mode="json")


@scireason_mcp_tool(toolset="papers")
def get_paper_by_doi_tool(doi: str) -> Dict[str, Any]:
    """Fetch a paper by DOI and return best-effort normalized metadata."""
    paper = get_paper_by_doi(doi)
    return paper.model_dump(mode="json") if paper else {}
