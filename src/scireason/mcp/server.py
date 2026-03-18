from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "MCP extra is not installed. Install with: pip install -e '.[mcp]'"
    ) from e

from scireason.papers import PaperSource, get_paper_by_doi, resolve_ids, search_papers

mcp = FastMCP("top-papers-graph")


@mcp.tool()
def search_papers_tool(
    query: str,
    limit: int = 10,
    sources: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Search scientific papers across multiple sources and return normalized metadata.

    Args:
        query: Free-text query.
        limit: Max results (1..50).
        sources: Optional list like ['semantic_scholar', 'openalex', 'crossref'].
    """
    limit = max(1, min(int(limit), 50))
    srcs = None
    if sources:
        srcs = []
        for s in sources:
            try:
                srcs.append(PaperSource(s))
            except Exception:
                continue
    papers = search_papers(query, limit=limit, sources=srcs)
    return [p.model_dump(mode="json") for p in papers]


@mcp.tool()
def resolve_ids_tool(identifier: str) -> Dict[str, Any]:
    """Resolve between DOI ⇄ PMID ⇄ arXivID ⇄ OpenAlexID (best-effort)."""
    return resolve_ids(identifier).model_dump(mode="json")


@mcp.tool()
def get_paper_by_doi_tool(doi: str) -> Dict[str, Any]:
    """Fetch a paper by DOI (best-effort), returning normalized metadata."""
    p = get_paper_by_doi(doi)
    return p.model_dump(mode="json") if p else {}


def main() -> None:
    # STDIO transport by default; works well for AI integrations.
    mcp.run()


if __name__ == "__main__":
    main()
