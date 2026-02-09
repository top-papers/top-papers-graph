from __future__ import annotations

from typing import List, Optional

try:
    from fastapi import FastAPI, HTTPException, Query
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "FastAPI extra is not installed. Install with: pip install -e '.[api]'"
    ) from e

from scireason.papers import PaperMetadata, PaperSource, get_paper_by_doi, resolve_ids, search_papers


def create_app() -> FastAPI:
    app = FastAPI(
        title="top-papers-graph API",
        version="0.2.0",
        description="Unified paper metadata API with normalization, caching, rate-limits, and id resolution.",
    )

    @app.get("/v1/search", response_model=List[PaperMetadata])
    def v1_search(
        q: str = Query(..., min_length=1, description="Free-text query"),
        limit: int = Query(25, ge=1, le=200),
        sources: Optional[List[PaperSource]] = Query(None, description="Subset of sources"),
        with_abstracts: bool = Query(False, description="Fetch abstracts (only for sources that support it)"),
    ) -> List[PaperMetadata]:
        return search_papers(q, limit=limit, sources=sources, with_abstracts=with_abstracts)

    @app.get("/v1/resolve")
    def v1_resolve(id: str = Query(..., description="doi:<...> | pmid:<...> | arxiv:<...> | openalex:<...>")):
        return resolve_ids(id)

    @app.get("/v1/paper/by-doi", response_model=PaperMetadata)
    def v1_paper_by_doi(doi: str = Query(..., min_length=3)):
        p = get_paper_by_doi(doi)
        if not p:
            raise HTTPException(status_code=404, detail="Not found")
        return p

    return app


app = create_app()
