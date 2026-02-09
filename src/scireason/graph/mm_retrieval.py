from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..mm.mm_embed import embed_text as mm_embed_text
from .qdrant_store import QdrantStore


def retrieve_mm(
    collection_mm: str,
    query: str,
    limit: int = 8,
    kind: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Кросс-модальный ретрив (текст -> (текст/картинки), при условии open_clip backend).

    kind: None|page_text|page_image — позволяет фильтровать выдачу.
    """
    qvec = mm_embed_text([query])[0]
    qd = QdrantStore()
    res = qd.search(collection=collection_mm, query_vector=qvec, limit=limit * 2)
    if kind:
        res = [r for r in res if r.get("payload", {}).get("kind") == kind]
    return res[:limit]
