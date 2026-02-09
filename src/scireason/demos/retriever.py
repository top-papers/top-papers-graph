from __future__ import annotations

from typing import Any, Dict, List, Optional

from qdrant_client.http import models as qm

from ..config import settings
from ..llm import embed
from ..graph.qdrant_store import QdrantStore
from .schemas import DemoExample, DemoTask
from .store import demo_collection_name, ensure_demo_collection, demo_filter


def retrieve_demos(
    *,
    task: DemoTask,
    domain: str,
    query: str,
    k: int,
    quality: Optional[str] = None,
    schema_version: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve top-k demo examples for a given task + query.

    Returns items of shape:
    {"score": float, "demo": DemoExample, "payload": dict}
    """
    if not query.strip():
        return []

    if quality is None:
        quality = getattr(settings, "demo_quality", "gold")
    if schema_version is None:
        schema_version = getattr(settings, "demo_schema_version", "1.0")

    try:
        qvec = embed([query])[0]
        # ensure collection exists (idempotent)
        ensure_demo_collection(task, vector_size=len(qvec))
        col = demo_collection_name(task)

        qd = QdrantStore()
        flt = demo_filter(domain=domain, quality=quality, schema_version=schema_version)
        res = qd.search(collection=col, query_vector=qvec, limit=max(k, 1) * 2, query_filter=flt)
        out: List[Dict[str, Any]] = []
        for r in res[:k]:
            pl = r.get("payload") or {}
            try:
                demo = DemoExample(
                    id=str(r.get("id")),
                    task=pl.get("task") or task,
                    domain=str(pl.get("domain") or domain),
                    schema_version=str(pl.get("schema_version") or schema_version),
                    quality=str(pl.get("quality") or quality),
                    tags=list(pl.get("tags") or []),
                    source=pl.get("source"),
                    input=dict(pl.get("input") or {}),
                    output=pl.get("output"),
                )
            except Exception:
                # If payload is malformed, skip it
                continue
            out.append({"score": float(r.get("score") or 0.0), "demo": demo, "payload": pl})
        return out
    except Exception:
        return []
