from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from qdrant_client.http import models as qm

from ..config import settings
from ..llm import embed
from ..graph.qdrant_store import QdrantStore
from .schemas import DemoExample, DemoTask


def demo_collection_name(task: DemoTask) -> str:
    if task == "temporal_triplets":
        return getattr(settings, "demo_collection_triplets", "demos_temporal_triplets")
    if task == "hypothesis_test":
        return getattr(settings, "demo_collection_hypothesis", "demos_hypothesis_test")
    raise ValueError(f"Unknown demo task: {task}")


def ensure_demo_collection(task: DemoTask, vector_size: Optional[int] = None) -> str:
    """Ensure that the demo collection exists in Qdrant.

    If vector_size is not provided, we infer it from a tiny embedding call.
    """
    name = demo_collection_name(task)
    if vector_size is None:
        vector_size = len(embed(["_demo_dim_probe_"])[0])
    qd = QdrantStore()
    qd.ensure_collection(name, vector_size=vector_size)

    # Optional: payload indexes (best-effort). If qdrant server doesn't support it, ignore.
    try:
        qd.create_payload_index(name, "domain", field_schema=qm.PayloadSchemaType.KEYWORD)
        qd.create_payload_index(name, "quality", field_schema=qm.PayloadSchemaType.KEYWORD)
        qd.create_payload_index(name, "schema_version", field_schema=qm.PayloadSchemaType.KEYWORD)
        qd.create_payload_index(name, "tags", field_schema=qm.PayloadSchemaType.KEYWORD)
    except Exception:
        pass
    return name


def upsert_demos(task: DemoTask, demos: Iterable[DemoExample]) -> int:
    """Embed demo inputs and upsert into the corresponding Qdrant collection."""
    demos_list = list(demos)
    if not demos_list:
        return 0

    texts = [d.input_text() for d in demos_list]
    vectors = embed(texts)
    vector_size = len(vectors[0]) if vectors else 384
    col = ensure_demo_collection(task, vector_size=vector_size)

    ids: List[str] = []
    payloads: List[Dict[str, Any]] = []
    for d in demos_list:
        ids.append(d.id)
        payloads.append(
            {
                "task": d.task,
                "domain": d.domain,
                "schema_version": d.schema_version,
                "quality": d.quality,
                "tags": d.tags,
                "source": d.source.model_dump() if d.source else None,
                "input": d.input,
                "output": d.output,
            }
        )

    qd = QdrantStore()
    qd.upsert(collection=col, ids=ids, vectors=vectors, payloads=payloads)
    return len(ids)


def demo_filter(*, domain: Optional[str] = None, quality: Optional[str] = None, schema_version: Optional[str] = None) -> qm.Filter:
    must: List[qm.FieldCondition] = []
    if domain:
        must.append(qm.FieldCondition(key="domain", match=qm.MatchValue(value=domain)))
    if quality:
        must.append(qm.FieldCondition(key="quality", match=qm.MatchValue(value=quality)))
    if schema_version:
        must.append(qm.FieldCondition(key="schema_version", match=qm.MatchValue(value=schema_version)))
    return qm.Filter(must=must)
