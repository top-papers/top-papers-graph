from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from ..config import settings


@dataclass
class QdrantStore:
    url: str = settings.qdrant_url
    api_key: str | None = settings.qdrant_api_key

    def __post_init__(self) -> None:
        """Create a Qdrant client.

        Supports both remote (HTTP) and local Qdrant modes:
        - ":memory:" for in-memory store
        - a filesystem path for a persistent local store

        Local mode is handy for CI/unit tests and quick smoke checks.
        """

        u = (self.url or "").strip()
        if u == ":memory:":
            self._client = QdrantClient(":memory:")
            return

        if u.startswith("http://") or u.startswith("https://"):
            self._client = QdrantClient(url=u, api_key=self.api_key)
            return

        # Treat anything else as a local path
        self._client = QdrantClient(path=u)

    def ensure_collection(self, name: str, vector_size: int) -> None:
        collections = [c.name for c in self._client.get_collections().collections]
        if name in collections:
            return
        self._client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
        )

    def create_payload_index(self, collection: str, field_name: str, field_schema: qm.PayloadSchemaType) -> None:
        """Best-effort payload index creation (useful for filtered demo search)."""
        try:
            self._client.create_payload_index(
                collection_name=collection,
                field_name=field_name,
                field_schema=field_schema,
            )
        except Exception:
            # Qdrant might already have it, or server might not support schema type (older versions)
            return

    def upsert(self, collection: str, ids: List[str], vectors: List[List[float]], payloads: List[Dict[str, Any]]) -> None:
        def _normalize_point_id(pid: str | int | uuid.UUID):
            """Normalize a point id for Qdrant.

            In qdrant-client, PointId is typically `int | str`.
            Our pipeline sometimes uses human-readable string ids (e.g. "paper:12").
            For non-UUID strings we deterministically map them to a UUIDv5 and store it
            as a string.
            """

            if isinstance(pid, int):
                return pid
            if isinstance(pid, uuid.UUID):
                return str(pid)
            try:
                return str(uuid.UUID(str(pid)))
            except Exception:
                return str(uuid.uuid5(uuid.NAMESPACE_URL, str(pid)))

        points = [
            qm.PointStruct(id=_normalize_point_id(pid), vector=vec, payload=pl)
            for pid, vec, pl in zip(ids, vectors, payloads, strict=True)
        ]
        self._client.upsert(collection_name=collection, points=points)

    def search(
        self,
        collection: str,
        query_vector: List[float],
        limit: int = 8,
        query_filter: Optional[qm.Filter] = None,
    ) -> List[Dict[str, Any]]:
        """Vector search.

        qdrant-client >= 1.16 removed the legacy `.search(...)` helper in favor of the
        unified `.query_points(...)` endpoint. We support both for compatibility.
        """

        # New API (qdrant-client 1.16+)
        if hasattr(self._client, "query_points"):
            resp = self._client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
            )
            points = getattr(resp, "points", resp)
            return [{"id": p.id, "score": p.score, "payload": p.payload} for p in points]

        # Legacy API (older qdrant-client)
        res = self._client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        return [{"id": r.id, "score": r.score, "payload": r.payload} for r in res]
