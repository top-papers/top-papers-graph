from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from ..config import settings


@dataclass
class QdrantStore:
    url: str = settings.qdrant_url
    api_key: str | None = settings.qdrant_api_key

    def __post_init__(self) -> None:
        self._client = QdrantClient(url=self.url, api_key=self.api_key)

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
        points = [
            qm.PointStruct(id=pid, vector=vec, payload=pl)
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
        res = self._client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter,
        )
        return [{"id": r.id, "score": r.score, "payload": r.payload} for r in res]
