from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import re
import uuid

try:  # pragma: no cover
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qm
except Exception:  # pragma: no cover
    QdrantClient = None  # type: ignore[assignment]
    qm = None  # type: ignore[assignment]

from ..config import settings


TokenPattern = re.compile(r"\w+", re.UNICODE)


@dataclass
class SparseVectorData:
    indices: List[int]
    values: List[float]

    def to_qdrant(self):
        if qm is None:  # pragma: no cover
            raise RuntimeError("qdrant-client is not installed")
        return qm.SparseVector(indices=self.indices, values=self.values)


@dataclass
class QdrantStore:
    url: str | None = None
    api_key: str | None = None

    def __post_init__(self) -> None:
        self.url = self.url if self.url is not None else settings.qdrant_url
        self.api_key = self.api_key if self.api_key is not None else settings.qdrant_api_key
        if QdrantClient is None or qm is None:
            raise RuntimeError("qdrant-client is not installed. Install optional dependencies: pip install qdrant-client")

        u = (self.url or "").strip()
        if u == ":memory:":
            self._client = QdrantClient(":memory:")
            return
        check_compatibility = bool(getattr(settings, "qdrant_check_compatibility", False))
        if u.startswith("http://") or u.startswith("https://"):
            self._client = QdrantClient(url=u, api_key=self.api_key, check_compatibility=check_compatibility)
            return
        self._client = QdrantClient(path=u)

    @staticmethod
    def _normalize_point_id(pid: str | int | uuid.UUID):
        if isinstance(pid, int):
            return pid
        if isinstance(pid, uuid.UUID):
            return str(pid)
        try:
            return str(uuid.UUID(str(pid)))
        except Exception:
            return str(uuid.uuid5(uuid.NAMESPACE_URL, str(pid)))

    @staticmethod
    def hashed_sparse_vector(text: str, *, dim: int | None = None) -> SparseVectorData:
        """Deterministic sparse vector without external dependencies.

        This keeps hybrid retrieval runnable out of the box even when SPLADE/BM25 models are
        not installed. It is intentionally simple but works well enough for smoke tests and local
        demos, while the collection layout is also compatible with stronger sparse encoders later.
        """

        dimension = int(dim or getattr(settings, "qdrant_sparse_dim", 2048) or 2048)
        counts: Dict[int, float] = {}
        for tok in TokenPattern.findall(text.lower()):
            digest = blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest, "little") % max(8, dimension)
            counts[idx] = counts.get(idx, 0.0) + 1.0
        if not counts:
            return SparseVectorData(indices=[], values=[])
        norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
        pairs = sorted((i, v / norm) for i, v in counts.items())
        return SparseVectorData(indices=[i for i, _ in pairs], values=[float(v) for _, v in pairs])

    def ensure_collection(self, name: str, vector_size: int) -> None:
        collections = [c.name for c in self._client.get_collections().collections]
        if name in collections:
            return
        self._client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
        )

    def ensure_hybrid_collection(
        self,
        name: str,
        *,
        dense_vector_size: int,
        dense_vector_name: Optional[str] = None,
        sparse_vector_name: Optional[str] = None,
    ) -> None:
        collections = [c.name for c in self._client.get_collections().collections]
        if name in collections:
            return
        dense_name = dense_vector_name or getattr(settings, "qdrant_dense_vector_name", "dense")
        sparse_name = sparse_vector_name or getattr(settings, "qdrant_sparse_vector_name", "sparse")
        self._client.create_collection(
            collection_name=name,
            vectors_config={dense_name: qm.VectorParams(size=dense_vector_size, distance=qm.Distance.COSINE)},
            sparse_vectors_config={sparse_name: qm.SparseVectorParams()},
        )

    def create_payload_index(self, collection: str, field_name: str, field_schema: qm.PayloadSchemaType) -> None:
        try:
            self._client.create_payload_index(collection_name=collection, field_name=field_name, field_schema=field_schema)
        except Exception:
            return

    def upsert(self, collection: str, ids: List[str], vectors: List[List[float]], payloads: List[Dict[str, Any]]) -> None:
        points = [qm.PointStruct(id=self._normalize_point_id(pid), vector=vec, payload=pl) for pid, vec, pl in zip(ids, vectors, payloads, strict=True)]
        self._client.upsert(collection_name=collection, points=points)

    def upsert_hybrid(
        self,
        collection: str,
        *,
        ids: Sequence[str],
        dense_vectors: Sequence[Sequence[float]],
        sparse_vectors: Sequence[SparseVectorData],
        payloads: Sequence[Dict[str, Any]],
        dense_vector_name: Optional[str] = None,
        sparse_vector_name: Optional[str] = None,
    ) -> None:
        dense_name = dense_vector_name or getattr(settings, "qdrant_dense_vector_name", "dense")
        sparse_name = sparse_vector_name or getattr(settings, "qdrant_sparse_vector_name", "sparse")
        points = []
        for pid, dvec, svec, payload in zip(ids, dense_vectors, sparse_vectors, payloads, strict=True):
            points.append(
                qm.PointStruct(
                    id=self._normalize_point_id(pid),
                    vector={dense_name: list(dvec), sparse_name: svec.to_qdrant()},
                    payload=dict(payload),
                )
            )
        self._client.upsert(collection_name=collection, points=points)

    def search(
        self,
        collection: str,
        query_vector: List[float],
        limit: int = 8,
        query_filter: Optional[qm.Filter] = None,
    ) -> List[Dict[str, Any]]:
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

        res = self._client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        return [{"id": r.id, "score": r.score, "payload": r.payload} for r in res]

    def hybrid_search(
        self,
        collection: str,
        *,
        dense_vector: Sequence[float],
        sparse_vector: SparseVectorData,
        limit: int = 8,
        query_filter: Optional[qm.Filter] = None,
        dense_vector_name: Optional[str] = None,
        sparse_vector_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        dense_name = dense_vector_name or getattr(settings, "qdrant_dense_vector_name", "dense")
        sparse_name = sparse_vector_name or getattr(settings, "qdrant_sparse_vector_name", "sparse")
        if not hasattr(self._client, "query_points"):
            return self.search(collection=collection, query_vector=list(dense_vector), limit=limit, query_filter=query_filter)

        resp = self._client.query_points(
            collection_name=collection,
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            prefetch=[
                qm.Prefetch(query=list(dense_vector), using=dense_name, limit=limit),
                qm.Prefetch(query=sparse_vector.to_qdrant(), using=sparse_name, limit=limit),
            ],
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        points = getattr(resp, "points", resp)
        return [{"id": p.id, "score": p.score, "payload": p.payload} for p in points]
