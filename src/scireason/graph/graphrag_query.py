from __future__ import annotations

from typing import List, Dict, Any, Optional
from rich.console import Console

from ..llm import embed
from .qdrant_store import QdrantStore
from .neo4j_store import Neo4jStore

console = Console()


def retrieve_context(collection: str, query: str, limit: int = 8) -> List[Dict[str, Any]]:
    qvec = embed([query])[0]
    qd = QdrantStore()
    return qd.search(collection=collection, query_vector=qvec, limit=limit)


def transitive_evidence(entity_a: str, entity_c: str, max_hops: int = 3) -> List[List[tuple[str, str, str]]]:
    neo = Neo4jStore()
    paths = neo.transitive_paths(entity_a=entity_a, entity_c=entity_c, max_hops=max_hops)
    neo.close()
    return paths
