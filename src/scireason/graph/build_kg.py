from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json
from rich.console import Console

from ..llm import embed
from .qdrant_store import QdrantStore
from .neo4j_store import Neo4jStore
from .triplet_extractor import extract_triplets

console = Console()


def build_from_paper_dir(paper_dir: Path, collection: str, domain: str = "Science") -> None:
    meta = json.loads((paper_dir / "meta.json").read_text(encoding="utf-8"))
    paper_id = meta.get("id") or paper_dir.name

    # chunks
    chunks = []
    chunk_ids = []
    for line in (paper_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        chunk_ids.append(rec["chunk_id"])
        chunks.append(rec["text"])

    vectors = embed(chunks)
    vector_size = len(vectors[0]) if vectors else 384

    qd = QdrantStore()
    qd.ensure_collection(collection, vector_size=vector_size)
    payloads = [{"paper_id": paper_id, "chunk_id": cid, "text": t} for cid, t in zip(chunk_ids, chunks)]
    qd.upsert(collection, ids=chunk_ids, vectors=vectors, payloads=payloads)

    neo = Neo4jStore()
    neo.ensure_schema()
    neo.upsert_paper(
        {
            "id": paper_id,
            "title": meta.get("title", ""),
            "year": meta.get("year", None),
            "source": meta.get("source", ""),
            "url": meta.get("url", ""),
        }
    )

    # triplets
    for t in chunks[:8]:  # MVP: ограничим количество чанков на бумагу
        try:
            triplets = extract_triplets(domain=domain, chunk_text=t)
        except Exception as e:
            console.print(f"[yellow]Triplets extraction failed: {e}[/yellow]")
            continue
        for tr in triplets:
            neo.upsert_triplet(
                paper_id=paper_id,
                subj=tr.subject,
                pred=tr.predicate,
                obj=tr.object,
                confidence=float(tr.confidence),
            )

    neo.close()
