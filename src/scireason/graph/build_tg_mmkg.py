from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
from rich.console import Console

from ..config import settings
from ..llm import embed
from ..temporal.schemas import TemporalEvent
from ..temporal.temporal_triplet_extractor import extract_temporal_triplets
from ..mm.mm_embed import embed_text as mm_embed_text, embed_images as mm_embed_images
from .qdrant_store import QdrantStore
from .neo4j_store import Neo4jStore
from .temporal_neo4j_store import Neo4jTemporalStore
from .mm_neo4j_store import Neo4jMMStore

console = Console()


def build_temporal_and_multimodal(
    paper_dir: Path,
    collection_text: str,
    collection_mm: Optional[str] = None,
    domain: str = "Science",
    max_chunks_for_triplets: int = 16,
) -> None:
    """Build:
    1) текстовые эмбеддинги (обычный RAG) -> Qdrant
    2) (опционально) мультимодальные эмбеддинги (CLIP) -> Qdrant
    3) темпоральные утверждения -> Neo4j (Assertion + Time + Event)
    4) (опционально) страницы/кэпшены -> Neo4j (Page)
    5) (best-effort) chunk/assertion embeddings -> Neo4j vector indexes

    Qdrant is preserved for backward compatibility, but Neo4j now also receives vector-friendly
    embeddings so the repo can operate with a unified graph + vector backend.
    """
    meta = json.loads((paper_dir / "meta.json").read_text(encoding="utf-8"))
    paper_id = meta.get("id") or paper_dir.name
    paper_year = meta.get("year", None)

    # ---------- 1) TEXT VECTOR INDEX ----------
    chunks, chunk_ids = [], []
    for line in (paper_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        chunk_ids.append(rec["chunk_id"])
        chunks.append(rec["text"])

    vectors = embed(chunks)
    vector_size = len(vectors[0]) if vectors else int(getattr(settings, "hash_embed_dim", 384) or 384)

    qd = QdrantStore()
    qd.ensure_collection(collection_text, vector_size=vector_size)
    payloads = [
        {"paper_id": paper_id, "chunk_id": cid, "chunk_index": i, "text": t, "kind": "chunk"}
        for i, (cid, t) in enumerate(zip(chunk_ids, chunks))
    ]
    qd.upsert(collection_text, ids=chunk_ids, vectors=vectors, payloads=payloads)

    # ---------- 2) PAPER NODE ----------
    neo = Neo4jStore()
    neo.ensure_schema()
    neo.upsert_paper(
        {
            "id": paper_id,
            "title": meta.get("title", ""),
            "year": paper_year,
            "source": meta.get("source", ""),
            "url": meta.get("url", ""),
        }
    )
    neo.close()

    # ---------- 3) TEMPORAL ASSERTIONS + EVENTS ----------
    tneo = Neo4jTemporalStore()
    tneo.ensure_schema()
    if bool(getattr(settings, "neo4j_vector_enabled", True)):
        try:
            tneo.ensure_vector_indexes(
                chunk_dimensions=vector_size,
                assertion_dimensions=int(getattr(settings, "neo4j_vector_assertion_dimensions", vector_size) or vector_size),
            )
        except Exception as e:
            console.print(f"[yellow]Neo4j vector indexes skipped: {e}[/yellow]")

    tneo.upsert_paper(
        {
            "id": paper_id,
            "title": meta.get("title", ""),
            "year": paper_year,
            "source": meta.get("source", ""),
            "url": meta.get("url", ""),
        }
    )

    event_counter = 0
    for idx, (cid, t, chunk_vec) in enumerate(zip(chunk_ids, chunks, vectors)):
        if idx >= max_chunks_for_triplets:
            break
        try:
            tneo.upsert_chunk(
                paper_id=paper_id,
                chunk_id=cid,
                text=t,
                chunk_index=idx,
                embedding=list(chunk_vec) if chunk_vec is not None else None,
            )
        except Exception as e:
            console.print(f"[yellow]Chunk upsert failed for {cid}: {e}[/yellow]")

        try:
            triplets = extract_temporal_triplets(domain=domain, chunk_text=t, paper_year=paper_year)
        except Exception as e:
            console.print(f"[yellow]Temporal triplets extraction failed: {e}[/yellow]")
            continue

        triplet_embeddings: list[list[float]] = []
        if triplets:
            try:
                triplet_embeddings = embed([tr.as_text() for tr in triplets])
            except Exception:
                triplet_embeddings = []

        for tr_idx, tr in enumerate(triplets):
            assertion_embedding = None
            if tr_idx < len(triplet_embeddings):
                assertion_embedding = list(triplet_embeddings[tr_idx])
            elif chunk_vec is not None:
                assertion_embedding = list(chunk_vec)

            assertion_id = tneo.upsert_assertion(
                paper_id=paper_id,
                t=tr,
                chunk_id=cid,
                evidence_quote=tr.evidence_quote,
                embedding=assertion_embedding,
                extraction_method="llm_triplet",
                review_status="pending",
            )

            event_counter += 1
            ev = TemporalEvent(
                event_id=f"{paper_id}:event:{event_counter:05d}",
                paper_id=str(paper_id),
                chunk_id=str(cid),
                assertion_id=str(assertion_id),
                subject=tr.subject,
                predicate=tr.predicate,
                object=tr.object,
                ts_start=tr.time.start if tr.time else (str(paper_year) if paper_year else None),
                ts_end=tr.time.end if tr.time else (str(paper_year) if paper_year else None),
                granularity=tr.time.granularity if tr.time else "year",
                confidence=float(tr.confidence),
                polarity=tr.polarity,
                event_type="extracted",
                extraction_method="llm_triplet",
                evidence_quote=tr.evidence_quote,
            )
            try:
                tneo.upsert_event(ev)
            except Exception as e:
                console.print(f"[yellow]Event upsert failed for assertion {assertion_id}: {e}[/yellow]")

    tneo.close()

    # ---------- 4) MULTIMODAL INDEX + PAGE NODES ----------
    pages_path = paper_dir / "mm" / "pages.jsonl"
    if not pages_path.exists():
        console.print("[dim]No multimodal pages.jsonl found — skip multimodal stage.[/dim]")
        return

    mmneo = Neo4jMMStore()
    mmneo.ensure_schema()

    page_recs: List[Dict[str, Any]] = [json.loads(l) for l in pages_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in page_recs:
        mmneo.upsert_page(
            paper_id=paper_id,
            page=int(r["page"]),
            text=r.get("text", ""),
            image_path=r.get("image_path", ""),
            vlm_caption=r.get("vlm_caption", "") or "",
            tables_md=r.get("tables_md", None),
            equations_md=r.get("equations_md", None),
        )
    mmneo.close()

    if not collection_mm:
        console.print("[dim]collection_mm is None — skip multimodal vector index.[/dim]")
        return

    try:
        mm_text_vecs = mm_embed_text([r.get("text", "") for r in page_recs])
        mm_img_vecs = mm_embed_images([Path(r["image_path"]) for r in page_recs])
    except Exception as e:
        console.print(f"[yellow]MM embeddings skipped: {e}[/yellow]")
        return

    dim = len(mm_text_vecs[0]) if mm_text_vecs else 512
    qd.ensure_collection(collection_mm, vector_size=dim)

    ids, vecs, payloads = [], [], []
    for r, v in zip(page_recs, mm_text_vecs):
        pid = f"page_text:{paper_id}:{int(r['page'])}"
        ids.append(pid)
        vecs.append(v)
        payloads.append(
            {"paper_id": paper_id, "page": int(r["page"]), "kind": "page_text", "text": r.get("text","")}
        )

    for r, v in zip(page_recs, mm_img_vecs):
        pid = f"page_img:{paper_id}:{int(r['page'])}"
        ids.append(pid)
        vecs.append(v)
        payloads.append(
            {"paper_id": paper_id, "page": int(r["page"]), "kind": "page_image", "image_path": r.get("image_path","")}
        )

    qd.upsert(collection_mm, ids=ids, vectors=vecs, payloads=payloads)
    console.print(f"[green]MM index updated:[/green] {collection_mm} (points={len(ids)})")
