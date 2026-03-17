from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

from rich.console import Console

from ..config import settings
from ..llm import embed
from ..mm.mm_embed import embed_text as mm_embed_text, embed_images as mm_embed_images
from ..mm.structured_pdf import StructuredChunk, load_structured_chunks
from ..temporal.schemas import TemporalEvent
from ..temporal.temporal_triplet_extractor import extract_temporal_triplets
from .mm_neo4j_store import Neo4jMMStore
from .neo4j_store import Neo4jStore
from .qdrant_store import QdrantStore
from .temporal_neo4j_store import Neo4jTemporalStore

console = Console()


@dataclass
class _ChunkForBuild:
    chunk_id: str
    modality: str
    text: str
    page: Optional[int] = None
    order: int = 0
    image_path: Optional[str] = None
    figure_or_table: Optional[str] = None
    table_markdown: Optional[str] = None
    section: Optional[str] = None
    summary: str = ""
    backend: str = "legacy"
    metadata: Dict[str, Any] | None = None

    def searchable_text(self) -> str:
        parts = [self.text or "", self.summary or "", self.table_markdown or ""]
        return "\n\n".join([p.strip() for p in parts if p and p.strip()]).strip()

    @classmethod
    def from_structured(cls, chunk: StructuredChunk) -> "_ChunkForBuild":
        return cls(
            chunk_id=chunk.chunk_id,
            modality=chunk.modality,
            text=chunk.text,
            page=chunk.page,
            order=chunk.order,
            image_path=chunk.image_path,
            figure_or_table=chunk.figure_or_table,
            table_markdown=chunk.table_markdown,
            section=chunk.section,
            summary=chunk.summary,
            backend=chunk.backend,
            metadata=chunk.metadata,
        )


def _load_chunk_inventory(paper_dir: Path) -> List[_ChunkForBuild]:
    structured = load_structured_chunks(paper_dir)
    if structured:
        out = [_ChunkForBuild.from_structured(c) for c in structured]
        out.sort(key=lambda c: (c.order, c.chunk_id))
        return out

    # Legacy fallback: only plain text chunks.
    out: list[_ChunkForBuild] = []
    chunks_path = paper_dir / "chunks.jsonl"
    if not chunks_path.exists():
        return out
    for idx, line in enumerate(chunks_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        out.append(
            _ChunkForBuild(
                chunk_id=str(rec.get("chunk_id") or f"{paper_dir.name}:{idx}"),
                modality="text",
                text=str(rec.get("text") or ""),
                order=idx,
                backend="legacy",
                metadata={},
            )
        )
    return out


def _truncate(s: str, n: int = 240) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def build_temporal_and_multimodal(
    paper_dir: Path,
    collection_text: str,
    collection_mm: Optional[str] = None,
    domain: str = "Science",
    max_chunks_for_triplets: int = 24,
) -> None:
    """Build an indexed temporal + multimodal evidence graph for a processed paper.

    New expert-pipeline behavior:
    - indexes structured text/table/figure/page chunks into Qdrant
    - stores chunk provenance in Neo4j with modality/page metadata
    - extracts temporal assertions from text/table/figure evidence
    - optionally indexes image-bearing chunks for cross-modal retrieval
    """
    meta = json.loads((paper_dir / "meta.json").read_text(encoding="utf-8"))
    paper_id = str(meta.get("id") or paper_dir.name)
    paper_year = meta.get("year", None)

    chunks = _load_chunk_inventory(paper_dir)
    if not chunks:
        console.print(f"[yellow]No chunks found for {paper_id}; skipping index build.[/yellow]")
        return

    # ---------- 1) TEXT VECTOR INDEX ----------
    search_texts: list[str] = []
    search_chunk_ids: list[str] = []
    search_payloads: list[dict[str, Any]] = []
    for c in chunks:
        text = c.searchable_text()
        if not text:
            continue
        search_texts.append(text)
        search_chunk_ids.append(c.chunk_id)
        search_payloads.append(
            {
                "paper_id": paper_id,
                "chunk_id": c.chunk_id,
                "chunk_index": c.order,
                "kind": "chunk",
                "modality": c.modality,
                "page": c.page,
                "figure_or_table": c.figure_or_table,
                "section": c.section,
                "summary": _truncate(c.summary or text, 400),
                "text": _truncate(text, 2000),
                "image_path": c.image_path,
            }
        )

    vectors = embed(search_texts) if search_texts else []
    vector_size = len(vectors[0]) if vectors else int(getattr(settings, "hash_embed_dim", 384) or 384)

    qd = QdrantStore()
    qd.ensure_collection(collection_text, vector_size=vector_size)
    if vectors:
        qd.upsert(collection_text, ids=search_chunk_ids, vectors=vectors, payloads=search_payloads)

    # ---------- 2) PAPER NODE ----------
    neo = Neo4jStore()
    try:
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
    finally:
        neo.close()

    # ---------- 3) TEMPORAL ASSERTIONS + EVENTS ----------
    tneo = Neo4jTemporalStore()
    tneo.ensure_schema()
    if bool(getattr(settings, "neo4j_vector_enabled", True)):
        try:
            tneo.ensure_vector_indexes(
                chunk_dimensions=vector_size,
                assertion_dimensions=int(
                    getattr(settings, "neo4j_vector_assertion_dimensions", vector_size) or vector_size
                ),
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

    vector_by_chunk = {cid: vec for cid, vec in zip(search_chunk_ids, vectors)}

    for c in chunks:
        text = c.searchable_text()
        emb = vector_by_chunk.get(c.chunk_id)
        try:
            tneo.upsert_chunk(
                paper_id=paper_id,
                chunk_id=c.chunk_id,
                text=text or c.text,
                chunk_index=c.order,
                embedding=list(emb) if emb is not None else None,
                modality=c.modality,
                page=c.page,
                figure_or_table=c.figure_or_table,
                image_path=c.image_path,
                table_markdown=c.table_markdown,
                section=c.section,
                summary=c.summary,
                backend=c.backend,
                metadata=c.metadata or {},
            )
        except Exception as e:
            console.print(f"[yellow]Chunk upsert failed for {c.chunk_id}: {e}[/yellow]")

    candidate_chunks = [
        c for c in chunks if c.modality in {"text", "table", "figure"} and len(c.searchable_text()) >= 40
    ]
    candidate_chunks.sort(key=lambda c: (0 if c.modality == "text" else 1 if c.modality == "table" else 2, c.order))

    event_counter = 0
    selected_for_triplets = candidate_chunks[: max(1, int(max_chunks_for_triplets))]
    for c in selected_for_triplets:
        chunk_text = c.searchable_text()
        if not chunk_text:
            continue
        try:
            triplets = extract_temporal_triplets(domain=domain, chunk_text=chunk_text, paper_year=paper_year)
        except Exception as e:
            console.print(f"[yellow]Temporal triplets extraction failed for {c.chunk_id}: {e}[/yellow]")
            continue

        triplet_embeddings: list[list[float]] = []
        if triplets:
            try:
                triplet_embeddings = embed([tr.as_text() for tr in triplets])
            except Exception:
                triplet_embeddings = []

        evidence_context = []
        if c.page is not None:
            evidence_context.append(f"page {c.page}")
        if c.figure_or_table:
            evidence_context.append(str(c.figure_or_table))
        evidence_prefix = ", ".join(evidence_context)

        for tr_idx, tr in enumerate(triplets):
            assertion_embedding = None
            if tr_idx < len(triplet_embeddings):
                assertion_embedding = list(triplet_embeddings[tr_idx])
            elif c.chunk_id in vector_by_chunk:
                assertion_embedding = list(vector_by_chunk[c.chunk_id])

            if evidence_prefix and not tr.evidence_quote:
                tr.evidence_quote = _truncate(f"{evidence_prefix}: {chunk_text}", 200)

            assertion_id = tneo.upsert_assertion(
                paper_id=paper_id,
                t=tr,
                chunk_id=c.chunk_id,
                evidence_quote=tr.evidence_quote,
                embedding=assertion_embedding,
                extraction_method=f"{c.modality}_llm_triplet",
                review_status="pending",
            )

            event_counter += 1
            ev = TemporalEvent(
                event_id=f"{paper_id}:event:{event_counter:05d}",
                paper_id=str(paper_id),
                chunk_id=str(c.chunk_id),
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
                extraction_method=f"{c.modality}_llm_triplet",
                evidence_quote=tr.evidence_quote,
            )
            try:
                tneo.upsert_event(ev)
            except Exception as e:
                console.print(f"[yellow]Event upsert failed for assertion {assertion_id}: {e}[/yellow]")

    tneo.close()

    # ---------- 4) PAGE / MULTIMODAL NODES ----------
    page_chunks = [c for c in chunks if c.modality == "page"]
    if page_chunks:
        mmneo = Neo4jMMStore()
        try:
            mmneo.ensure_schema()
            for c in page_chunks:
                mmneo.upsert_page(
                    paper_id=paper_id,
                    page=int(c.page or 0),
                    text=c.text,
                    image_path=c.image_path or "",
                    vlm_caption=str((c.metadata or {}).get("vlm_caption") or ""),
                    tables_md=(c.metadata or {}).get("tables_md"),
                    equations_md=(c.metadata or {}).get("equations_md"),
                )
        finally:
            mmneo.close()

    if not collection_mm:
        console.print("[dim]collection_mm is None — skip multimodal vector index.[/dim]")
        return

    image_chunks = [c for c in chunks if c.image_path and Path(str(c.image_path)).exists()]
    if not image_chunks:
        console.print("[dim]No image-bearing chunks found — skip multimodal vector index.[/dim]")
        return

    try:
        mm_text_vecs = mm_embed_text([c.searchable_text() or c.summary or c.modality for c in image_chunks])
        mm_img_vecs = mm_embed_images([Path(str(c.image_path)) for c in image_chunks])
    except Exception as e:
        console.print(f"[yellow]MM embeddings skipped: {e}[/yellow]")
        return

    dim = len(mm_text_vecs[0]) if mm_text_vecs else (len(mm_img_vecs[0]) if mm_img_vecs else 512)
    qd.ensure_collection(collection_mm, vector_size=dim)

    ids: list[str] = []
    vecs: list[list[float]] = []
    payloads: list[dict[str, Any]] = []

    for c, v in zip(image_chunks, mm_text_vecs):
        pid = f"mm_text:{c.chunk_id}"
        ids.append(pid)
        vecs.append(v)
        payloads.append(
            {
                "paper_id": paper_id,
                "chunk_id": c.chunk_id,
                "page": c.page,
                "kind": f"{c.modality}_text",
                "modality": c.modality,
                "figure_or_table": c.figure_or_table,
                "text": _truncate(c.searchable_text(), 2000),
                "image_path": c.image_path,
            }
        )

    for c, v in zip(image_chunks, mm_img_vecs):
        pid = f"mm_img:{c.chunk_id}"
        ids.append(pid)
        vecs.append(v)
        payloads.append(
            {
                "paper_id": paper_id,
                "chunk_id": c.chunk_id,
                "page": c.page,
                "kind": f"{c.modality}_image",
                "modality": c.modality,
                "figure_or_table": c.figure_or_table,
                "summary": _truncate(c.summary or c.searchable_text(), 400),
                "image_path": c.image_path,
            }
        )

    if ids:
        qd.upsert(collection_mm, ids=ids, vectors=vecs, payloads=payloads)
        console.print(f"[green]MM index updated:[/green] {collection_mm} (points={len(ids)})")
