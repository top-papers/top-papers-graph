from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
from rich.console import Console

from ..config import settings
from ..contracts import ChunkRecord
from ..llm import embed
from ..temporal.schemas import TemporalEvent
from ..temporal.temporal_triplet_extractor import extract_temporal_triplets
from ..mm.mm_embed import embed_text as mm_embed_text, embed_images as mm_embed_images
from .qdrant_store import QdrantStore
from .temporal_neo4j_store import Neo4jTemporalStore
from .mm_neo4j_store import Neo4jMMStore
from .memgraph_store import MemgraphTemporalStore

console = Console()


def _load_chunk_records(paper_dir: Path, paper_id: str) -> List[ChunkRecord]:
    records: List[ChunkRecord] = []
    chunks_path = paper_dir / 'chunks.jsonl'
    for idx, line in enumerate(chunks_path.read_text(encoding='utf-8').splitlines()):
        if not line.strip():
            continue
        raw = json.loads(line)
        if isinstance(raw, dict):
            raw.setdefault('chunk_id', f'{paper_id}:{idx}')
            raw.setdefault('paper_id', paper_id)
            if 'modality' not in raw:
                raw['modality'] = 'text'
            if 'source_backend' not in raw:
                raw['source_backend'] = str((raw.get('metadata') or {}).get('source_backend', 'legacy_text'))
            records.append(ChunkRecord.model_validate(raw))
        else:
            records.append(
                ChunkRecord(
                    chunk_id=f'{paper_id}:{idx}',
                    paper_id=paper_id,
                    text=str(raw),
                    modality='text',
                    source_backend='legacy_text',
                )
            )
    return records


def _payload_from_chunk(rec: ChunkRecord, idx: int) -> Dict[str, Any]:
    payload = rec.to_payload()
    payload['chunk_index'] = idx
    payload['kind'] = 'chunk'
    payload['paper_id'] = rec.paper_id
    return payload


def build_temporal_and_multimodal(
    paper_dir: Path,
    collection_text: str,
    collection_mm: Optional[str] = None,
    domain: str = 'Science',
    max_chunks_for_triplets: int = 16,
) -> None:
    """Build the full temporal + multimodal stack for one processed paper.

    The upgraded flow is:
    1) `ChunkRecord`-aware ingestion from `chunks.jsonl`
    2) Qdrant dense or dense+sparse hybrid indexing
    3) Temporal assertions/events in Neo4j and/or Memgraph
    4) Optional multimodal page nodes + multimodal Qdrant index
    5) Best-effort Memgraph MAGE analytics snapshot
    """
    meta = json.loads((paper_dir / 'meta.json').read_text(encoding='utf-8'))
    paper_id = meta.get('id') or paper_dir.name
    paper_year = meta.get('year', None)

    chunk_records = _load_chunk_records(paper_dir=paper_dir, paper_id=paper_id)
    chunks = [rec.text or '' for rec in chunk_records]
    chunk_ids = [rec.chunk_id for rec in chunk_records]

    # ---------- 1) TEXT VECTOR INDEX ----------
    vectors = embed(chunks)
    vector_size = len(vectors[0]) if vectors else int(getattr(settings, 'hash_embed_dim', 384) or 384)
    qd = QdrantStore()
    payloads = [_payload_from_chunk(rec, i) for i, rec in enumerate(chunk_records)]

    retrieval_mode = str(getattr(settings, 'qdrant_retrieval_mode', 'hybrid') or 'hybrid').lower()
    if retrieval_mode == 'hybrid':
        qd.ensure_hybrid_collection(collection_text, dense_vector_size=vector_size)
        sparse_vectors = [qd.hashed_sparse_vector(t) for t in chunks]
        qd.upsert_hybrid(
            collection_text,
            ids=chunk_ids,
            dense_vectors=vectors,
            sparse_vectors=sparse_vectors,
            payloads=payloads,
        )
    else:
        qd.ensure_collection(collection_text, vector_size=vector_size)
        qd.upsert(collection_text, ids=chunk_ids, vectors=vectors, payloads=payloads)

    # ---------- 2) GRAPH BACKENDS ----------
    graph_backend = str(getattr(settings, 'graph_backend', 'dual') or 'dual').lower()
    use_neo4j = graph_backend in {'dual', 'neo4j'}
    use_memgraph = graph_backend in {'dual', 'memgraph'}

    tneo: Optional[Neo4jTemporalStore] = None
    mem: Optional[MemgraphTemporalStore] = None

    if use_neo4j:
        tneo = Neo4jTemporalStore()
        tneo.ensure_schema()
        if bool(getattr(settings, 'neo4j_vector_enabled', True)):
            try:
                tneo.ensure_vector_indexes(
                    chunk_dimensions=vector_size,
                    assertion_dimensions=int(getattr(settings, 'neo4j_vector_assertion_dimensions', vector_size) or vector_size),
                )
            except Exception as e:
                console.print(f'[yellow]Neo4j vector indexes skipped: {e}[/yellow]')
        tneo.upsert_paper(
            {
                'id': paper_id,
                'title': meta.get('title', ''),
                'year': paper_year,
                'source': meta.get('source', ''),
                'url': meta.get('url', ''),
            }
        )

    if use_memgraph:
        mem = MemgraphTemporalStore()
        mem.ensure_schema()
        mem.upsert_paper(
            {
                'id': paper_id,
                'title': meta.get('title', ''),
                'year': paper_year,
                'source': meta.get('source', ''),
                'url': meta.get('url', ''),
            }
        )

    event_counter = 0
    try:
        for idx, (rec, chunk_vec) in enumerate(zip(chunk_records, vectors)):
            if idx >= max_chunks_for_triplets:
                break

            if tneo is not None:
                try:
                    tneo.upsert_chunk(
                        paper_id=paper_id,
                        chunk_id=rec.chunk_id,
                        text=rec.text,
                        chunk_index=idx,
                        embedding=list(chunk_vec) if chunk_vec is not None else None,
                    )
                except Exception as e:
                    console.print(f'[yellow]Neo4j chunk upsert failed for {rec.chunk_id}: {e}[/yellow]')

            if mem is not None:
                try:
                    mem.upsert_chunk(rec)
                except Exception as e:
                    console.print(f'[yellow]Memgraph chunk upsert failed for {rec.chunk_id}: {e}[/yellow]')

            try:
                triplets = extract_temporal_triplets(domain=domain, chunk_text=rec.text, paper_year=paper_year)
            except Exception as e:
                console.print(f'[yellow]Temporal triplets extraction failed: {e}[/yellow]')
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

                assertion_id: Optional[str] = None
                if tneo is not None:
                    try:
                        assertion_id = tneo.upsert_assertion(
                            paper_id=paper_id,
                            t=tr,
                            chunk_id=rec.chunk_id,
                            evidence_quote=tr.evidence_quote,
                            embedding=assertion_embedding,
                            extraction_method='llm_triplet',
                            review_status='pending',
                        )
                    except Exception as e:
                        console.print(f'[yellow]Neo4j assertion upsert failed: {e}[/yellow]')

                if mem is not None:
                    try:
                        mem_assertion_id = mem.upsert_assertion(
                            paper_id=paper_id,
                            triplet=tr,
                            chunk_id=rec.chunk_id,
                            extraction_method='llm_triplet',
                        )
                        assertion_id = assertion_id or mem_assertion_id
                    except Exception as e:
                        console.print(f'[yellow]Memgraph assertion upsert failed: {e}[/yellow]')

                event_counter += 1
                ev = TemporalEvent(
                    event_id=f'{paper_id}:event:{event_counter:05d}',
                    paper_id=str(paper_id),
                    chunk_id=str(rec.chunk_id),
                    assertion_id=str(assertion_id) if assertion_id else None,
                    subject=tr.subject,
                    predicate=tr.predicate,
                    object=tr.object,
                    ts_start=tr.time.start if tr.time else (str(paper_year) if paper_year else None),
                    ts_end=tr.time.end if tr.time else (str(paper_year) if paper_year else None),
                    granularity=tr.time.granularity if tr.time else 'year',
                    confidence=float(tr.confidence),
                    polarity=tr.polarity,
                    event_type='extracted',
                    extraction_method='llm_triplet',
                    evidence_quote=tr.evidence_quote,
                )
                if tneo is not None:
                    try:
                        tneo.upsert_event(ev)
                    except Exception as e:
                        console.print(f'[yellow]Neo4j event upsert failed for assertion {assertion_id}: {e}[/yellow]')
                if mem is not None:
                    try:
                        mem.upsert_event(ev)
                    except Exception as e:
                        console.print(f'[yellow]Memgraph event upsert failed for assertion {assertion_id}: {e}[/yellow]')
    finally:
        if tneo is not None:
            tneo.close()

    # ---------- 3) OPTIONAL MAGE ANALYTICS SNAPSHOT ----------
    if mem is not None:
        try:
            snapshot = mem.run_mage_analytics(limit=20)
            (paper_dir / 'memgraph_mage_snapshot.json').write_text(
                json.dumps(snapshot, ensure_ascii=False, indent=2),
                encoding='utf-8',
            )
        except Exception as e:
            console.print(f'[yellow]Memgraph MAGE analytics skipped: {e}[/yellow]')
        finally:
            mem.close()

    # ---------- 4) MULTIMODAL INDEX + PAGE NODES ----------
    pages_path = paper_dir / 'mm' / 'pages.jsonl'
    if not pages_path.exists():
        console.print('[dim]No multimodal pages.jsonl found — skip multimodal stage.[/dim]')
        return

    page_recs: List[Dict[str, Any]] = [json.loads(l) for l in pages_path.read_text(encoding='utf-8').splitlines() if l.strip()]

    if use_neo4j:
        mmneo = Neo4jMMStore()
        mmneo.ensure_schema()
        try:
            for r in page_recs:
                mmneo.upsert_page(
                    paper_id=paper_id,
                    page=int(r['page']),
                    text=r.get('text', ''),
                    image_path=r.get('image_path', ''),
                    vlm_caption=r.get('vlm_caption', '') or '',
                    tables_md=r.get('tables_md', None),
                    equations_md=r.get('equations_md', None),
                )
        finally:
            mmneo.close()

    if not collection_mm:
        console.print('[dim]collection_mm is None — skip multimodal vector index.[/dim]')
        return

    try:
        mm_text_vecs = mm_embed_text([r.get('text', '') for r in page_recs])
        mm_img_vecs = mm_embed_images([Path(r['image_path']) for r in page_recs])
    except Exception as e:
        console.print(f'[yellow]MM embeddings skipped: {e}[/yellow]')
        return

    dim = len(mm_text_vecs[0]) if mm_text_vecs else 512
    qd.ensure_collection(collection_mm, vector_size=dim)

    ids, vecs, payloads_mm = [], [], []
    for r, v in zip(page_recs, mm_text_vecs):
        pid = f"page_text:{paper_id}:{int(r['page'])}"
        ids.append(pid)
        vecs.append(v)
        payloads_mm.append(
            {'paper_id': paper_id, 'page': int(r['page']), 'kind': 'page_text', 'text': r.get('text', '')}
        )

    for r, v in zip(page_recs, mm_img_vecs):
        pid = f"page_img:{paper_id}:{int(r['page'])}"
        ids.append(pid)
        vecs.append(v)
        payloads_mm.append(
            {'paper_id': paper_id, 'page': int(r['page']), 'kind': 'page_image', 'image_path': r.get('image_path', '')}
        )

    qd.upsert(collection_mm, ids=ids, vectors=vecs, payloads=payloads_mm)
    console.print(f'[green]MM index updated:[/green] {collection_mm} (points={len(ids)})')
