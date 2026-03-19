from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from typing import Any, Dict, Iterable, List, Optional

try:  # pragma: no cover
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover
    GraphDatabase = None  # type: ignore[assignment]

from ..config import settings
from ..contracts import ChunkRecord, HypothesisArtifact
from ..temporal.schemas import TemporalEvent, TemporalTriplet, TimeInterval



def _assertion_id(paper_id: str, t: TemporalTriplet) -> str:
    raw = (
        f"{paper_id}|{t.subject}|{t.predicate}|{t.object}|{t.polarity}|"
        f"{t.time.start if t.time else ''}|{t.time.end if t.time else ''}"
    )
    return sha1(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class MemgraphTemporalStore:
    uri: str | None = None
    user: str | None = None
    password: str | None = None

    def __post_init__(self) -> None:
        self.uri = self.uri if self.uri is not None else settings.memgraph_uri
        self.user = self.user if self.user is not None else settings.memgraph_user
        self.password = self.password if self.password is not None else settings.memgraph_password
        if GraphDatabase is None:
            raise RuntimeError("neo4j python driver is not installed.")
        auth = None
        if (self.user or "").strip():
            auth = (self.user, self.password)
        self._driver = GraphDatabase.driver(self.uri, auth=auth)

    def close(self) -> None:
        self._driver.close()

    def ensure_schema(self) -> None:
        queries = [
            "CREATE INDEX ON :Paper(id)",
            "CREATE INDEX ON :Chunk(chunk_id)",
            "CREATE INDEX ON :Entity(name)",
            "CREATE INDEX ON :Time(key)",
            "CREATE INDEX ON :Assertion(id)",
            "CREATE INDEX ON :Event(id)",
            "CREATE INDEX ON :Hypothesis(id)",
        ]
        with self._driver.session() as s:
            for q in queries:
                try:
                    s.run(q)
                except Exception:
                    continue

    def upsert_paper(self, paper: Dict[str, Any]) -> None:
        q = """
        MERGE (p:Paper {id: $id})
        SET p.title = $title,
            p.year = $year,
            p.source = $source,
            p.url = $url
        """
        with self._driver.session() as s:
            s.run(q, **paper)

    def upsert_chunk(self, chunk: ChunkRecord) -> None:
        q = """
        MATCH (p:Paper {id:$paper_id})
        MERGE (c:Chunk {chunk_id:$chunk_id})
        SET c.text=$text,
            c.page=$page,
            c.bbox=$bbox,
            c.modality=$modality,
            c.source_backend=$source_backend,
            c.reading_order=$reading_order,
            c.lang=$lang,
            c.table_html=$table_html,
            c.table_md=$table_md,
            c.image_path=$image_path,
            c.metadata=$metadata
        MERGE (p)-[:HAS_CHUNK]->(c)
        """
        with self._driver.session() as s:
            s.run(
                q,
                paper_id=chunk.paper_id,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                page=chunk.page,
                bbox=chunk.bbox,
                modality=chunk.modality,
                source_backend=chunk.source_backend,
                reading_order=chunk.reading_order,
                lang=chunk.lang,
                table_html=chunk.table_html,
                table_md=chunk.table_md,
                image_path=chunk.image_path,
                metadata=chunk.metadata,
            )

    def upsert_time(self, interval: TimeInterval) -> str:
        start = interval.start or ""
        end = interval.end or start
        key = f"{interval.granularity}:{start}:{end}"
        q = """
        MERGE (t:Time {key:$key})
        SET t.start=$start, t.end=$end, t.granularity=$granularity
        RETURN t.key AS key
        """
        with self._driver.session() as s:
            rec = s.run(q, key=key, start=start, end=end, granularity=interval.granularity).single()
            return str(rec["key"]) if rec else key

    def upsert_assertion(
        self,
        *,
        paper_id: str,
        triplet: TemporalTriplet,
        chunk_id: Optional[str] = None,
        extraction_method: str = "llm_triplet",
    ) -> str:
        aid = _assertion_id(paper_id, triplet)
        time_key = None
        if triplet.time:
            time_key = self.upsert_time(triplet.time)
        q = """
        MATCH (p:Paper {id:$paper_id})
        MERGE (s:Entity {name:$subject})
        MERGE (o:Entity {name:$object})
        MERGE (a:Assertion {id:$aid})
        SET a.predicate=$predicate,
            a.confidence=$confidence,
            a.polarity=$polarity,
            a.evidence_quote=$evidence_quote,
            a.paper_id=$paper_id,
            a.extraction_method=$extraction_method
        MERGE (a)-[:SUBJECT]->(s)
        MERGE (a)-[:OBJECT]->(o)
        MERGE (p)-[:HAS_ASSERTION]->(a)
        WITH a
        OPTIONAL MATCH (t:Time {key:$time_key})
        FOREACH (_ IN CASE WHEN t IS NULL THEN [] ELSE [1] END | MERGE (a)-[:AT_TIME]->(t))
        RETURN a.id AS aid
        """
        with self._driver.session() as s:
            rec = s.run(
                q,
                paper_id=paper_id,
                subject=triplet.subject,
                object=triplet.object,
                predicate=triplet.predicate,
                confidence=float(triplet.confidence),
                polarity=triplet.polarity,
                evidence_quote=triplet.evidence_quote,
                extraction_method=extraction_method,
                aid=aid,
                time_key=time_key,
            ).single()
        if chunk_id:
            with self._driver.session() as s:
                s.run(
                    """
                    MATCH (a:Assertion {id:$aid})
                    MATCH (c:Chunk {chunk_id:$chunk_id})
                    MERGE (a)-[:EVIDENCE]->(c)
                    """,
                    aid=aid,
                    chunk_id=chunk_id,
                )
        return str(rec["aid"]) if rec else aid

    def upsert_event(self, event: TemporalEvent) -> str:
        event_id = event.stable_id()
        time_key = self.upsert_time(TimeInterval(start=event.ts_start, end=event.ts_end, granularity=event.granularity))
        q = """
        MATCH (p:Paper {id:$paper_id})
        MERGE (s:Entity {name:$subject})
        MERGE (o:Entity {name:$object})
        MATCH (t:Time {key:$time_key})
        MERGE (e:Event {id:$event_id})
        SET e.predicate=$predicate,
            e.confidence=$confidence,
            e.polarity=$polarity,
            e.ts_start=$ts_start,
            e.ts_end=$ts_end,
            e.granularity=$granularity,
            e.weight=$weight,
            e.event_type=$event_type,
            e.extraction_method=$extraction_method,
            e.chunk_id=$chunk_id,
            e.assertion_id=$assertion_id,
            e.evidence_quote=$evidence_quote
        MERGE (e)-[:SOURCE_ENTITY]->(s)
        MERGE (e)-[:TARGET_ENTITY]->(o)
        MERGE (e)-[:FROM_PAPER]->(p)
        MERGE (e)-[:AT_TIME]->(t)
        RETURN e.id AS event_id
        """
        with self._driver.session() as s:
            rec = s.run(
                q,
                paper_id=event.paper_id,
                subject=event.subject,
                object=event.object,
                predicate=event.predicate,
                confidence=float(event.confidence),
                polarity=event.polarity,
                ts_start=event.ts_start,
                ts_end=event.ts_end,
                granularity=event.granularity,
                weight=float(event.weight),
                event_type=event.event_type,
                extraction_method=event.extraction_method,
                chunk_id=event.chunk_id,
                assertion_id=event.assertion_id,
                evidence_quote=event.evidence_quote,
                event_id=event_id,
                time_key=time_key,
            ).single()
        return str(rec["event_id"]) if rec else event_id

    def upsert_hypothesis(self, hypothesis_id: str, artifact: HypothesisArtifact) -> None:
        q = """
        MERGE (h:Hypothesis {id:$id})
        SET h.title=$title,
            h.premise=$premise,
            h.mechanism=$mechanism,
            h.time_scope=$time_scope,
            h.proposed_experiment=$proposed_experiment,
            h.confidence_score=$confidence_score,
            h.candidate_kind=$candidate_kind,
            h.source_term=$source_term,
            h.target_term=$target_term,
            h.predicate=$predicate,
            h.graph_signals=$graph_signals
        """
        with self._driver.session() as s:
            s.run(
                q,
                id=hypothesis_id,
                title=artifact.hypothesis.title,
                premise=artifact.hypothesis.premise,
                mechanism=artifact.hypothesis.mechanism,
                time_scope=artifact.time_scope or artifact.hypothesis.time_scope,
                proposed_experiment=artifact.hypothesis.proposed_experiment,
                confidence_score=int(artifact.hypothesis.confidence_score),
                candidate_kind=artifact.candidate_kind,
                source_term=artifact.source_term,
                target_term=artifact.target_term,
                predicate=artifact.predicate,
                graph_signals=artifact.graph_signals,
            )

    def run_mage_analytics(self, *, limit: int = 20) -> Dict[str, List[Dict[str, Any]]]:
        """Run a small analytics snapshot with MAGE procedures if present.

        Queries follow the documented MAGE procedure names such as `pagerank.get()`,
        `betweenness_centrality.get()` and `community_detection.get()`. Failures are non-fatal.
        """

        snapshots: Dict[str, List[Dict[str, Any]]] = {}
        query_specs = {
            "pagerank": "CALL pagerank.get() YIELD node, rank RETURN coalesce(node.name, node.id, node.chunk_id) AS node_id, rank ORDER BY rank DESC LIMIT $limit",
            "betweenness": "CALL betweenness_centrality.get(TRUE, TRUE) YIELD node, betweenness_centrality RETURN coalesce(node.name, node.id, node.chunk_id) AS node_id, betweenness_centrality ORDER BY betweenness_centrality DESC LIMIT $limit",
            "communities": "CALL community_detection.get() YIELD node, community_id RETURN coalesce(node.name, node.id, node.chunk_id) AS node_id, community_id LIMIT $limit",
        }
        with self._driver.session() as s:
            for name, query in query_specs.items():
                try:
                    rows = s.run(query, limit=limit)
                    snapshots[name] = [dict(r) for r in rows]
                except Exception:
                    snapshots[name] = []
        return snapshots
