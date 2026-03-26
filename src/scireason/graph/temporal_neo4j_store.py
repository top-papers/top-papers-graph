from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from typing import Any, Dict, Iterable, Optional

try:  # pragma: no cover
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover
    GraphDatabase = None  # type: ignore[assignment]

from ..config import settings
from ..temporal.schemas import TemporalEvent, TimeInterval, TemporalTriplet


def _assertion_id(paper_id: str, t: TemporalTriplet) -> str:
    key = (
        f"{paper_id}|{t.subject}|{t.predicate}|{t.object}|{t.polarity}|"
        f"{t.time.start if t.time else ''}|{t.time.end if t.time else ''}"
    )
    return sha1(key.encode("utf-8")).hexdigest()[:16]


@dataclass
class Neo4jTemporalStore:
    uri: str | None = None
    user: str | None = None
    password: str | None = None

    def __post_init__(self) -> None:
        self.uri = self.uri if self.uri is not None else settings.neo4j_uri
        self.user = self.user if self.user is not None else settings.neo4j_user
        self.password = self.password if self.password is not None else settings.neo4j_password
        if GraphDatabase is None:
            raise RuntimeError(
                "neo4j python driver is not installed. Install base dependencies with neo4j support enabled."
            )
        self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self) -> None:
        self._driver.close()

    def ensure_schema(self) -> None:
        cypher = [
            "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT time_key IF NOT EXISTS FOR (t:Time) REQUIRE t.key IS UNIQUE",
            "CREATE CONSTRAINT assertion_id IF NOT EXISTS FOR (a:Assertion) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE",
        ]
        with self._driver.session() as s:
            for q in cypher:
                s.run(q)

    def ensure_vector_indexes(
        self,
        *,
        chunk_dimensions: Optional[int] = None,
        assertion_dimensions: Optional[int] = None,
        entity_dimensions: Optional[int] = None,
    ) -> None:
        """Best-effort vector indexes for Neo4j as graph + vector DB.

        Compatible with current Neo4j vector index syntax. Older servers may reject these
        statements; callers should treat failures as non-fatal.
        """

        queries = []
        if chunk_dimensions:
            queries.append(
                (
                    "chunk_embedding_idx",
                    f"CREATE VECTOR INDEX chunk_embedding_idx IF NOT EXISTS "
                    f"FOR (c:Chunk) ON c.embedding "
                    f"OPTIONS {{indexConfig: {{`vector.dimensions`: {int(chunk_dimensions)}, `vector.similarity_function`: 'cosine'}}}}",
                )
            )
        if assertion_dimensions:
            queries.append(
                (
                    "assertion_embedding_idx",
                    f"CREATE VECTOR INDEX assertion_embedding_idx IF NOT EXISTS "
                    f"FOR (a:Assertion) ON a.embedding "
                    f"OPTIONS {{indexConfig: {{`vector.dimensions`: {int(assertion_dimensions)}, `vector.similarity_function`: 'cosine'}}}}",
                )
            )
        if entity_dimensions:
            queries.append(
                (
                    "entity_embedding_idx",
                    f"CREATE VECTOR INDEX entity_embedding_idx IF NOT EXISTS "
                    f"FOR (e:Entity) ON e.embedding "
                    f"OPTIONS {{indexConfig: {{`vector.dimensions`: {int(entity_dimensions)}, `vector.similarity_function`: 'cosine'}}}}",
                )
            )
        with self._driver.session() as s:
            for _, q in queries:
                try:
                    s.run(q)
                except Exception:
                    # Best effort only: older Neo4j versions or restricted deployments may not support vector indexes.
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

    def upsert_chunk(
        self,
        paper_id: str,
        chunk_id: str,
        text: str,
        chunk_index: Optional[int] = None,
        embedding: Optional[list[float]] = None,
    ) -> None:
        """Upsert a text chunk node for provenance and optional vector retrieval."""
        t = (text or "").strip()
        if len(t) > 4000:
            t = t[:4000] + "…"

        q = """
        MATCH (p:Paper {id:$paper_id})
        MERGE (c:Chunk {id:$chunk_id})
        SET c.paper_id=$paper_id,
            c.idx=$chunk_index,
            c.text=$text,
            c.embedding=$embedding
        MERGE (p)-[:HAS_CHUNK]->(c)
        """
        with self._driver.session() as s:
            s.run(
                q,
                paper_id=paper_id,
                chunk_id=chunk_id,
                chunk_index=chunk_index,
                text=t,
                embedding=embedding,
            )

    def upsert_time(self, interval: TimeInterval) -> str:
        start = interval.start or ""
        end = interval.end or start
        key = f"{interval.granularity}:{start}:{end}"

        q = """
        MERGE (t:Time {key: $key})
        SET t.granularity = $granularity,
            t.start = $start,
            t.end = $end
        RETURN t.key as key
        """
        with self._driver.session() as s:
            rec = s.run(q, key=key, granularity=interval.granularity, start=start, end=end).single()
            assert rec
            return rec["key"]

    def upsert_assertion(
        self,
        paper_id: str,
        t: TemporalTriplet,
        chunk_id: Optional[str] = None,
        evidence_quote: Optional[str] = None,
        embedding: Optional[list[float]] = None,
        extraction_method: str = "llm_triplet",
        review_status: str = "pending",
    ) -> str:
        aid = _assertion_id(paper_id, t)
        time_key = None
        if t.time:
            time_key = self.upsert_time(t.time)

        q = """
        MERGE (s:Entity {name: $subj})
        MERGE (o:Entity {name: $obj})
        MATCH (p:Paper {id: $paper_id})
        MERGE (a:Assertion {id: $aid})
        SET a.predicate = $pred,
            a.confidence = $confidence,
            a.polarity = $polarity,
            a.paper_id = $paper_id,
            a.evidence_quote = $evidence_quote,
            a.embedding = $embedding,
            a.extraction_method = $extraction_method,
            a.review_status = $review_status,
            a.object = $obj,
            a.subject = $subj
        MERGE (a)-[:SUBJECT]->(s)
        MERGE (a)-[:OBJECT]->(o)
        MERGE (p)-[:HAS_ASSERTION]->(a)
        MERGE (a)-[:ASSERTED_IN]->(p)
        WITH a
        OPTIONAL MATCH (t:Time {key: $time_key})
        FOREACH (_ IN CASE WHEN t IS NULL THEN [] ELSE [1] END |
            MERGE (a)-[:AT_TIME]->(t)
        )
        RETURN a.id as aid
        """
        with self._driver.session() as s:
            rec = s.run(
                q,
                aid=aid,
                subj=t.subject,
                obj=t.object,
                pred=t.predicate,
                confidence=float(t.confidence),
                polarity=t.polarity,
                paper_id=paper_id,
                time_key=time_key,
                evidence_quote=(evidence_quote or t.evidence_quote or None),
                embedding=embedding,
                extraction_method=extraction_method,
                review_status=review_status,
            ).single()
            assert rec
            assertion_id = rec["aid"]

        if chunk_id:
            q_ev = """
            MATCH (a:Assertion {id:$aid})
            MERGE (c:Chunk {id:$chunk_id})
            MERGE (a)-[e:EVIDENCE]->(c)
            SET e.quote = $quote
            """
            quote = (evidence_quote or t.evidence_quote or "").strip()
            if len(quote) > 200:
                quote = quote[:200] + "…"
            with self._driver.session() as s:
                s.run(q_ev, aid=assertion_id, chunk_id=chunk_id, quote=quote)

        return assertion_id

    def upsert_event(self, event: TemporalEvent) -> str:
        event_id = event.stable_id()
        time_key = self.upsert_time(
            TimeInterval(start=event.ts_start, end=event.ts_end, granularity=event.granularity)
        )
        q = """
        MERGE (s:Entity {name:$subj})
        MERGE (o:Entity {name:$obj})
        MATCH (p:Paper {id:$paper_id})
        MATCH (t:Time {key:$time_key})
        MERGE (e:Event {id:$event_id})
        SET e.paper_id=$paper_id,
            e.chunk_id=$chunk_id,
            e.assertion_id=$assertion_id,
            e.predicate=$pred,
            e.confidence=$confidence,
            e.polarity=$polarity,
            e.ts_start=$ts_start,
            e.ts_end=$ts_end,
            e.granularity=$granularity,
            e.split=$split,
            e.event_type=$event_type,
            e.extraction_method=$extraction_method,
            e.weight=$weight,
            e.evidence_quote=$evidence_quote
        MERGE (e)-[:SOURCE_ENTITY]->(s)
        MERGE (e)-[:TARGET_ENTITY]->(o)
        MERGE (e)-[:FROM_PAPER]->(p)
        MERGE (e)-[:AT_TIME]->(t)
        WITH e
        OPTIONAL MATCH (a:Assertion {id:$assertion_id})
        FOREACH (_ IN CASE WHEN a IS NULL THEN [] ELSE [1] END |
            MERGE (e)-[:ASSERTS]->(a)
        )
        RETURN e.id as event_id
        """
        with self._driver.session() as s:
            rec = s.run(
                q,
                event_id=event_id,
                paper_id=event.paper_id,
                chunk_id=event.chunk_id,
                assertion_id=event.assertion_id,
                subj=event.subject,
                obj=event.object,
                pred=event.predicate,
                confidence=float(event.confidence),
                polarity=event.polarity,
                ts_start=event.ts_start,
                ts_end=event.ts_end,
                granularity=event.granularity,
                split=event.split,
                event_type=event.event_type,
                extraction_method=event.extraction_method,
                weight=float(event.weight),
                evidence_quote=event.evidence_quote,
                time_key=time_key,
            ).single()
            assert rec
            return str(rec["event_id"])

    def export_event_stream(self, limit: int = 1000) -> list[dict[str, Any]]:
        q = """
        MATCH (e:Event)-[:SOURCE_ENTITY]->(s:Entity)
        MATCH (e)-[:TARGET_ENTITY]->(o:Entity)
        OPTIONAL MATCH (e)-[:AT_TIME]->(t:Time)
        RETURN e.id as event_id,
               e.paper_id as paper_id,
               e.chunk_id as chunk_id,
               e.assertion_id as assertion_id,
               s.name as subject,
               e.predicate as predicate,
               o.name as object,
               e.confidence as confidence,
               e.polarity as polarity,
               coalesce(e.ts_start, t.start) as ts_start,
               coalesce(e.ts_end, t.end) as ts_end,
               coalesce(e.granularity, t.granularity) as granularity,
               e.split as split,
               e.event_type as event_type,
               e.extraction_method as extraction_method,
               e.weight as weight,
               e.evidence_quote as evidence_quote
        ORDER BY coalesce(e.ts_start, t.start) ASC, e.id ASC
        LIMIT $limit
        """
        with self._driver.session() as s:
            return [dict(r) for r in s.run(q, limit=int(limit))]

    def search_chunks_by_vector(self, index_name: str, query_embedding: list[float], limit: int = 8) -> list[dict[str, Any]]:
        q = """
        CALL db.index.vector.queryNodes($index_name, $limit, $embedding)
        YIELD node, score
        RETURN node.id as id, node.paper_id as paper_id, node.text as text, score
        ORDER BY score DESC
        """
        with self._driver.session() as s:
            return [dict(r) for r in s.run(q, index_name=index_name, limit=int(limit), embedding=query_embedding)]

    def query_assertions(self, entity: str, time: Optional[TimeInterval] = None, limit: int = 50):
        """Простой поиск утверждений вокруг сущности с (опциональным) фильтром по времени."""
        time_key = None
        if time:
            start = time.start or ""
            end = time.end or start
            time_key = f"{time.granularity}:{start}:{end}"

        q = """
        MATCH (e:Entity {name:$entity})
        MATCH (a:Assertion)-[:SUBJECT|OBJECT]->(e)
        OPTIONAL MATCH (a)-[:AT_TIME]->(t:Time)
        WITH a, t
        WHERE coalesce(a.status, 'active') <> 'replaced'
          AND ($time_key IS NULL OR (t.key = $time_key))
        RETURN a.id as id, a.predicate as predicate, a.confidence as confidence, a.polarity as polarity,
               t.start as t_start, t.end as t_end, t.granularity as granularity
        ORDER BY a.confidence DESC
        LIMIT $limit
        """
        with self._driver.session() as s:
            return [dict(r) for r in s.run(q, entity=entity, time_key=time_key, limit=limit)]

    def apply_expert_override(
        self,
        subj: str,
        pred: str,
        obj: str,
        verdict: str,
        weight: float,
        time_interval: str = "unknown",
        *,
        start_date: str = "unknown",
        end_date: str = "unknown",
        valid_from: str = "unknown",
        valid_to: str = "+inf",
        time_source: str = "unknown",
    ) -> None:
        q = """
        MATCH (a:Assertion)-[:SUBJECT]->(s:Entity {name: $subj})
        MATCH (a)-[:OBJECT]->(o:Entity {name: $obj})
        WHERE a.predicate = $pred
        SET a.expert_verdict = $verdict,
            a.expert_weight = $weight,
            a.expert_time_interval = $time_interval,
            a.expert_start_date = $start_date,
            a.expert_end_date = $end_date,
            a.expert_valid_from = $valid_from,
            a.expert_valid_to = $valid_to,
            a.expert_time_source = $time_source
        """
        with self._driver.session() as s:
            s.run(
                q,
                subj=subj,
                obj=obj,
                pred=pred,
                verdict=verdict,
                weight=float(weight),
                time_interval=time_interval,
                start_date=start_date,
                end_date=end_date,
                valid_from=valid_from,
                valid_to=valid_to,
                time_source=time_source,
            )

    def get_assertion_details(self, assertion_id: str) -> Optional[dict]:
        q = """
        MATCH (p:Paper)-[:HAS_ASSERTION]->(a:Assertion {id:$aid})
        MATCH (a)-[:SUBJECT]->(s:Entity)
        MATCH (a)-[:OBJECT]->(o:Entity)
        OPTIONAL MATCH (a)-[:AT_TIME]->(t:Time)
        RETURN p.id as paper_id,
               s.name as subject,
               a.predicate as predicate,
               o.name as object,
               a.polarity as polarity,
               a.confidence as confidence,
               a.evidence_quote as evidence_quote,
               t.start as t_start,
               t.end as t_end,
               t.granularity as granularity
        """
        with self._driver.session() as s:
            rec = s.run(q, aid=assertion_id).single()
            return dict(rec) if rec else None

    def link_replacement(self, old_id: str, new_id: str, *, rationale: str = "", reviewer_id: str = "") -> None:
        q = """
        MATCH (old:Assertion {id:$old_id})
        MATCH (new:Assertion {id:$new_id})
        MERGE (old)-[r:REPLACED_BY]->(new)
        SET old.status = 'replaced',
            r.rationale = $rationale,
            r.reviewer_id = $reviewer_id
        """
        with self._driver.session() as s:
            s.run(q, old_id=old_id, new_id=new_id, rationale=rationale, reviewer_id=reviewer_id)
