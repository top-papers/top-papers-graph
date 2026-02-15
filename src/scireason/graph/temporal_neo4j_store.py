from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from typing import Any, Dict, Optional
try:  # pragma: no cover
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover
    GraphDatabase = None  # type: ignore[assignment]

from ..config import settings
from ..temporal.schemas import TimeInterval, TemporalTriplet


def _assertion_id(paper_id: str, t: TemporalTriplet) -> str:
    key = f"{paper_id}|{t.subject}|{t.predicate}|{t.object}|{t.polarity}|{t.time.start if t.time else ''}|{t.time.end if t.time else ''}"
    return sha1(key.encode("utf-8")).hexdigest()[:16]


@dataclass
class Neo4jTemporalStore:
    uri: str = settings.neo4j_uri
    user: str = settings.neo4j_user
    password: str = settings.neo4j_password

    def __post_init__(self) -> None:
        if GraphDatabase is None:
            raise RuntimeError(
                "neo4j python driver is not installed. Install optional dependencies: pip install -e '.[rag]'"
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
        ]
        with self._driver.session() as s:
            for q in cypher:
                s.run(q)

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

    def upsert_chunk(self, paper_id: str, chunk_id: str, text: str, chunk_index: Optional[int] = None) -> None:
        """Upsert a text chunk node for provenance.

        We keep the text truncated to avoid huge Neo4j properties. The full chunk lives in Qdrant.
        """
        t = (text or "").strip()
        if len(t) > 2000:
            t = t[:2000] + "…"

        q = """
        MATCH (p:Paper {id:$paper_id})
        MERGE (c:Chunk {id:$chunk_id})
        SET c.paper_id=$paper_id,
            c.idx=$chunk_index,
            c.text=$text
        MERGE (p)-[:HAS_CHUNK]->(c)
        """
        with self._driver.session() as s:
            s.run(q, paper_id=paper_id, chunk_id=chunk_id, chunk_index=chunk_index, text=t)

    def upsert_time(self, interval: TimeInterval) -> str:
        # MVP: time node = start/end/granularity. В дальнейшем можно строить иерархию year->month->day (TG-RAG).
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
            a.evidence_quote = $evidence_quote
        MERGE (a)-[:SUBJECT]->(s)
        MERGE (a)-[:OBJECT]->(o)
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
            ).single()
            assert rec
            assertion_id = rec["aid"]

        # Optional provenance edge to a chunk node
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
    ) -> None:
        """Tag Assertion nodes with expert verdict/weight (best effort).

        Notes
        -----
        * Matching is done by (subject, predicate, object). In many domains expert `time_interval`
          refers to experimental conditions (T, SOC, chemistry, ...), not to calendar time.
        * We store the expert fields directly on the Assertion node so they can be used by
          retrievers and reward models without rebuilding the graph.
        """
        q = """
        MATCH (a:Assertion)-[:SUBJECT]->(s:Entity {name: $subj})
        MATCH (a)-[:OBJECT]->(o:Entity {name: $obj})
        WHERE a.predicate = $pred
        SET a.expert_verdict = $verdict,
            a.expert_weight = $weight,
            a.expert_time_interval = $time_interval
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
            )

    def get_assertion_details(self, assertion_id: str) -> Optional[dict]:
        """Fetch a minimal assertion record for expert correction workflows."""
        q = """
        MATCH (p:Paper)-[:HAS_ASSERTION]->(a:Assertion {id:$aid})
        MATCH (s:Entity)-[:SUBJECT_OF]->(a)
        OPTIONAL MATCH (a)-[:AT_TIME]->(t:Time)
        RETURN p.id as paper_id,
               s.name as subject,
               a.predicate as predicate,
               a.object as object,
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
        """Mark an assertion as replaced by another one (used for temporal corrections)."""
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
