from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Dict, Iterable, List, Optional
import json

import networkx as nx

try:  # pragma: no cover
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover
    GraphDatabase = None  # type: ignore[assignment]

from ..config import settings
from ..contracts import ChunkRecord, HypothesisArtifact
from ..temporal.schemas import TemporalEvent, TemporalTriplet, TimeInterval


LOCAL_STATE_KEYS = ('papers', 'chunks', 'entities', 'times', 'assertions', 'events', 'hypotheses')


def _assertion_id(paper_id: str, t: TemporalTriplet) -> str:
    raw = (
        f"{paper_id}|{t.subject}|{t.predicate}|{t.object}|{t.polarity}|"
        f"{t.time.start if t.time else ''}|{t.time.end if t.time else ''}"
    )
    return sha1(raw.encode('utf-8')).hexdigest()[:16]


class _LocalResult:
    def __init__(self, rows: Iterable[Dict[str, Any]]) -> None:
        self._rows = [dict(r) for r in rows]

    def __iter__(self):
        return iter(self._rows)

    def single(self) -> Optional[Dict[str, Any]]:
        return self._rows[0] if self._rows else None


class _LocalSession:
    def __init__(self, store: 'MemgraphTemporalStore') -> None:
        self._store = store

    def __enter__(self) -> '_LocalSession':
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def run(self, query: str, **params: Any) -> _LocalResult:
        return self._store._run_local_query(query, **params)


class _LocalDriver:
    def __init__(self, store: 'MemgraphTemporalStore') -> None:
        self._store = store

    def session(self) -> _LocalSession:
        return _LocalSession(self._store)

    def close(self) -> None:
        self._store._persist_local_state()


@dataclass
class MemgraphTemporalStore:
    uri: str | None = None
    user: str | None = None
    password: str | None = None

    def __post_init__(self) -> None:
        self.uri = self.uri if self.uri is not None else settings.memgraph_uri
        self.user = self.user if self.user is not None else settings.memgraph_user
        self.password = self.password if self.password is not None else settings.memgraph_password
        self._local_mode = self._is_local_uri(self.uri)
        if self._local_mode:
            self._local_state_path = self._resolve_local_state_path(self.uri)
            self._local_state = self._load_local_state(self._local_state_path)
            self._driver = _LocalDriver(self)
            return

        if GraphDatabase is None:
            raise RuntimeError('neo4j python driver is not installed.')
        auth = None
        if (self.user or '').strip():
            auth = (self.user, self.password)
        self._driver = GraphDatabase.driver(self.uri, auth=auth)

    @staticmethod
    def _is_local_uri(uri: Optional[str]) -> bool:
        u = (uri or '').strip()
        return u in {':memory:', 'memory://'} or u.startswith('file://') or u.startswith('local://')

    @staticmethod
    def _resolve_local_state_path(uri: Optional[str]) -> Optional[Path]:
        u = (uri or '').strip()
        if u in {':memory:', 'memory://'}:
            return None
        if u.startswith('file://'):
            raw = u[len('file://'):]
        elif u.startswith('local://'):
            raw = u[len('local://'):]
        else:
            raw = u
        raw = raw or str(Path(gettempdir()) / 'memgraph_local_state.json')
        return Path(raw).expanduser().resolve()

    @staticmethod
    def _empty_local_state() -> Dict[str, Dict[str, Any]]:
        return {k: {} for k in LOCAL_STATE_KEYS}

    def _load_local_state(self, path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
        state = self._empty_local_state()
        if path and path.exists():
            try:
                raw = json.loads(path.read_text(encoding='utf-8'))
                if isinstance(raw, dict):
                    for key in LOCAL_STATE_KEYS:
                        val = raw.get(key)
                        if isinstance(val, dict):
                            state[key] = val
            except Exception:
                pass
        return state

    def _persist_local_state(self) -> None:
        if not getattr(self, '_local_mode', False):
            return
        path = getattr(self, '_local_state_path', None)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._local_state, ensure_ascii=False, indent=2), encoding='utf-8')

    def close(self) -> None:
        self._driver.close()

    def ensure_schema(self) -> None:
        if self._local_mode:
            self._persist_local_state()
            return
        queries = [
            'CREATE INDEX ON :Paper(id)',
            'CREATE INDEX ON :Chunk(chunk_id)',
            'CREATE INDEX ON :Entity(name)',
            'CREATE INDEX ON :Time(key)',
            'CREATE INDEX ON :Assertion(id)',
            'CREATE INDEX ON :Event(id)',
            'CREATE INDEX ON :Hypothesis(id)',
        ]
        with self._driver.session() as s:
            for q in queries:
                try:
                    s.run(q)
                except Exception:
                    continue

    def upsert_paper(self, paper: Dict[str, Any]) -> None:
        payload = {
            'id': paper.get('id'),
            'title': paper.get('title'),
            'year': paper.get('year'),
            'source': paper.get('source'),
            'url': paper.get('url'),
            'country_id': paper.get('country_id'),
            'country_label': paper.get('country_label'),
            'city_id': paper.get('city_id'),
            'city_label': paper.get('city_label'),
            'science_branch_ids': paper.get('science_branch_ids'),
            'science_branch_labels': paper.get('science_branch_labels'),
        }
        if self._local_mode:
            pid = str(payload.get('id') or '')
            if pid:
                self._local_state['papers'][pid] = payload
                self._persist_local_state()
            return

        q = """
        MERGE (p:Paper {id: $id})
        SET p.title = $title,
            p.year = $year,
            p.source = $source,
            p.url = $url,
            p.country_id = $country_id,
            p.country_label = $country_label,
            p.city_id = $city_id,
            p.city_label = $city_label,
            p.science_branch_ids = $science_branch_ids,
            p.science_branch_labels = $science_branch_labels
        """
        with self._driver.session() as s:
            s.run(q, **payload)

    def upsert_chunk(self, chunk: ChunkRecord) -> None:
        if self._local_mode:
            payload = chunk.model_dump(mode='json') if hasattr(chunk, 'model_dump') else dict(chunk.__dict__)
            self._local_state['chunks'][str(chunk.chunk_id)] = payload
            self._persist_local_state()
            return

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
        start = interval.start or ''
        end = interval.end or start
        key = f'{interval.granularity}:{start}:{end}'
        if self._local_mode:
            self._local_state['times'][key] = {'key': key, 'start': start, 'end': end, 'granularity': interval.granularity}
            self._persist_local_state()
            return key

        q = """
        MERGE (t:Time {key:$key})
        SET t.start=$start, t.end=$end, t.granularity=$granularity
        RETURN t.key AS key
        """
        with self._driver.session() as s:
            rec = s.run(q, key=key, start=start, end=end, granularity=interval.granularity).single()
            return str(rec['key']) if rec else key

    def upsert_assertion(
        self,
        *,
        paper_id: str,
        triplet: TemporalTriplet,
        chunk_id: Optional[str] = None,
        extraction_method: str = 'llm_triplet',
    ) -> str:
        aid = _assertion_id(paper_id, triplet)
        time_key = None
        if triplet.time:
            time_key = self.upsert_time(triplet.time)
        if self._local_mode:
            self._local_state['entities'].setdefault(triplet.subject, {'name': triplet.subject})
            self._local_state['entities'].setdefault(triplet.object, {'name': triplet.object})
            self._local_state['assertions'][aid] = {
                'id': aid,
                'paper_id': paper_id,
                'chunk_id': chunk_id,
                'subject': triplet.subject,
                'predicate': triplet.predicate,
                'object': triplet.object,
                'confidence': float(triplet.confidence),
                'polarity': triplet.polarity,
                'evidence_quote': triplet.evidence_quote,
                'time_key': time_key,
                'extraction_method': extraction_method,
            }
            self._persist_local_state()
            return aid

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
        return str(rec['aid']) if rec else aid

    def upsert_event(self, event: TemporalEvent) -> str:
        event_id = event.stable_id()
        time_key = self.upsert_time(TimeInterval(start=event.ts_start, end=event.ts_end, granularity=event.granularity))
        if self._local_mode:
            self._local_state['entities'].setdefault(event.subject, {'name': event.subject})
            self._local_state['entities'].setdefault(event.object, {'name': event.object})
            self._local_state['events'][event_id] = {
                'id': event_id,
                'paper_id': event.paper_id,
                'chunk_id': event.chunk_id,
                'assertion_id': event.assertion_id,
                'subject': event.subject,
                'predicate': event.predicate,
                'object': event.object,
                'confidence': float(event.confidence),
                'polarity': event.polarity,
                'ts_start': event.ts_start,
                'ts_end': event.ts_end,
                'granularity': event.granularity,
                'weight': float(event.weight),
                'event_type': event.event_type,
                'extraction_method': event.extraction_method,
                'evidence_quote': event.evidence_quote,
                'time_key': time_key,
            }
            self._persist_local_state()
            return event_id

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
        return str(rec['event_id']) if rec else event_id

    def upsert_hypothesis(self, hypothesis_id: str, artifact: HypothesisArtifact) -> None:
        if self._local_mode:
            self._local_state['hypotheses'][hypothesis_id] = {
                'id': hypothesis_id,
                'artifact': artifact.model_dump(mode='json') if hasattr(artifact, 'model_dump') else {},
            }
            self._persist_local_state()
            return

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

    def _analytics_from_local_state(self, *, limit: int = 20) -> Dict[str, List[Dict[str, Any]]]:
        g = nx.DiGraph()
        for rec in self._local_state['assertions'].values():
            s = str(rec.get('subject') or '').strip()
            o = str(rec.get('object') or '').strip()
            if s and o:
                g.add_edge(s, o)
        for rec in self._local_state['events'].values():
            s = str(rec.get('subject') or '').strip()
            o = str(rec.get('object') or '').strip()
            if s and o:
                g.add_edge(s, o)

        if g.number_of_nodes() == 0:
            return {'pagerank': [], 'betweenness': [], 'communities': []}

        try:
            pr = nx.pagerank(g)
        except Exception:
            pr = {n: 0.0 for n in g.nodes}
        try:
            bet = nx.betweenness_centrality(g.to_undirected())
        except Exception:
            bet = {n: 0.0 for n in g.nodes}
        comps = list(nx.connected_components(g.to_undirected()))
        community_map: Dict[str, int] = {}
        for cid, comp in enumerate(comps):
            for node in comp:
                community_map[str(node)] = cid

        pagerank_rows = [
            {'node_id': str(node), 'rank': float(score)}
            for node, score in sorted(pr.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        ]
        betweenness_rows = [
            {'node_id': str(node), 'betweenness_centrality': float(score)}
            for node, score in sorted(bet.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        ]
        communities_rows = [
            {'node_id': str(node), 'community_id': int(community_map[str(node)])}
            for node in list(g.nodes)[:limit]
        ]
        return {
            'pagerank': pagerank_rows,
            'betweenness': betweenness_rows,
            'communities': communities_rows,
        }

    def _run_local_query(self, query: str, **params: Any) -> _LocalResult:
        q = ' '.join((query or '').split())
        if q == 'MATCH (n:Paper) RETURN count(n) AS c':
            return _LocalResult([{'c': len(self._local_state['papers'])}])
        if q == 'MATCH (n:Chunk) RETURN count(n) AS c':
            return _LocalResult([{'c': len(self._local_state['chunks'])}])
        if q == 'MATCH (n:Assertion) RETURN count(n) AS c':
            return _LocalResult([{'c': len(self._local_state['assertions'])}])
        if q == 'MATCH (n:Event) RETURN count(n) AS c':
            return _LocalResult([{'c': len(self._local_state['events'])}])
        limit = int(params.get('limit', 20) or 20)
        analytics = self._analytics_from_local_state(limit=limit)
        if q.startswith('CALL pagerank.get()'):
            return _LocalResult(analytics['pagerank'])
        if q.startswith('CALL betweenness_centrality.get'):
            return _LocalResult(analytics['betweenness'])
        if q.startswith('CALL community_detection.get()'):
            return _LocalResult(analytics['communities'])
        return _LocalResult([])

    def run_mage_analytics(self, *, limit: int = 20) -> Dict[str, List[Dict[str, Any]]]:
        """Run a small analytics snapshot with MAGE procedures if present.

        Queries follow the documented MAGE procedure names such as `pagerank.get()`,
        `betweenness_centrality.get()` and `community_detection.get()`. Failures are non-fatal.
        """
        if self._local_mode:
            return self._analytics_from_local_state(limit=limit)

        snapshots: Dict[str, List[Dict[str, Any]]] = {}
        query_specs = {
            'pagerank': 'CALL pagerank.get() YIELD node, rank RETURN coalesce(node.name, node.id, node.chunk_id) AS node_id, rank ORDER BY rank DESC LIMIT $limit',
            'betweenness': 'CALL betweenness_centrality.get(TRUE, TRUE) YIELD node, betweenness_centrality RETURN coalesce(node.name, node.id, node.chunk_id) AS node_id, betweenness_centrality ORDER BY betweenness_centrality DESC LIMIT $limit',
            'communities': 'CALL community_detection.get() YIELD node, community_id RETURN coalesce(node.name, node.id, node.chunk_id) AS node_id, community_id LIMIT $limit',
        }
        with self._driver.session() as s:
            for name, query in query_specs.items():
                try:
                    rows = s.run(query, limit=limit)
                    snapshots[name] = [dict(r) for r in rows]
                except Exception:
                    snapshots[name] = []
        return snapshots
