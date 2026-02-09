from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from neo4j import GraphDatabase

from ..config import settings


@dataclass
class Neo4jStore:
    uri: str = settings.neo4j_uri
    user: str = settings.neo4j_user
    password: str = settings.neo4j_password

    def __post_init__(self) -> None:
        self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self) -> None:
        self._driver.close()

    def ensure_schema(self) -> None:
        cypher = [
            "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
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

    def upsert_triplet(self, paper_id: str, subj: str, pred: str, obj: str, confidence: float = 0.6) -> None:
        q = """
        MERGE (s:Entity {name: $subj})
        MERGE (o:Entity {name: $obj})
        MERGE (s)-[r:REL {predicate: $pred}]->(o)
        ON CREATE SET r.confidence = $confidence
        ON MATCH SET r.confidence = max(r.confidence, $confidence)
        WITH s,o
        MATCH (p:Paper {id: $paper_id})
        MERGE (p)-[:MENTIONS]->(s)
        MERGE (p)-[:MENTIONS]->(o)
        """
        with self._driver.session() as s:
            s.run(q, subj=subj, pred=pred, obj=obj, confidence=confidence, paper_id=paper_id)

    def transitive_paths(self, entity_a: str, entity_c: str, max_hops: int = 3) -> List[List[Tuple[str, str, str]]]:
        """Возвращает пути A -> ... -> C как список ребер (subj,pred,obj)."""
        q = """
        MATCH p = (a:Entity {name:$a})-[r:REL*1..$k]->(c:Entity {name:$c})
        RETURN p
        LIMIT 20
        """
        out = []
        with self._driver.session() as s:
            for rec in s.run(q, a=entity_a, c=entity_c, k=max_hops):
                path = rec["p"]
                edges = []
                # path.relationships: list of Relationship
                nodes = list(path.nodes)
                rels = list(path.relationships)
                for i, rel in enumerate(rels):
                    subj = nodes[i].get("name")
                    obj = nodes[i+1].get("name")
                    pred = rel.get("predicate")
                    edges.append((subj, pred, obj))
                out.append(edges)
        return out
