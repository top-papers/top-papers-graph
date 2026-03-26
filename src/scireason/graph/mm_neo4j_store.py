from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
try:  # pragma: no cover
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover
    GraphDatabase = None  # type: ignore[assignment]

from ..config import settings


@dataclass
class Neo4jMMStore:
    """Хранилище мультимодальных артефактов (страницы/изображения/таблицы) в Neo4j.

    MVP: одна страница = один визуальный объект (рендер страницы).
    Дальше можно детектировать отдельные фигуры/таблицы и заводить отдельные ноды.
    """
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
            "CREATE CONSTRAINT page_key IF NOT EXISTS FOR (pg:Page) REQUIRE pg.key IS UNIQUE",
        ]
        with self._driver.session() as s:
            for q in cypher:
                s.run(q)

    def upsert_page(
        self,
        paper_id: str,
        page: int,
        text: str,
        image_path: str,
        vlm_caption: str = "",
        tables_md: Optional[str] = None,
        equations_md: Optional[str] = None,
    ) -> None:
        key = f"{paper_id}:{page}"
        q = """
        MATCH (p:Paper {id:$paper_id})
        MERGE (pg:Page {key:$key})
        SET pg.paper_id=$paper_id,
            pg.page=$page,
            pg.text=$text,
            pg.image_path=$image_path,
            pg.vlm_caption=$vlm_caption,
            pg.tables_md=$tables_md,
            pg.equations_md=$equations_md
        MERGE (p)-[:HAS_PAGE]->(pg)
        """
        with self._driver.session() as s:
            s.run(
                q,
                paper_id=paper_id,
                key=key,
                page=page,
                text=text,
                image_path=image_path,
                vlm_caption=vlm_caption,
                tables_md=tables_md,
                equations_md=equations_md,
            )
