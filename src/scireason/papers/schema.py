from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PaperSource(str, Enum):
    semantic_scholar = "semantic_scholar"
    crossref = "crossref"
    openalex = "openalex"
    pubmed = "pubmed"
    europe_pmc = "europe_pmc"
    biorxiv = "biorxiv"
    medrxiv = "medrxiv"
    arxiv = "arxiv"
    unknown = "unknown"


class Author(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    orcid: Optional[str] = None
    affiliation: Optional[str] = None


class Venue(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: Optional[str] = None
    publisher: Optional[str] = None
    issn_l: Optional[str] = None


class ExternalIds(BaseModel):
    model_config = ConfigDict(extra="ignore")

    doi: Optional[str] = None
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    arxiv: Optional[str] = None
    openalex: Optional[str] = None
    semantic_scholar: Optional[str] = None

    def best_canonical(self) -> Optional[str]:
        for k in ("doi", "pmid", "arxiv", "openalex", "semantic_scholar", "pmcid"):
            v = getattr(self, k, None)
            if v:
                prefix = {
                    "doi": "doi",
                    "pmid": "pmid",
                    "pmcid": "pmc",
                    "arxiv": "arxiv",
                    "openalex": "openalex",
                    "semantic_scholar": "s2",
                }[k]
                return f"{prefix}:{v}"
        return None


class PaperMetadata(BaseModel):
    """Unified paper metadata.

    The goal is to normalize heterogeneous source responses (Crossref/OpenAlex/S2/etc.)
    into a stable schema for downstream graph building.
    """

    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., description="Canonical paper id like doi:<DOI> or pmid:<PMID> or arxiv:<id>.")
    source: PaperSource = PaperSource.unknown

    title: str
    abstract: Optional[str] = None

    authors: List[Author] = Field(default_factory=list)
    venue: Optional[Venue] = None

    year: Optional[int] = None
    published_date: Optional[date] = None

    url: Optional[str] = None
    pdf_url: Optional[str] = None

    citation_count: Optional[int] = None

    ids: ExternalIds = Field(default_factory=ExternalIds)

    raw: Optional[Dict[str, Any]] = Field(default=None, description="Original record for debugging/enrichment.")

    @field_validator("id")
    @classmethod
    def _strip_id(cls, v: str) -> str:
        return (v or "").strip()

    @field_validator("title")
    @classmethod
    def _strip_title(cls, v: str) -> str:
        return (v or "").strip()

    @staticmethod
    def parse_date(value: Any) -> Optional[date]:
        if value is None:
            return None
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if isinstance(value, datetime):
            return value.date()
        s = str(value).strip()
        if not s:
            return None
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y/%m", "%Y"):
            try:
                dt = datetime.strptime(s, fmt)
                return dt.date()
            except Exception:
                continue
        return None

    @classmethod
    def build_canonical_id(cls, ids: ExternalIds, fallback: Optional[str] = None) -> str:
        return ids.best_canonical() or (fallback or "").strip() or "unknown:unknown"
