from __future__ import annotations

"""Temporal knowledge graph builder (term-level).

This module focuses on the *course requirement*:
"from a user query -> automatically build a **temporal** knowledge graph
from multiple papers -> propose testable hypotheses".

We support two extraction backends:
1) `llm_triplets`  - uses the existing TemporalTriplet extractor (best quality)
2) `cooccurrence`  - builds edges from term co-occurrence (no LLM required)

Both backends produce a unified in-memory representation that can be exported to JSON.
"""

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple

from rich.console import Console

from ..domain import DomainConfig
from ..temporal.schemas import TemporalTriplet
from ..temporal.temporal_triplet_extractor import extract_temporal_triplets
from .term_extraction import TermCandidate, extract_terms_rake


console = Console()

EdgeMode = Literal["auto", "llm_triplets", "cooccurrence"]


@dataclass(frozen=True)
class PaperRecord:
    paper_id: str
    title: str
    year: Optional[int]
    text: str
    url: str = ""
    source: str = ""


@dataclass
class NodeStats:
    term: str
    doc_freq: int = 0
    yearly_doc_freq: Dict[int, int] = field(default_factory=dict)


@dataclass
class EdgeStats:
    source: str
    target: str
    predicate: str
    directed: bool = True

    # Aggregates
    total_count: int = 0
    yearly_count: Dict[int, int] = field(default_factory=dict)
    confidence_sum: float = 0.0
    confidence_n: int = 0
    polarity_counts: Dict[str, int] = field(default_factory=lambda: {"supports": 0, "contradicts": 0, "unknown": 0})

    # Provenance
    papers: Set[str] = field(default_factory=set)
    evidence_quotes: List[Dict[str, str]] = field(default_factory=list)  # {paper_id, quote}

    # Derived features / final score
    features: Dict[str, float] = field(default_factory=dict)
    score: float = 0.0

    @property
    def mean_confidence(self) -> float:
        if self.confidence_n <= 0:
            return 0.0
        return float(self.confidence_sum) / float(self.confidence_n)


@dataclass
class TemporalKnowledgeGraph:
    """A lightweight temporal KG representation exportable to JSON."""

    nodes: Dict[str, NodeStats] = field(default_factory=dict)  # key = canonical term
    edges: List[EdgeStats] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "meta": self.meta,
            "nodes": [
                {
                    "term": n.term,
                    "doc_freq": n.doc_freq,
                    "yearly_doc_freq": n.yearly_doc_freq,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "predicate": e.predicate,
                    "directed": e.directed,
                    "total_count": e.total_count,
                    "yearly_count": e.yearly_count,
                    "mean_confidence": e.mean_confidence,
                    "polarity_counts": e.polarity_counts,
                    "papers": sorted(list(e.papers))[:50],
                    "evidence_quotes": e.evidence_quotes[:10],
                    "features": e.features,
                    "score": e.score,
                }
                for e in self.edges
            ],
        }

    def dump_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def _norm_term(t: str) -> str:
    return " ".join(str(t or "").strip().lower().split())


def _year_from_triplet(tr: TemporalTriplet, paper_year: Optional[int]) -> Optional[int]:
    if tr.time and tr.time.start:
        s = str(tr.time.start)
        # Accept YYYY or YYYY-MM or YYYY-MM-DD
        try:
            y = int(s[:4])
            if 1800 <= y <= 2100:
                return y
        except Exception:
            pass
    return paper_year


def load_papers_from_processed(
    processed_dir: Path,
    *,
    max_papers: Optional[int] = None,
    max_chars_per_paper: int = 40_000,
) -> List[PaperRecord]:
    """Load PaperRecord list from `data/processed/papers/<paper_id>/...`."""
    out: List[PaperRecord] = []
    if not processed_dir.exists():
        return out

    for d in sorted([p for p in processed_dir.iterdir() if p.is_dir()]):
        meta_path = d / "meta.json"
        chunks_path = d / "chunks.jsonl"
        if not meta_path.exists() or not chunks_path.exists():
            continue

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        pid = str(meta.get("id") or d.name)
        title = str(meta.get("title") or "").strip()
        year = meta.get("year")
        try:
            year_int = int(year) if year is not None else None
        except Exception:
            year_int = None

        # Join chunks (best-effort, truncated)
        texts: List[str] = []
        total = 0
        for line in chunks_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            t = str(rec.get("text") or "")
            if not t:
                continue
            if total >= max_chars_per_paper:
                break
            remaining = max_chars_per_paper - total
            texts.append(t[:remaining])
            total += len(t[:remaining])

        text = "\n\n".join(texts).strip()
        if not text and meta.get("abstract"):
            text = str(meta.get("abstract") or "").strip()

        out.append(
            PaperRecord(
                paper_id=pid,
                title=title,
                year=year_int,
                text=text,
                url=str(meta.get("url") or ""),
                source=str(meta.get("source") or ""),
            )
        )

        if max_papers and len(out) >= max_papers:
            break

    return out


def _sentences(text: str) -> List[str]:
    # Simple sentence splitter (language-agnostic-ish)
    parts = []
    for p in (text or "").replace("\n", " ").split("."):
        p = p.strip()
        if len(p) >= 20:
            parts.append(p)
    return parts if parts else [text]


def _apply_expert_overrides(edges: Dict[Tuple[str, str, str], EdgeStats], overrides_path: Optional[Path]) -> None:
    """Adjust edge scores using expert graph reviews compiled into JSONL.

    The compiled overrides format is produced by `top-papers-graph refresh-feedback`.
    """

    if not overrides_path or not overrides_path.exists():
        return

    # key: (subj,pred,obj) -> weight
    w: Dict[Tuple[str, str, str], float] = {}
    for line in overrides_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        s = _norm_term(str(rec.get("subject") or ""))
        p = _norm_term(str(rec.get("predicate") or ""))
        o = _norm_term(str(rec.get("object") or ""))
        if not s or not p or not o:
            continue
        try:
            weight = float(rec.get("weight") or 0.0)
        except Exception:
            weight = 0.0
        w[(s, p, o)] = w.get((s, p, o), 0.0) + weight

    if not w:
        return

    # Apply as a feature; the final score will incorporate it.
    for k, edge in edges.items():
        ew = w.get(k)
        if ew is None:
            continue
        edge.features["expert_weight"] = float(ew)


def build_temporal_kg(
    papers: Sequence[PaperRecord],
    *,
    domain: DomainConfig,
    query: str = "",
    edge_mode: EdgeMode = "auto",
    max_terms_per_paper: int = 25,
    recent_window_years: int = 3,
    expert_overrides_path: Optional[Path] = Path("data/derived/expert_overrides.jsonl"),
) -> TemporalKnowledgeGraph:
    """Build a temporal KG from a list of papers."""

    if not papers:
        return TemporalKnowledgeGraph(meta={"domain": domain.domain_id, "query": query, "n_papers": 0})

    # Domain knobs (optional)
    cfg = dict(domain.term_graph or {})  # type: ignore[attr-defined]
    max_terms_per_paper = int(cfg.get("max_terms_per_paper", max_terms_per_paper))
    recent_window_years = int(cfg.get("recent_window_years", recent_window_years))
    language = str(cfg.get("language", "en"))

    nodes: Dict[str, NodeStats] = {}
    edges: Dict[Tuple[str, str, str], EdgeStats] = {}

    # For PMI-like calculations we need doc-level term sets
    paper_term_sets: Dict[str, Set[str]] = {}

    # Decide edge mode
    chosen_mode = edge_mode
    if edge_mode == "auto":
        chosen_mode = "llm_triplets"

    for pr in papers:
        if not pr.text.strip():
            continue
        year = pr.year

        # -------- Strategy 1: LLM temporal triplets --------
        if chosen_mode == "llm_triplets":
            try:
                triplets = extract_temporal_triplets(domain=domain.title, chunk_text=pr.text, paper_year=year)
            except Exception as e:
                console.print(f"[yellow]Temporal triplets failed for {pr.paper_id}: {e}. Falling back to co-occurrence.[/yellow]")
                chosen_mode = "cooccurrence"
                triplets = []

            term_set: Set[str] = set()
            for tr in triplets:
                s = _norm_term(tr.subject)
                o = _norm_term(tr.object)
                p = _norm_term(tr.predicate)
                if not s or not o or not p:
                    continue
                term_set.add(s)
                term_set.add(o)

                y = _year_from_triplet(tr, year)
                if y is None:
                    continue

                key = (s, p, o)
                if key not in edges:
                    edges[key] = EdgeStats(source=s, target=o, predicate=p, directed=True)
                e = edges[key]
                e.total_count += 1
                e.yearly_count[y] = e.yearly_count.get(y, 0) + 1
                e.papers.add(pr.paper_id)
                e.confidence_sum += float(tr.confidence)
                e.confidence_n += 1
                e.polarity_counts[tr.polarity] = e.polarity_counts.get(tr.polarity, 0) + 1
                if tr.evidence_quote:
                    e.evidence_quotes.append({"paper_id": pr.paper_id, "quote": tr.evidence_quote})

            paper_term_sets[pr.paper_id] = term_set

        # -------- Strategy 2: Term co-occurrence (no LLM) --------
        if chosen_mode == "cooccurrence":
            # Use title+text for keyphrases.
            base_text = f"{pr.title}\n\n{pr.text}".strip()
            terms: List[TermCandidate] = extract_terms_rake(
                base_text,
                max_terms=max_terms_per_paper,
                language=language,
                boost_terms=domain.keywords,
            )
            term_list = [_norm_term(t.term) for t in terms if _norm_term(t.term)]
            term_set = set(term_list)
            paper_term_sets[pr.paper_id] = term_set

            # Sentence-level co-occurrence (lightweight)
            if year is None:
                continue
            low_text = pr.text.lower()
            sents = _sentences(low_text)
            for snt in sents:
                present = [t for t in term_list if t and (t in snt)]
                if len(present) < 2:
                    continue
                # de-dup within a sentence
                uniq = sorted(set(present))
                for i in range(len(uniq)):
                    for j in range(i + 1, len(uniq)):
                        a, b = uniq[i], uniq[j]
                        # undirected stable key
                        src, tgt = (a, b) if a <= b else (b, a)
                        key = (src, "cooccurs_with", tgt)
                        if key not in edges:
                            edges[key] = EdgeStats(source=src, target=tgt, predicate="cooccurs_with", directed=False)
                        e = edges[key]
                        e.total_count += 1
                        e.yearly_count[year] = e.yearly_count.get(year, 0) + 1
                        e.papers.add(pr.paper_id)

    # Node stats from paper_term_sets
    for pr in papers:
        ts = paper_term_sets.get(pr.paper_id) or set()
        if not ts:
            continue
        for term in ts:
            if term not in nodes:
                nodes[term] = NodeStats(term=term)
            n = nodes[term]
            n.doc_freq += 1
            if pr.year is not None:
                n.yearly_doc_freq[pr.year] = n.yearly_doc_freq.get(pr.year, 0) + 1

    # Derived edge features & scores
    n_docs = max(1, len(paper_term_sets))
    years_all = sorted({y for e in edges.values() for y in e.yearly_count.keys()})
    max_year = max(years_all) if years_all else None

    def pmi(edge: EdgeStats) -> float:
        # PMI based on paper-level co-occurrence
        du = nodes.get(edge.source).doc_freq if nodes.get(edge.source) else 0
        dv = nodes.get(edge.target).doc_freq if nodes.get(edge.target) else 0
        dij = len(edge.papers)
        if du <= 0 or dv <= 0 or dij <= 0:
            return 0.0
        return math.log((dij * n_docs) / float(du * dv) + 1e-9)

    def trend(edge: EdgeStats) -> float:
        if max_year is None:
            return 0.0
        rw = max(1, recent_window_years)
        recent_years = [y for y in range(max_year - rw + 1, max_year + 1)]
        recent = sum(edge.yearly_count.get(y, 0) for y in recent_years)
        total = sum(edge.yearly_count.values())
        past = total - recent
        # Normalized trend: recent share minus expected share
        recent_share = recent / float(max(1, total))
        expected = rw / float(max(1, len(years_all))) if years_all else 0.0
        return recent_share - expected

    # Apply expert overrides as a feature
    _apply_expert_overrides(edges, expert_overrides_path)

    edge_list: List[EdgeStats] = []
    for e in edges.values():
        e.features.setdefault("pmi", pmi(e))
        e.features.setdefault("trend", trend(e))
        e.features.setdefault("log_count", math.log1p(float(e.total_count)))
        e.features.setdefault("mean_conf", e.mean_confidence)

        expert_w = float(e.features.get("expert_weight", 0.0) or 0.0)
        # Final score (heuristic; can be replaced by a learned model later)
        e.score = (
            1.0 * e.features["log_count"]
            + 0.75 * e.features["pmi"]
            + 1.25 * e.features["trend"]
            + 0.5 * e.features["mean_conf"]
            + 1.5 * expert_w
        )
        edge_list.append(e)

    edge_list.sort(key=lambda x: x.score, reverse=True)

    return TemporalKnowledgeGraph(
        nodes=nodes,
        edges=edge_list,
        meta={
            "domain": domain.domain_id,
            "domain_title": domain.title,
            "query": query,
            "n_papers": len(paper_term_sets),
            "edge_mode": chosen_mode,
            "years": years_all,
        },
    )
