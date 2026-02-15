from __future__ import annotations

"""Hypothesis generation from a temporal term graph.

This module is intentionally pragmatic:
* It produces **testable** hypothesis drafts from graph signals (emergence / link prediction).
* It can optionally use an LLM to rewrite the draft into higher-quality scientific language,
  but the pipeline still works without an LLM.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from rich.console import Console

from ..llm import chat_json
from ..reward.rule_based import RuleBasedReward
from ..schemas import Citation, HypothesisDraft
from ..temporal.temporal_kg_builder import EdgeStats, PaperRecord, TemporalKnowledgeGraph


console = Console()


HYP_SCHEMA_HINT = """Ожидается JSON объект HypothesisDraft:
{
  "title": "...",
  "premise": "...",
  "mechanism": "...",
  "time_scope": "...",
  "proposed_experiment": "...",
  "supporting_evidence": [{"source_id":"...","text_snippet":"..."}],
  "confidence_score": 1-10
}
"""


def _sentence_snippet(text: str, a: str, b: str, max_len: int = 220) -> str:
    t = (text or "")
    low = t.lower()
    a = a.lower()
    b = b.lower()
    # naive sentence split
    for s in t.replace("\n", " ").split("."):
        sl = s.lower()
        if a in sl and b in sl:
            s = s.strip()
            if len(s) > max_len:
                return s[: max_len - 1] + "…"
            return s
    # fallback: window around first occurrence
    idx = low.find(a)
    if idx < 0:
        idx = low.find(b)
    if idx < 0:
        return (t[: max_len - 1] + "…") if len(t) > max_len else t
    start = max(0, idx - 80)
    end = min(len(t), idx + 140)
    sn = t[start:end].strip()
    if len(sn) > max_len:
        sn = sn[: max_len - 1] + "…"
    return sn


def _paper_map(papers: Sequence[PaperRecord]) -> Dict[str, PaperRecord]:
    return {p.paper_id: p for p in papers}


@dataclass(frozen=True)
class HypothesisCandidate:
    kind: str  # emerging_edge | missing_link
    source: str
    target: str
    predicate: str
    score: float
    time_scope: str
    evidence: List[Citation]
    graph_signals: Dict[str, float]


def generate_candidates(
    kg: TemporalKnowledgeGraph,
    *,
    papers: Sequence[PaperRecord],
    top_k: int = 10,
    recent_window_years: int = 3,
    min_edge_count: int = 2,
) -> List[HypothesisCandidate]:
    """Generate hypothesis candidates based on temporal KG signals."""

    years = kg.meta.get("years") or []
    max_year = max(years) if years else None
    if max_year is None:
        # still produce something from top edges
        max_year = 0

    time_scope = ""
    if max_year:
        start = max_year - max(1, recent_window_years) + 1
        time_scope = f"{start}-{max_year}"

    pmap = _paper_map(papers)

    # ---- 1) Emerging edges (high trend) ----
    emerging: List[HypothesisCandidate] = []
    for e in kg.edges:
        if e.total_count < min_edge_count:
            continue
        tr = float(e.features.get("trend", 0.0) or 0.0)
        if tr <= 0.05:
            continue

        ev: List[Citation] = []
        if e.evidence_quotes:
            for q in e.evidence_quotes[:3]:
                ev.append(Citation(source_id=str(q.get("paper_id")), text_snippet=str(q.get("quote"))[:220]))
        else:
            # build from paper texts
            for pid in list(sorted(e.papers))[:3]:
                pr = pmap.get(pid)
                if not pr:
                    continue
                ev.append(Citation(source_id=pid, text_snippet=_sentence_snippet(pr.text, e.source, e.target)))

        emerging.append(
            HypothesisCandidate(
                kind="emerging_edge",
                source=e.source,
                target=e.target,
                predicate=e.predicate,
                score=float(e.score),
                time_scope=time_scope,
                evidence=ev,
                graph_signals={
                    "edge_score": float(e.score),
                    "trend": tr,
                    "pmi": float(e.features.get("pmi", 0.0) or 0.0),
                    "mean_conf": float(e.features.get("mean_conf", 0.0) or 0.0),
                },
            )
        )

    emerging.sort(key=lambda c: c.score, reverse=True)
    emerging = emerging[: top_k]

    # If no edges satisfy the emergence threshold (small corpus / early run),
    # fall back to "strongest" edges by score so the pipeline always produces hypotheses.
    if not emerging:
        for e in kg.edges[:top_k]:
            ev: List[Citation] = []
            if e.evidence_quotes:
                for q in e.evidence_quotes[:3]:
                    ev.append(Citation(source_id=str(q.get("paper_id")), text_snippet=str(q.get("quote"))[:220]))
            else:
                for pid in list(sorted(e.papers))[:3]:
                    pr = pmap.get(pid)
                    if not pr:
                        continue
                    ev.append(Citation(source_id=pid, text_snippet=_sentence_snippet(pr.text, e.source, e.target)))

            emerging.append(
                HypothesisCandidate(
                    kind="top_edge",
                    source=e.source,
                    target=e.target,
                    predicate=e.predicate,
                    score=float(e.score),
                    time_scope=time_scope,
                    evidence=ev,
                    graph_signals={
                        "edge_score": float(e.score),
                        "trend": float(e.features.get("trend", 0.0) or 0.0),
                        "pmi": float(e.features.get("pmi", 0.0) or 0.0),
                        "mean_conf": float(e.features.get("mean_conf", 0.0) or 0.0),
                    },
                )
            )

        emerging.sort(key=lambda c: c.score, reverse=True)
        emerging = emerging[:top_k]

    # ---- 2) Missing links via common neighbors (cheap link prediction) ----
    # Build a neighbor map from top edges only (to keep O(n^2) manageable)
    top_edges_for_lp = kg.edges[: min(400, len(kg.edges))]
    neigh: Dict[str, set[str]] = {}
    for e in top_edges_for_lp:
        a, b = e.source, e.target
        if a == b:
            continue
        neigh.setdefault(a, set()).add(b)
        neigh.setdefault(b, set()).add(a)

    candidates: Dict[Tuple[str, str], float] = {}
    for v, ns in neigh.items():
        ns_list = list(ns)
        for i in range(len(ns_list)):
            for j in range(i + 1, len(ns_list)):
                a, b = ns_list[i], ns_list[j]
                if a == b:
                    continue
                u, w = (a, b) if a <= b else (b, a)
                # Skip if already connected in top edges
                if w in neigh.get(u, set()):
                    continue
                # common neighbor count heuristic
                candidates[(u, w)] = candidates.get((u, w), 0.0) + 1.0

    missing: List[HypothesisCandidate] = []
    for (u, w), cn in sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)[: top_k]:
        # evidence: papers that mention both u and w (best-effort)
        ev: List[Citation] = []
        for pr in papers:
            low = pr.text.lower()
            if u in low and w in low:
                ev.append(Citation(source_id=pr.paper_id, text_snippet=_sentence_snippet(pr.text, u, w)))
            if len(ev) >= 3:
                break

        missing.append(
            HypothesisCandidate(
                kind="missing_link",
                source=u,
                target=w,
                predicate="may_relate_to",
                score=float(cn),
                time_scope=time_scope,
                evidence=ev,
                graph_signals={"common_neighbors": float(cn)},
            )
        )

    # Combine
    out = emerging + missing
    out.sort(key=lambda c: c.score, reverse=True)
    return out[: top_k]


def _template_hypothesis(c: HypothesisCandidate) -> HypothesisDraft:
    if c.kind in {"emerging_edge", "top_edge"}:
        title = f"{c.source} {c.predicate} {c.target} (emerging)"
        premise = (
            f"В последние годы ({c.time_scope}) в литературе заметно растёт количество упоминаний/утверждений, "
            f"связывающих '{c.source}' и '{c.target}' через отношение '{c.predicate}'."
        ).strip()
        mechanism = (
            "Возможный механизм следует уточнить: связь может быть прямой (каузальной), "
            "опосредованной третьими факторами или отражать смену методологии/датасетов. "
            "В качестве подсказки используйте общих соседей в графе и условия экспериментов."
        )
    else:
        title = f"{c.source} ↔ {c.target}: потенциальная скрытая связь"
        premise = (
            f"В темпоральном графе знаний '{c.source}' и '{c.target}' имеют общих соседей, "
            "но прямой связи в текущем корпусе не наблюдается. Это может указывать на "
            "недоисследованную связь или на разрыв между поддоменами."
        ).strip()
        mechanism = (
            "Предположительно, связь может проявляться через общий механизм/посредник (shared neighbors) "
            "или через перенос методики из одного поддомена в другой."
        )

    proposed_experiment = (
        "Проверка: (1) сформулировать операциональные метрики для обоих терминов; "
        "(2) собрать/выбрать датасет или экспериментальную установку, где можно контролировать конфаундеры; "
        "(3) сравнить базовый подход без фактора 'A' и вариант с фактором 'A', измеряя эффект на 'B'; "
        "(4) провести репликацию на независимых данных."
    )

    return HypothesisDraft(
        title=title,
        premise=premise,
        mechanism=mechanism,
        time_scope=c.time_scope,
        proposed_experiment=proposed_experiment,
        supporting_evidence=c.evidence,
        confidence_score=max(3, min(9, int(round(5 + float(c.graph_signals.get("trend", 0.0) or 0.0) * 10))))
        if c.kind in {"emerging_edge", "top_edge"}
        else 5,
    )


def _llm_hypothesis(domain: str, query: str, c: HypothesisCandidate) -> Optional[HypothesisDraft]:
    # Best-effort: if LLM isn't configured, the caller catches exceptions.
    evidence_block = "\n".join([f"- ({e.source_id}) {e.text_snippet}" for e in c.evidence])

    system = (
        f"Ты — научный ассистент. Домен: {domain}. "
        "Сформулируй проверяемую гипотезу (falsifiable), не выдумывай факты. "
        "Опирайся только на предоставленные граф-сигналы и выдержки." 
    )
    user = (
        f"Research query: {query}\n\n"
        f"Candidate type: {c.kind}\n"
        f"Relation: {c.source} |{c.predicate}| {c.target}\n"
        f"Time scope: {c.time_scope}\n"
        f"Graph signals: {json.dumps(c.graph_signals, ensure_ascii=False)}\n\n"
        f"Evidence snippets:\n{evidence_block}\n\n"
        "Сделай гипотезу максимально проверяемой: опиши метрики, контрольные условия, ожидаемый эффект."
    )

    data = chat_json(system=system, user=user, schema_hint=HYP_SCHEMA_HINT, temperature=0.2)
    return HypothesisDraft.model_validate(data)


def generate_hypotheses(
    kg: TemporalKnowledgeGraph,
    *,
    papers: Sequence[PaperRecord],
    domain: str,
    query: str,
    top_k: int = 8,
    use_llm: bool = True,
    expert_overrides_path: Path = Path("data/derived/expert_overrides.jsonl"),
) -> List[HypothesisDraft]:
    """Generate ranked hypothesis drafts from a temporal KG."""

    candidates = generate_candidates(kg, papers=papers, top_k=top_k)
    drafts: List[HypothesisDraft] = []

    for c in candidates:
        hyp: Optional[HypothesisDraft] = None
        if use_llm:
            try:
                hyp = _llm_hypothesis(domain=domain, query=query, c=c)
            except Exception as e:
                console.print(f"[yellow]LLM hypothesis failed ({type(e).__name__}: {e}). Using template.[/yellow]")
        if hyp is None:
            hyp = _template_hypothesis(c)
        drafts.append(hyp)

    # Reward-model ranking (uses expert overrides compiled from reviews)
    rm = RuleBasedReward(overrides_path=expert_overrides_path)
    scored = [(rm.score(h.model_dump()).score, h) for h in drafts]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [h for _, h in scored[:top_k]]
