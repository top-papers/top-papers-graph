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

from ..config import settings
from ..llm import chat_json
from ..reward.rule_based import RuleBasedReward
from ..schemas import Citation, HypothesisDraft
from ..temporal.temporal_kg_builder import EdgeStats, PaperRecord, TemporalKnowledgeGraph

# Optional GNN link prediction (PyTorch Geometric). Must not break base installation.
try:  # pragma: no cover
    from ..agentic.graph_tools import build_nx_graph
    from ..gnn.pyg_link_prediction import PyGLinkPredConfig, PyGUnavailableError, pyg_link_prediction
except Exception:  # pragma: no cover
    build_nx_graph = None
    PyGLinkPredConfig = None
    PyGUnavailableError = Exception
    pyg_link_prediction = None

# Optional agentic enhancer (code-writing) that can use graph algorithms.
try:  # pragma: no cover
    from ..agents.graph_candidate_agent import agent_generate_candidates
except Exception:  # pragma: no cover
    agent_generate_candidates = None


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
    query: str = "",
    domain: str = "",
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

    # ---- 2b) Missing links via GNN (optional; PyTorch Geometric) ----
    # This is intentionally best-effort: if PyG is not installed, we fall back to classic methods.
    gnn_missing: List[HypothesisCandidate] = []
    if (
        bool(getattr(settings, "hyp_gnn_enabled", False))
        and pyg_link_prediction is not None
        and PyGLinkPredConfig is not None
        and build_nx_graph is not None
    ):
        try:
            cfg = PyGLinkPredConfig(
                epochs=int(getattr(settings, "hyp_gnn_epochs", 80) or 80),
                hidden_dim=int(getattr(settings, "hyp_gnn_hidden_dim", 64) or 64),
                lr=float(getattr(settings, "hyp_gnn_lr", 0.01) or 0.01),
                node_cap=int(getattr(settings, "hyp_gnn_node_cap", 300) or 300),
                seed=int(getattr(settings, "hyp_gnn_seed", 7) or 7),
                device="cpu",
            )
            G = build_nx_graph(kg, directed=False, min_total_count=1)
            preds = pyg_link_prediction(G, top_k=int(top_k), config=cfg)
        except PyGUnavailableError as e:
            console.print("[yellow]GNN disabled/unavailable:[/yellow] ", end="")
            console.print(str(e), markup=False)
            preds = []
        except Exception as e:
            console.print("[yellow]GNN link prediction failed:[/yellow] ", end="")
            console.print(f"{type(e).__name__}: {e}", markup=False)
            preds = []

        for u, w, prob in preds:
            # evidence: papers that mention both u and w (best-effort)
            ev: List[Citation] = []
            ul, wl = str(u).lower(), str(w).lower()
            for pr in papers:
                low = pr.text.lower()
                if ul in low and wl in low:
                    ev.append(Citation(source_id=pr.paper_id, text_snippet=_sentence_snippet(pr.text, ul, wl)))
                if len(ev) >= 3:
                    break

            gnn_missing.append(
                HypothesisCandidate(
                    kind="gnn_missing_link",
                    source=str(u),
                    target=str(w),
                    predicate="may_relate_to",
                    # Scale to be comparable with other heuristic scores.
                    score=float(prob) * 10.0,
                    time_scope=time_scope,
                    evidence=ev,
                    graph_signals={
                        "gnn_prob": float(prob),
                        "gnn_hidden_dim": float(cfg.hidden_dim),
                        "gnn_epochs": float(cfg.epochs),
                    },
                )
            )

    # Combine
    out = emerging + missing + gnn_missing

    # ---- 3) Agentic graph reasoning (optional) ----
    if agent_generate_candidates is not None:
        try:
            agent_cands = agent_generate_candidates(
                kg,
                papers=papers,
                query=query,
                domain=domain,
                top_k=top_k,
                recent_window_years=recent_window_years,
            )
        except Exception:
            agent_cands = []

        # Map to HypothesisCandidate and avoid exact duplicates.
        seen = {(c.source, c.predicate, c.target) for c in out}
        for ac in agent_cands:
            k = (ac.source, ac.predicate, ac.target)
            if k in seen:
                continue
            seen.add(k)
            out.append(
                HypothesisCandidate(
                    kind=f"agent_{ac.kind}",
                    source=ac.source,
                    target=ac.target,
                    predicate=ac.predicate,
                    score=float(ac.score),
                    time_scope=ac.time_scope,
                    evidence=ac.evidence,
                    graph_signals={"agent_score": float(ac.score), **(ac.graph_signals or {})},
                )
            )
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

    candidates = generate_candidates(kg, papers=papers, query=query, domain=domain, top_k=top_k)
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
