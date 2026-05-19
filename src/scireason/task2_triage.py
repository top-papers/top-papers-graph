"""Автоматический triage триплетов Task 2 на корзины accept / reject / review.

Модуль содержит rule-based эвристики, классифицирующие каждый извлечённый
триплет в один из трёх вердиктов **без** вызова LLM:

* **accept** — почти наверняка валидный научный результат
* **reject** — шум: фрагмент формулы, процедурная деталь, благодарность
  или иное неинформативное утверждение
* **review** — пограничный случай, нужна экспертная оценка

После автотриажа предзаполненные вердикты впрыскиваются в исходный
``automatic_triplets.csv`` колонкой ``verdict`` — существующий
``task2_offline_review`` подхватывает их автоматически как pre-fill для
ручной разметки.

Интеграция
----------
* CLI: ``scireason triage-triplets --bundle-dir <path>``
* Pipeline: вызывается автоматически в конце ``prepare-task2-validation``
  при флаге ``--triage``.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Verdict type
# ---------------------------------------------------------------------------
Verdict = Literal["accept", "reject", "review"]

# ---------------------------------------------------------------------------
# Noise patterns (compiled once)
# ---------------------------------------------------------------------------

# Entities that are clearly not scientific concepts
_NOISE_ENTITY_RE = re.compile(
    r"^("
    r"\d[\d.,/%\s]*$"                       # pure numbers: "50%", "0.3", "25 percent"
    r"|[\d.,]+\s*(percent|samples|decile)"   # "50% samples", "0.3 first decile"
    r"|∂.*[/=]"                              # formula fragments: ∂y(t)/∂t
    r"|\(?\d+\)?$"                           # bare numbers in parens
    r"|[a-z]\(\w\)"                          # function notation: h(l), f(x)
    r"|eq\.?\s*\d"                           # equation references
    r"|table\s*\d|figure\s*\d|fig\.\s*\d"   # table/figure references
    r"|appendix\s*[a-z]"                     # appendix references
    r"|p\s*[<>=]\s*[\d.]"                    # p-values: p < 0.05
    r")",
    re.IGNORECASE,
)

# Predicates that almost never carry scientific meaning
_JUNK_PREDICATES = frozenset({
    "is_estimated_to", "is_segmented_into", "consists_of",
    "was_carried_out_during", "was_destroyed_after_experiment",
    "calculated_at", "provide_comments", "assist_research",
    "acknowledges", "thanks", "is_cited_by", "is_referenced_in",
    "is_described_in", "is_shown_in", "is_listed_in",
    "is_defined_as", "is_denoted_by", "is_abbreviated_as",
    "is_equal_to", "is_written_as", "is_formatted_as",
})

# Predicates that are strong signals of scientific content
_GOOD_PREDICATES = frozenset({
    "causes", "leads_to", "results_in", "improves", "reduces",
    "increases", "decreases", "affects", "influences", "predicts",
    "estimates", "quantifies", "outperforms", "enables", "prevents",
    "inhibits", "promotes", "mediates", "drives", "induces",
    "depends_on", "requires", "assumes", "violates", "supports",
    "contradicts", "extends", "generalizes", "refines",
    "applied_to", "used_for", "combined_with", "compared_to",
    # Common relationship predicates that still carry meaning
    "uses", "incorporates", "measures", "involves", "allows",
    "are_used_for", "is_used_for", "used_in", "is_computed_from",
    "is_determined_by", "is_measured_by", "arises_from",
    "rely_on", "relies_on", "manages", "maximizes", "minimizes",
    "affect", "reflects", "addresses", "accounts_for",
    "overcomes", "mitigates", "captures", "leverages",
})

# Procedural / acknowledgement noise in evidence text
_EVIDENCE_NOISE_RE = re.compile(
    r"(thank\w*\s+(the|our|my)|acknowledg|seminar\s+participant|"
    r"workshop\s+participant|anonymous\s+referee|anonymous\s+reviewer|"
    r"copyright\s*©|©\s*\d{4})",
    re.IGNORECASE,
)

# Entities that are too generic to be useful
_GENERIC_ENTITIES = frozenset({
    "model", "method", "approach", "result", "results", "data",
    "features", "study", "paper", "analysis", "research",
    "experiment", "system", "process", "problem", "solution",
    "value", "values", "variable", "variables", "parameter",
    "parameters", "function", "performance", "information",
    "useful", "important", "significant", "different",
})


# ---------------------------------------------------------------------------
# Single-triplet triage
# ---------------------------------------------------------------------------

def _clean(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _entity_is_noise(entity: str) -> bool:
    e = _clean(entity)
    if not e or len(e) < 2:
        return True
    if _NOISE_ENTITY_RE.match(e):
        return True
    if e in _GENERIC_ENTITIES:
        return True
    # Too short single-char or two-char entities (l, t, y, x)
    if len(e) <= 2 and e.isalpha():
        return True
    return False


def _predicate_is_junk(pred: str) -> bool:
    return _clean(pred) in _JUNK_PREDICATES


def _predicate_is_good(pred: str) -> bool:
    return _clean(pred) in _GOOD_PREDICATES


def _evidence_is_noise(snippet: str) -> bool:
    """Похож ли *текст сниппета* (не raw JSON) на благодарности/метаданные."""
    text = _clean(snippet)
    if not text or len(text) < 10:
        return False
    if _EVIDENCE_NOISE_RE.search(text):
        return True
    return False


def _evidence_snippet(row: Dict[str, Any]) -> str:
    """Достать текстовый сниппет из поля evidence (может быть JSON-строкой)."""
    raw = row.get("evidence") or row.get("evidence_text") or ""
    if isinstance(raw, str) and raw.strip().startswith("{"):
        try:
            obj = json.loads(raw)
            return str(obj.get("snippet_or_summary") or obj.get("snippet") or "")
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            log.debug("Could not parse evidence JSON: %s", exc)
    return str(raw)


def _importance(row: Dict[str, Any]) -> float:
    try:
        return float(row.get("importance_score") or 0.0)
    except (ValueError, TypeError):
        return 0.0


def triage_triplet(row: Dict[str, Any]) -> "TriageResult":
    """Классифицировать один триплет в accept / reject / review.

    Возвращает :class:`TriageResult` с вердиктом и человекочитаемыми причинами.
    """
    subj = _clean(row.get("subject"))
    pred = _clean(row.get("predicate"))
    obj = _clean(row.get("object"))
    snippet = _evidence_snippet(row)
    importance = _importance(row)

    reasons: list[str] = []
    score = 0.0  # positive = accept, negative = reject

    # --- reject signals ---
    if _entity_is_noise(subj):
        reasons.append(f"subject is noise: '{subj}'")
        score -= 2.0
    if _entity_is_noise(obj):
        reasons.append(f"object is noise: '{obj}'")
        score -= 2.0
    if _predicate_is_junk(pred):
        reasons.append(f"junk predicate: '{pred}'")
        score -= 3.0
    if _evidence_is_noise(snippet):
        reasons.append(f"evidence looks like acknowledgement/metadata")
        score -= 2.0
    if subj == obj:
        reasons.append("self-loop: subject == object")
        score -= 3.0
    # Very long entity names are usually sentence fragments
    if len(subj) > 80:
        reasons.append(f"subject too long ({len(subj)} chars)")
        score -= 1.5
    if len(obj) > 80:
        reasons.append(f"object too long ({len(obj)} chars)")
        score -= 1.5

    # --- accept signals ---
    if _predicate_is_good(pred):
        reasons.append(f"good predicate: '{pred}'")
        score += 2.0
    if importance >= 0.3:
        reasons.append(f"high importance: {importance:.3f}")
        score += 1.5
    elif importance >= 0.15:
        reasons.append(f"moderate importance: {importance:.3f}")
        score += 0.5
    if snippet and len(snippet) > 30 and not _EVIDENCE_NOISE_RE.search(snippet):
        reasons.append("has substantive evidence snippet")
        score += 1.0

    # Number of papers supporting (multi-paper = stronger)
    papers_raw = row.get("papers") or "[]"
    if isinstance(papers_raw, str):
        try:
            papers_list = json.loads(papers_raw.replace("'", '"'))
        except (json.JSONDecodeError, ValueError):
            papers_list = [papers_raw]
    else:
        papers_list = papers_raw if isinstance(papers_raw, list) else [papers_raw]
    if len(papers_list) > 1:
        reasons.append(f"multi-paper support ({len(papers_list)} papers)")
        score += 2.0

    # Assertion quality score — strongest signal when available
    try:
        qs = float(row.get("quality_score") or -1)
    except (ValueError, TypeError):
        qs = -1.0
    if qs >= 0:
        if qs >= 0.7:
            reasons.append(f"high quality score: {qs:.3f}")
            score += 2.5
        elif qs >= 0.4:
            reasons.append(f"moderate quality score: {qs:.3f}")
            score += 1.0
        elif qs < 0.2:
            reasons.append(f"low quality score: {qs:.3f}")
            score -= 2.0

    # --- verdict ---
    if score >= 2.0:
        verdict: Verdict = "accept"
    elif score <= -2.0:
        verdict = "reject"
    else:
        verdict = "review"

    return TriageResult(
        assertion_id=str(row.get("assertion_id") or ""),
        verdict=verdict,
        score=round(score, 2),
        reasons=reasons,
        row=row,
    )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TriageResult:
    assertion_id: str
    verdict: Verdict
    score: float
    reasons: list[str]
    row: Dict[str, Any] = field(repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assertion_id": self.assertion_id,
            "verdict": self.verdict,
            "score": self.score,
            "reasons": self.reasons,
        }


# ---------------------------------------------------------------------------
# Batch triage
# ---------------------------------------------------------------------------

@dataclass
class TriageSummary:
    total: int
    accepted: int
    rejected: int
    review: int
    results: list[TriageResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "review": self.review,
            "results": [r.to_dict() for r in self.results],
        }


def triage_triplets(rows: Sequence[Dict[str, Any]]) -> TriageSummary:
    """Прогнать triage на списке триплетов (dict'ы из CSV/JSON)."""
    results = [triage_triplet(r) for r in rows]
    return TriageSummary(
        total=len(results),
        accepted=sum(1 for r in results if r.verdict == "accept"),
        rejected=sum(1 for r in results if r.verdict == "reject"),
        review=sum(1 for r in results if r.verdict == "review"),
        results=results,
    )


def triage_bundle(bundle_dir: str | Path) -> TriageSummary:
    """Загрузить automatic_triplets из bundle-папки и прогнать triage.

    Помимо отчёта в ``triage_results.json``, перезаписывает исходный CSV/JSON
    с проставленным полем ``verdict`` — чтобы ``task2_offline_review``
    подхватил автоматические вердикты как pre-fill для ручной разметки.
    """
    bd = Path(bundle_dir)

    # Поддиректория submission'а (единственная вложенная папка) или сам bundle_dir.
    candidates = [p for p in bd.iterdir() if p.is_dir() and not p.name.startswith(".")]
    sub = candidates[0] if len(candidates) == 1 else bd

    csv_path = sub / "automatic_triplets.csv"
    json_path = sub / "automatic_triplets.json"
    source: Path
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        source = csv_path
    elif json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("assertions", "rows", "triplets", "edges"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        rows = payload if isinstance(payload, list) else []
        source = json_path
    else:
        raise FileNotFoundError(f"No automatic_triplets.csv or .json in {sub}")

    summary = triage_triplets(rows)

    # Отчёт.
    (sub / "triage_results.json").write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Pre-fill: проставляем verdict в исходные строки и переписываем источник.
    by_id = {r.assertion_id: r for r in summary.results}
    for row in rows:
        result = by_id.get(str(row.get("assertion_id") or ""))
        if result and not str(row.get("verdict") or "").strip():
            row["verdict"] = result.verdict
    if source.suffix == ".csv" and rows:
        fieldnames = list(dict.fromkeys(["verdict", *rows[0].keys()]))
        with source.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    elif source.suffix == ".json":
        source.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary
