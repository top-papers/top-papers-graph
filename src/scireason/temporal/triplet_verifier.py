"""Верификация извлечённых триплетов на уровне рёбер.

Работает на словаре ``edges``, построенном :func:`build_temporal_kg` после
нормализации сущностей. Для каждой статьи отправляет пакет рёбер в LLM,
который классифицирует каждое как ``valid`` / ``noise`` / ``speculative``.
Рёбра с вердиктом ``noise`` удаляются из графа и сохраняются отдельно.

Применяются два уровня фильтрации:

1. **Confidence gate** — рёбра с ``mean_confidence < threshold`` отбрасываются
   до LLM-вызова (дёшево, без затрат на модель).
2. **LLM verification** — оставшиеся рёбра уходят в модель пакетами по
   ``batch_size``.

Если LLM-вызов падает, используется rule-based fallback по ``total_count``
и ``mean_confidence``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple

from rich.console import Console

from ..config import settings

console = Console(highlight=False)
log = logging.getLogger(__name__)

Verdict = Literal["valid", "noise", "speculative"]

# ---------------------------------------------------------------------------
# Confidence gate
# ---------------------------------------------------------------------------

def confidence_gate(
    edges: Dict[Tuple[str, str, str], Any],
    *,
    threshold: float = 0.5,
) -> Tuple[Dict[Tuple[str, str, str], Any], List[Any]]:
    """Отсечь рёбра с mean_confidence ниже порога.

    Возвращает (kept_edges, rejected_edges).
    """
    kept: Dict[Tuple[str, str, str], Any] = {}
    rejected: List[Any] = []
    for key, edge in edges.items():
        mc = getattr(edge, "mean_confidence", 0.0)
        if mc < threshold:
            rejected.append(edge)
        else:
            kept[key] = edge
    return kept, rejected


# ---------------------------------------------------------------------------
# LLM verification
# ---------------------------------------------------------------------------

_VERIFY_SYSTEM_PROMPT = """\
Ты — эксперт по анализу научных публикаций. Классифицируй каждое извлечённое утверждение:
- "valid": реальный научный результат, вывод или установленная связь, подтверждённая текстом статьи.
- "noise": процедурная деталь, фрагмент формулы, благодарность, описание структуры статьи, тривиальное определение или слишком абстрактное утверждение.
- "speculative": правдоподобно, но нет достаточного прямого подтверждения в тексте.

Будь строгим. Описание методологии эксперимента ("мы разбили выборку на 50/50") — это noise.
Тривиальные определения ("X is a type of Y") — это noise.
Верни ТОЛЬКО JSON массив объектов с полями "id" (int) и "verdict" (string). Без пояснений.\
"""

_VERIFY_SCHEMA_HINT = '[{"id": 1, "verdict": "valid"}, {"id": 2, "verdict": "noise"}]'


def _format_edge_for_verification(idx: int, edge: Any) -> str:
    """Сериализовать ребро в компактную строку для prompt'а верификации."""
    s = getattr(edge, "source", "?")
    p = getattr(edge, "predicate", "?")
    o = getattr(edge, "target", "?")
    conf = getattr(edge, "mean_confidence", 0.0)
    count = getattr(edge, "total_count", 1)
    n_papers = len(getattr(edge, "papers", set()))

    # Best evidence quote
    quotes = getattr(edge, "evidence_quotes", [])
    quote = ""
    if quotes:
        first = quotes[0]
        if isinstance(first, dict):
            quote = str(first.get("quote") or first.get("snippet_or_summary") or "")[:150]
        elif isinstance(first, str):
            quote = first[:150]

    line = f"#{idx}. [{s}] --{p}--> [{o}]  (conf={conf:.2f}, chunks={count}, papers={n_papers})"
    if quote:
        line += f'\n    Цитата: "{quote}"'
    return line


def _build_verification_prompt(
    paper_title: str,
    paper_abstract: str,
    edge_items: Sequence[Tuple[int, Any]],
) -> Tuple[str, str]:
    """Собрать (system, user) prompt'ы для пакета рёбер."""
    formatted = "\n".join(
        _format_edge_for_verification(idx, edge) for idx, edge in edge_items
    )
    user = f'Статья: "{paper_title}"\nАннотация: "{paper_abstract[:600]}"\n\nКлассифицируй следующие утверждения:\n\n{formatted}'
    return _VERIFY_SYSTEM_PROMPT, user


def _parse_verification_response(
    raw: Any,
    expected_ids: Set[int],
) -> Dict[int, Verdict]:
    """Разобрать JSON-ответ LLM в словарь id → verdict."""
    verdicts: Dict[int, Verdict] = {}
    if not isinstance(raw, list):
        return verdicts
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("id", -1))
        except (ValueError, TypeError):
            continue
        v = str(item.get("verdict", "")).strip().lower()
        if v not in ("valid", "noise", "speculative"):
            v = "speculative"  # conservative default
        if idx in expected_ids:
            verdicts[idx] = v  # type: ignore[assignment]
    return verdicts


def _fallback_verdict(edge: Any) -> Verdict:
    """Rule-based fallback при сбое LLM-верификации.

    Сначала пробует quality_score из edge.features (если посчитан скорером),
    иначе работает по эвристикам redundancy + confidence.
    """
    # Auto-reject predicates with >50% historical rejection rate
    pred = getattr(edge, "predicate", "").lower().strip()
    if pred in ("associated_with", "is_associated_with", "are_associated_with", "related_to"):
        return "noise"

    feats = getattr(edge, "features", {}) or {}
    qs = float(feats.get("quality_score", -1))
    if qs >= 0:
        if qs >= 0.7:
            return "valid"
        if qs <= 0.25:
            return "noise"
        return "speculative"

    count = getattr(edge, "total_count", 1)
    conf = getattr(edge, "mean_confidence", 0.0)
    n_papers = len(getattr(edge, "papers", set()))

    if (count >= 2 and conf >= 0.6) or n_papers >= 2:
        return "valid"
    if count == 1 and conf < 0.5:
        return "noise"
    return "speculative"


@dataclass
class VerificationResult:
    """Сводка по проходу верификации."""
    total_edges: int = 0
    confidence_rejected: int = 0
    llm_valid: int = 0
    llm_noise: int = 0
    llm_speculative: int = 0
    fallback_used: bool = False

    @property
    def kept(self) -> int:
        return self.llm_valid + self.llm_speculative

    def summary_line(self) -> str:
        return (
            f"confidence_gate=-{self.confidence_rejected}, "
            f"valid={self.llm_valid}, noise={self.llm_noise}, "
            f"speculative={self.llm_speculative}, kept={self.kept}/{self.total_edges}"
        )


def verify_edges(
    edges: Dict[Tuple[str, str, str], Any],
    papers: Sequence[Any],
    *,
    confidence_threshold: float = 0.5,
    batch_size: int = 80,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> Tuple[Dict[Tuple[str, str, str], Any], List[Any], VerificationResult]:
    """Запустить confidence gate + LLM-верификацию для словаря рёбер.

    Возвращает (kept_edges, rejected_edges, result).
    """
    result = VerificationResult(total_edges=len(edges))

    # --- Phase 1: Confidence gate ---
    edges, low_conf = confidence_gate(edges, threshold=confidence_threshold)
    result.confidence_rejected = len(low_conf)
    if low_conf:
        console.print(
            f"  [yellow]Confidence gate: removed {len(low_conf)} edges "
            f"(mean_confidence < {confidence_threshold})[/yellow]"
        )

    if not edges:
        return edges, low_conf, result

    # Build paper lookup for context
    paper_map: Dict[str, Any] = {}
    for pr in papers:
        pid = getattr(pr, "paper_id", None) or ""
        paper_map[pid] = pr

    # --- Phase 2: LLM verification ---
    # Group edges by their primary paper
    paper_edges: Dict[str, List[Tuple[Tuple[str, str, str], Any]]] = {}
    for key, edge in edges.items():
        edge_papers = getattr(edge, "papers", set())
        primary = next(iter(edge_papers)) if edge_papers else "__unknown__"
        paper_edges.setdefault(primary, []).append((key, edge))

    all_verdicts: Dict[Tuple[str, str, str], Verdict] = {}

    try:
        from ..llm import chat_json
    except ImportError:
        log.warning("LLM module not available, using fallback verdicts")
        for key, edge in edges.items():
            all_verdicts[key] = _fallback_verdict(edge)
        result.fallback_used = True
        return _apply_verdicts(edges, all_verdicts, low_conf, result)

    for paper_id, items in paper_edges.items():
        pr = paper_map.get(paper_id)
        title = getattr(pr, "title", paper_id) if pr else paper_id
        abstract = (getattr(pr, "text", "") or "")[:600] if pr else ""

        # Batch
        for batch_start in range(0, len(items), batch_size):
            batch = items[batch_start : batch_start + batch_size]
            indexed: List[Tuple[int, Any]] = []
            key_by_idx: Dict[int, Tuple[str, str, str]] = {}
            for i, (key, edge) in enumerate(batch, start=1):
                indexed.append((i, edge))
                key_by_idx[i] = key

            system, user = _build_verification_prompt(title, abstract, indexed)
            expected_ids = set(key_by_idx.keys())

            try:
                from ..llm import temporary_llm_selection
                with temporary_llm_selection(llm_provider=llm_provider, llm_model=llm_model):
                    raw = chat_json(
                        system=system,
                        user=user,
                        schema_hint=_VERIFY_SCHEMA_HINT,
                        temperature=0.0,
                    )
                verdicts = _parse_verification_response(raw, expected_ids)
            except Exception as e:
                console.print(f"  [red]LLM verification failed for {paper_id}: {e}. Using rule-based fallback.[/red]")
                verdicts = {}
                result.fallback_used = True

            # Fill missing verdicts with fallback
            for idx, key in key_by_idx.items():
                if idx in verdicts:
                    all_verdicts[key] = verdicts[idx]
                else:
                    edge = edges[key]
                    all_verdicts[key] = _fallback_verdict(edge)
                    result.fallback_used = True

    return _apply_verdicts(edges, all_verdicts, low_conf, result)


def _apply_verdicts(
    edges: Dict[Tuple[str, str, str], Any],
    verdicts: Dict[Tuple[str, str, str], Verdict],
    rejected_so_far: List[Any],
    result: VerificationResult,
) -> Tuple[Dict[Tuple[str, str, str], Any], List[Any], VerificationResult]:
    """Раскидать рёбра по вердиктам и обновить счётчики result."""
    kept: Dict[Tuple[str, str, str], Any] = {}
    rejected = list(rejected_so_far)

    for key, edge in edges.items():
        v = verdicts.get(key, "speculative")
        if v == "valid":
            result.llm_valid += 1
            kept[key] = edge
        elif v == "noise":
            result.llm_noise += 1
            rejected.append(edge)
        else:  # speculative
            result.llm_speculative += 1
            kept[key] = edge

    return kept, rejected, result
