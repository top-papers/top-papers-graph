"""Независимая LLM-оценка извлечённых триплетов в режиме скептика.

Использует отдельный LLM-вызов (ollama) в роли «скептический рецензент»,
который независимо размечает каждый триплет, выданный пайплайном извлечения.
Метки критика служат независимым target'ом для precision / recall / agreement
и разрывают циклическую валидацию, при которой пайплайн оценивает сам себя.

Реализация следует шаблону LLM-as-Judge (Zheng et al. 2023).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import requests

log = logging.getLogger(__name__)

Verdict = Literal["valid", "noise", "borderline"]

# ---------------------------------------------------------------------------
# Critic prompt — deliberately different perspective from the extraction prompt
# ---------------------------------------------------------------------------

_CRITIC_SYSTEM = """\
You are a skeptical scientific reviewer.  Your job is to decide whether an
extracted assertion (subject → predicate → object) represents a **real,
non-trivial scientific finding** that is supported by the provided evidence.

For each assertion, respond with a JSON object:
{
  "verdict": "valid" | "noise" | "borderline",
  "rationale": "<one sentence explaining your decision>"
}

Criteria:
- "valid": The assertion describes a concrete, verifiable scientific result or
  causal/temporal relationship.  The evidence snippet clearly supports it.
- "noise": The assertion is a procedural detail, a formula fragment, a trivial
  definition (X is a type of Y), an acknowledgement, a metadata artefact, or
  too vague to be meaningful.
- "borderline": The claim is plausible but the evidence is insufficient, or
  the assertion is overly generic (e.g. "model improves performance").

Be strict.  When in doubt between valid and borderline, choose borderline.
When in doubt between borderline and noise, choose noise.
Respond ONLY with the JSON object, no extra text.\
"""


@dataclass
class CriticVerdict:
    subject: str
    predicate: str
    object: str
    evidence: str
    verdict: Verdict
    rationale: str
    latency_seconds: float = 0.0


@dataclass
class CriticReport:
    """Aggregated results of adversarial evaluation."""
    verdicts: List[CriticVerdict] = field(default_factory=list)
    model: str = ""
    total_time_seconds: float = 0.0

    @property
    def n_valid(self) -> int:
        return sum(1 for v in self.verdicts if v.verdict == "valid")

    @property
    def n_noise(self) -> int:
        return sum(1 for v in self.verdicts if v.verdict == "noise")

    @property
    def n_borderline(self) -> int:
        return sum(1 for v in self.verdicts if v.verdict == "borderline")

    def summary(self) -> Dict[str, Any]:
        n = len(self.verdicts)
        return {
            "total": n,
            "valid": self.n_valid,
            "noise": self.n_noise,
            "borderline": self.n_borderline,
            "valid_rate": round(self.n_valid / n, 3) if n else 0,
            "noise_rate": round(self.n_noise / n, 3) if n else 0,
            "model": self.model,
            "total_time_seconds": round(self.total_time_seconds, 1),
        }


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

def _ollama_chat(
    model: str,
    system: str,
    user: str,
    *,
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0,
    timeout: int = 120,
) -> str:
    """Послать chat completion в ollama и вернуть текст ответа."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }
    resp = requests.post(
        f"{base_url}/api/chat",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _parse_critic_response(text: str) -> tuple[Verdict, str]:
    """Извлечь verdict и rationale из JSON-ответа критика."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        obj = json.loads(text)
        v = str(obj.get("verdict", "")).strip().lower()
        if v not in ("valid", "noise", "borderline"):
            v = "borderline"
        rationale = str(obj.get("rationale", ""))
        return v, rationale  # type: ignore[return-value]
    except (json.JSONDecodeError, AttributeError):
        # Try to extract verdict from free text
        lower = text.lower()
        if "noise" in lower:
            return "noise", text[:200]
        if "valid" in lower:
            return "valid", text[:200]
        return "borderline", f"unparseable response: {text[:200]}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_triplets(
    triplets: List[Dict[str, Any]],
    *,
    model: str = "qwen2.5:7b-instruct",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0,
) -> CriticReport:
    """Прогнать независимую LLM-оценку для списка триплетов.

    Каждый dict должен содержать subject, predicate, object и опционально evidence.
    Возвращает CriticReport с независимыми вердиктами по каждому триплету.
    """
    report = CriticReport(model=model)
    t0 = time.monotonic()

    for i, t in enumerate(triplets):
        subj = str(t.get("subject", ""))
        pred = str(t.get("predicate", ""))
        obj = str(t.get("object", ""))
        evidence = str(t.get("evidence") or t.get("evidence_text") or t.get("snippet_or_summary") or "")

        user_msg = f"Assertion: [{subj}] --{pred}--> [{obj}]"
        if evidence:
            user_msg += f'\nEvidence: "{evidence[:500]}"'

        t_start = time.monotonic()
        try:
            raw = _ollama_chat(
                model=model,
                system=_CRITIC_SYSTEM,
                user=user_msg,
                base_url=base_url,
                temperature=temperature,
            )
            verdict, rationale = _parse_critic_response(raw)
        except Exception as exc:
            log.warning("Critic call failed for triplet %d: %s", i, exc)
            verdict, rationale = "borderline", f"critic error: {exc}"
        latency = time.monotonic() - t_start

        report.verdicts.append(CriticVerdict(
            subject=subj,
            predicate=pred,
            object=obj,
            evidence=evidence[:300],
            verdict=verdict,
            rationale=rationale,
            latency_seconds=round(latency, 2),
        ))

        if (i + 1) % 10 == 0:
            log.info("Critic evaluated %d/%d triplets", i + 1, len(triplets))

    report.total_time_seconds = time.monotonic() - t0
    return report


def build_test_set_from_trajectories(
    trajectory_dir: str,
    *,
    n_noise: int = 15,
) -> List[Dict[str, Any]]:
    """Собрать смешанный тестовый набор: экспертные траектории + синтетический шум.

    Положительные примеры — рёбра из YAML-траекторий, размеченных экспертом.
    Отрицательные — заранее заготовленные шумные триплеты (формулы, ссылки на
    таблицы, благодарности и т.п.). Возвращает сбалансированный список для оценки.
    """
    import yaml
    from pathlib import Path

    triplets: List[Dict[str, Any]] = []
    tdir = Path(trajectory_dir)

    # Load expert-curated edges as positives
    for yf in sorted(tdir.glob("*.yaml")):
        if yf.name.startswith("_"):
            continue
        try:
            doc = yaml.safe_load(yf.read_text(encoding="utf-8"))
        except Exception:
            continue
        steps = {s["step_id"]: s for s in (doc.get("steps") or []) if isinstance(s, dict)}
        for edge in (doc.get("edges") or []):
            from_step = steps.get(edge.get("from_step"))
            to_step = steps.get(edge.get("to_step"))
            if not from_step or not to_step:
                continue
            subj_claim = from_step.get("claim", "")
            obj_claim = to_step.get("claim", "")
            # Use first ~40 chars of claim as entity name
            subj = subj_claim[:80].split(",")[0].strip()
            obj = obj_claim[:80].split(",")[0].strip()
            sources = from_step.get("sources") or []
            evidence = sources[0].get("snippet_or_summary", "") if sources else ""
            triplets.append({
                "subject": subj,
                "predicate": edge.get("label", "leads_to"),
                "object": obj,
                "evidence": evidence,
                "expected_label": "valid",
                "source_file": yf.name,
            })

    # Generate synthetic noise triplets
    noise_templates = [
        {"subject": "Table 3", "predicate": "is_shown_in", "object": "Appendix B", "evidence": "See supplementary materials."},
        {"subject": "p < 0.05", "predicate": "is_estimated_to", "object": "significance level", "evidence": "Statistical tests were performed."},
        {"subject": "the authors", "predicate": "acknowledges", "object": "anonymous reviewers", "evidence": "We thank the anonymous reviewers for their helpful comments."},
        {"subject": "x", "predicate": "is_equal_to", "object": "f(y)", "evidence": "Let x = f(y) where f is defined in eq. 3."},
        {"subject": "Section 4", "predicate": "describes", "object": "methodology", "evidence": "In this section we describe our methodology."},
        {"subject": "model", "predicate": "uses", "object": "data", "evidence": "The model was trained on the dataset."},
        {"subject": "results", "predicate": "are_shown_in", "object": "Figure 2", "evidence": "Results are presented in Figure 2."},
        {"subject": "50%", "predicate": "consists_of", "object": "training samples", "evidence": "We split the data 50/50 for training and testing."},
        {"subject": "Eq. 7", "predicate": "is_derived_from", "object": "Eq. 3", "evidence": "By substituting (3) into (5) we obtain (7)."},
        {"subject": "study", "predicate": "was_carried_out_during", "object": "2019-2021", "evidence": "The study was conducted between 2019 and 2021."},
        {"subject": "performance", "predicate": "improves", "object": "results", "evidence": "Our method improves performance."},
        {"subject": "approach", "predicate": "is_a", "object": "method", "evidence": "Our approach is a novel method."},
        {"subject": "experiment", "predicate": "involves", "object": "participants", "evidence": "The experiment involved 200 participants."},
        {"subject": "copyright 2024", "predicate": "associated_with", "object": "IEEE", "evidence": "Copyright 2024 IEEE."},
        {"subject": "∂L/∂θ", "predicate": "is_computed_from", "object": "backpropagation", "evidence": "Gradients are computed via backpropagation."},
    ]
    for nt in noise_templates[:n_noise]:
        triplets.append({**nt, "expected_label": "noise", "source_file": "synthetic_noise"})

    return triplets


def compute_agreement_metrics(
    report: CriticReport,
    triplets: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Посчитать метрики согласия между expected-метками и вердиктами критика.

    Трёхклассовые вердикты сводятся к бинарному формату (valid vs non-valid)
    для стандартных метрик: precision / recall / F1 / accuracy / Cohen's kappa.
    """
    assert len(report.verdicts) == len(triplets)

    # Binary mapping: valid=positive, noise/borderline=negative
    tp = fp = fn = tn = 0
    confusion = {"valid_valid": 0, "valid_noise": 0, "valid_borderline": 0,
                 "noise_valid": 0, "noise_noise": 0, "noise_borderline": 0}

    for v, t in zip(report.verdicts, triplets):
        expected = t.get("expected_label", "valid")
        predicted = v.verdict

        confusion[f"{expected}_{predicted}"] = confusion.get(f"{expected}_{predicted}", 0) + 1

        # Binary: expected=valid → positive; expected=noise → negative
        exp_pos = (expected == "valid")
        pred_pos = (predicted == "valid")

        if exp_pos and pred_pos:
            tp += 1
        elif exp_pos and not pred_pos:
            fn += 1
        elif not exp_pos and pred_pos:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    # Cohen's kappa
    n = tp + tn + fp + fn
    if n > 0:
        p_observed = (tp + tn) / n
        p_expected = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (n * n)
        kappa = (p_observed - p_expected) / (1 - p_expected) if p_expected < 1.0 else 0.0
    else:
        kappa = 0.0

    return {
        "binary_metrics": {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "accuracy": round(accuracy, 3),
            "cohens_kappa": round(kappa, 3),
        },
        "confusion_matrix": confusion,
        "counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }
