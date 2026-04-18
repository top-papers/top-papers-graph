"""Shared prompt serialisation helpers for the SFT/GRPO builders."""
from __future__ import annotations

from typing import Any

SFT_SYSTEM_PROMPT = "You are a careful scientific extraction assistant."
GRPO_SYSTEM_PROMPT = "You review scientific temporal-KG assertions."


def _join(parts: list[str]) -> str:
    return "\n".join(p for p in parts if p is not None and p != "")


def _format_paper(paper: dict[str, Any]) -> str:
    year = paper.get("year")
    title = (paper.get("title") or "").strip()
    base = paper.get("id", "<unknown>")
    year_str = f" ({year})" if year else ""
    title_str = f" — {title}" if title else ""
    if paper.get("resolved") is False:
        return f"- {base}{year_str}{title_str} [unresolved]"
    return f"- {base}{year_str}{title_str}"


def _format_source(src: dict[str, Any]) -> str:
    locator = src.get("locator") or ""
    page = src.get("page")
    meta = " / ".join(
        part for part in [
            src.get("paper_ref_id") or src.get("source") or "",
            f"p. {page}" if page else "",
            locator,
        ] if part
    )
    snippet = (src.get("snippet_or_summary") or "").strip()
    kind = src.get("type", "text")
    head = f"[{kind}] {meta}" if meta else f"[{kind}]"
    if snippet:
        return f"{head}\n  > {snippet}"
    return head


def _format_conditions(cond: dict[str, Any] | None) -> str:
    cond = cond or {}
    rows = []
    for key in ("system", "environment", "protocol", "notes"):
        value = (cond.get(key) or "").strip()
        if value:
            rows.append(f"- {key}: {value}")
    return "\n".join(rows) if rows else "- (none)"


def build_sft_user_prompt(
    doc: dict[str, Any],
    step: dict[str, Any],
    previous_steps: list[dict[str, Any]],
) -> str:
    domain_label = doc.get("domain_label") or doc.get("domain") or ""
    papers_block = "\n".join(_format_paper(p) for p in doc.get("papers", []) or []) or "(none)"
    sources_block = "\n".join(_format_source(s) for s in step.get("sources", []) or []) or "(none)"
    conditions_block = _format_conditions(step.get("conditions"))

    history_lines: list[str] = []
    for prev in previous_steps:
        history_lines.append(
            f"Step {prev['step_id']}. {prev.get('claim','').strip()}\n"
            f"  inference: {(prev.get('inference') or '').strip()}\n"
            f"  next_question: {(prev.get('next_question') or '').strip()}"
        )
    history_block = "\n".join(history_lines) if history_lines else "(no previous steps)"

    return _join(
        [
            f"Topic: {doc.get('topic','')}",
            f"Domain: {domain_label}",
            f"Cutoff year: {doc.get('cutoff_year')}",
            "",
            "Papers:",
            papers_block,
            "",
            f"Step {step['step_id']} current claim:",
            (step.get("claim") or "").strip(),
            "",
            f"Temporal window: {step.get('start_date')} — {step.get('end_date')} (time_source: {step.get('time_source','')})",
            f"Importance: {step.get('importance','')}",
            "",
            "Conditions:",
            conditions_block,
            "",
            "Sources:",
            sources_block,
            "",
            "Previous reasoning:",
            history_block,
            "",
            "Produce JSON with keys {inference, next_question}.",
        ]
    )


def build_gold_assertion_sft_prompt(bundle: dict[str, Any], assertion: dict[str, Any]) -> str:
    evidence = assertion.get("evidence") or {}
    paper_ids = ", ".join(assertion.get("paper_ids") or []) or "(none)"
    return _join(
        [
            f"Topic: {bundle.get('topic','')}",
            f"Domain: {bundle.get('domain','')}",
            f"Cutoff year: {bundle.get('cutoff_year')}",
            "",
            "Papers: " + paper_ids,
            "",
            "Given the evidence below, reconstruct the gold triple (subject, predicate, object)",
            "and its temporal window (start_date / end_date).",
            "",
            "Evidence:",
            (evidence.get("text") or "").strip() or "(no text)",
            f"Page: {evidence.get('page')}" if evidence.get("page") else "",
            f"Figure/Table: {evidence.get('figure_or_table')}" if evidence.get("figure_or_table") else "",
            "",
            "Produce JSON with keys {subject, predicate, object, start_date, end_date}.",
        ]
    )


def build_grpo_user_prompt(bundle: dict[str, Any], assertion: dict[str, Any]) -> str:
    evidence = assertion.get("evidence") or {}
    paper_ids = ", ".join(assertion.get("paper_ids") or []) or "(none)"
    triple = (
        f"{assertion.get('subject','')} — {assertion.get('predicate','')} — {assertion.get('object','')}"
    )
    return _join(
        [
            f"Topic: {bundle.get('topic','')}",
            f"Domain: {bundle.get('domain','')}",
            f"Cutoff year: {bundle.get('cutoff_year')}",
            "",
            "Papers: " + paper_ids,
            "",
            "Candidate assertion:",
            f"  triple: {triple}",
            f"  start_date: {assertion.get('start_date')}",
            f"  end_date: {assertion.get('end_date')}",
            f"  importance_score: {assertion.get('importance_score')}",
            "",
            "Evidence:",
            (evidence.get("text") or "").strip() or "(no text)",
            f"Page: {evidence.get('page')}" if evidence.get("page") else "",
            f"Figure/Table: {evidence.get('figure_or_table')}" if evidence.get("figure_or_table") else "",
            "",
            "Decide whether this assertion is supported by the evidence.",
            "Produce JSON with keys {verdict, rationale}.",
        ]
    )
