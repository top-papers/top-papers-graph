#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from pydantic import TypeAdapter

from scireason.schemas import HypothesisDraft
from scireason.agents.hypothesis_tester import test_hypothesis


def evaluate_case(case: Dict[str, Any], *, use_demos: bool, default_collection: str) -> Dict[str, Any]:
    domain = str(case.get("domain") or "Science")
    collection_text = str(case.get("collection_text") or default_collection)

    hyp_obj = case.get("hypothesis")
    if hyp_obj is None:
        raise ValueError("case must contain 'hypothesis' (HypothesisDraft JSON)")
    hypothesis = TypeAdapter(HypothesisDraft).validate_python(hyp_obj)

    ctx_override = case.get("ctx_override")
    if ctx_override is not None and not isinstance(ctx_override, list):
        raise ValueError("ctx_override must be a list (same shape as retrieve_context results)")

    expected = case.get("expected") or {}
    expected_verdict = expected.get("verdict")

    pred = test_hypothesis(
        domain=domain,
        hypothesis=hypothesis,
        collection_text=collection_text,
        k=int(case.get("k", 12)),
        use_demos=use_demos,
        ctx_override=ctx_override,
    )

    verdict_ok = None
    if expected_verdict is not None:
        verdict_ok = (pred.verdict == expected_verdict)

    # Evidence recall: what fraction of expected supporting source_ids appear in predicted supporting_evidence
    exp_support = expected.get("supporting_evidence") or []
    exp_ids: Set[str] = set()
    for it in exp_support:
        if isinstance(it, dict) and it.get("source_id"):
            exp_ids.add(str(it["source_id"]))

    pred_ids = {e.source_id for e in pred.supporting_evidence}
    evidence_recall = None
    if exp_ids:
        evidence_recall = len(exp_ids & pred_ids) / len(exp_ids)

    return {
        "verdict_ok": verdict_ok,
        "evidence_recall": evidence_recall,
        "pred_verdict": pred.verdict,
        "expected_verdict": expected_verdict,
        "n_pred_support": len(pred.supporting_evidence),
        "n_expected_support": len(exp_ids),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate hypothesis testing on a gold set.")
    ap.add_argument("--gold-dir", type=str, default="data/gold/hypotheses", help="Directory with gold JSON cases.")
    ap.add_argument("--collection-text", type=str, default="papers_text", help="Default Qdrant collection name for ctx retrieval.")
    ap.add_argument("--variant", choices=["baseline", "fewshot", "both"], default="both")
    args = ap.parse_args()

    gold_dir = Path(args.gold_dir)
    cases = sorted(list(gold_dir.glob("*.json")))
    if not cases:
        raise SystemExit(f"No gold cases found in {gold_dir}. Create JSON files there (see data/gold/README.md).")

    def run(use_demos: bool) -> Dict[str, Any]:
        rows = []
        for p in cases:
            case = json.loads(p.read_text(encoding="utf-8"))
            try:
                r = evaluate_case(case, use_demos=use_demos, default_collection=args.collection_text)
                rows.append(r)
            except Exception as e:
                rows.append({"error": str(e), "verdict_ok": None, "evidence_recall": None})

        verdict_rows = [r for r in rows if r.get("verdict_ok") is not None and "error" not in r]
        verdict_acc = sum(1 for r in verdict_rows if r["verdict_ok"]) / len(verdict_rows) if verdict_rows else None

        evid_rows = [r for r in rows if r.get("evidence_recall") is not None and "error" not in r]
        evid_avg = sum(r["evidence_recall"] for r in evid_rows) / len(evid_rows) if evid_rows else None

        return {
            "n_cases": len(rows),
            "verdict_accuracy": verdict_acc,
            "avg_evidence_recall": evid_avg,
            "errors": sum(1 for r in rows if "error" in r),
        }

    out: Dict[str, Any] = {"gold_dir": str(gold_dir), "n_cases": len(cases)}
    if args.variant in ("baseline", "both"):
        out["baseline"] = run(use_demos=False)
    if args.variant in ("fewshot", "both"):
        out["fewshot"] = run(use_demos=True)

    if "baseline" in out and "fewshot" in out and out["baseline"]["verdict_accuracy"] is not None and out["fewshot"]["verdict_accuracy"] is not None:
        out["delta_verdict_accuracy"] = out["fewshot"]["verdict_accuracy"] - out["baseline"]["verdict_accuracy"]

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
