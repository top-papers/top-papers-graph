#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from scireason.temporal.temporal_triplet_extractor import extract_temporal_triplets


def _norm(s: str) -> str:
    return " ".join(s.lower().strip().split())


def _key(t: Dict[str, Any]) -> Tuple[str, str, str]:
    return (_norm(str(t.get("subject", ""))), _norm(str(t.get("predicate", ""))), _norm(str(t.get("object", ""))))


def _time_equal(a: Any, b: Any) -> bool:
    if a is None and b is None:
        return True
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    return (
        str(a.get("start") or "") == str(b.get("start") or "")
        and str(a.get("end") or "") == str(b.get("end") or "")
        and str(a.get("granularity") or "") == str(b.get("granularity") or "")
    )


def evaluate_case(case: Dict[str, Any], *, use_demos: bool) -> Dict[str, Any]:
    domain = str(case.get("domain") or "Science")
    chunk_text = str(case["chunk_text"])
    paper_year = case.get("paper_year")
    gold = case.get("expected_triplets") or case.get("expected") or []
    if not isinstance(gold, list):
        raise ValueError("expected_triplets must be a list")

    pred_models = extract_temporal_triplets(domain=domain, chunk_text=chunk_text, paper_year=paper_year, use_demos=use_demos)
    pred = [m.model_dump() for m in pred_models]

    gold_map = { _key(t): t for t in gold }
    pred_keys = [_key(t) for t in pred]
    gold_keys = list(gold_map.keys())

    matched = [k for k in pred_keys if k in gold_map]
    precision = len(matched) / max(len(pred_keys), 1)
    recall = len(matched) / max(len(gold_keys), 1)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    # time accuracy on matched triplets where gold has time specified
    time_hits = 0
    time_total = 0
    for k in matched:
        g = gold_map[k]
        # find the corresponding predicted (first match)
        p = next((x for x in pred if _key(x) == k), None)
        if p is None:
            continue
        if g.get("time") is not None:
            time_total += 1
            if _time_equal(g.get("time"), p.get("time")):
                time_hits += 1
    time_acc = time_hits / time_total if time_total else None

    return {
        "n_gold": len(gold_keys),
        "n_pred": len(pred_keys),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "time_acc": time_acc,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate temporal triplet extraction on a gold set.")
    ap.add_argument("--gold-dir", type=str, default="data/gold/triplets", help="Directory with gold JSON cases.")
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
                r = evaluate_case(case, use_demos=use_demos)
                rows.append(r)
            except Exception as e:
                rows.append({"error": str(e), "n_gold": 0, "n_pred": 0, "precision": 0, "recall": 0, "f1": 0, "time_acc": None})

        f1s = [r["f1"] for r in rows if "error" not in r]
        precs = [r["precision"] for r in rows if "error" not in r]
        recs = [r["recall"] for r in rows if "error" not in r]
        time_accs = [r["time_acc"] for r in rows if r.get("time_acc") is not None and "error" not in r]

        return {
            "n_cases": len(rows),
            "avg_precision": sum(precs) / max(len(precs), 1),
            "avg_recall": sum(recs) / max(len(recs), 1),
            "avg_f1": sum(f1s) / max(len(f1s), 1),
            "avg_time_acc": (sum(time_accs) / len(time_accs)) if time_accs else None,
            "errors": sum(1 for r in rows if "error" in r),
        }

    out: Dict[str, Any] = {"gold_dir": str(gold_dir), "n_cases": len(cases)}
    if args.variant in ("baseline", "both"):
        out["baseline"] = run(use_demos=False)
    if args.variant in ("fewshot", "both"):
        out["fewshot"] = run(use_demos=True)

    if "baseline" in out and "fewshot" in out:
        out["delta_f1"] = out["fewshot"]["avg_f1"] - out["baseline"]["avg_f1"]
        out["delta_precision"] = out["fewshot"]["avg_precision"] - out["baseline"]["avg_precision"]
        out["delta_recall"] = out["fewshot"]["avg_recall"] - out["baseline"]["avg_recall"]

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
