#!/usr/bin/env python3
"""Run adversarial LLM evaluation on expert trajectory triplets.

Builds a test set from expert trajectories + synthetic noise,
runs an independent LLM critic (ollama), and computes agreement metrics.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from scireason.eval.critic_agent import (
    build_test_set_from_trajectories,
    compute_agreement_metrics,
    evaluate_triplets,
)


def main() -> None:
    trajectory_dir = str(
        Path(__file__).resolve().parent.parent.parent / "data" / "experts" / "trajectories"
    )
    out_dir = Path(__file__).resolve().parent.parent.parent / "data" / "derived"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = "qwen2.5:7b-instruct"

    print(f"Building test set from {trajectory_dir} ...")
    triplets = build_test_set_from_trajectories(trajectory_dir, n_noise=15)
    print(f"  {len(triplets)} triplets ({sum(1 for t in triplets if t['expected_label']=='valid')} valid, "
          f"{sum(1 for t in triplets if t['expected_label']=='noise')} noise)")

    print(f"\nRunning adversarial critic ({model}) ...")
    report = evaluate_triplets(triplets, model=model)

    print(f"\nCritic summary: {json.dumps(report.summary(), indent=2)}")

    metrics = compute_agreement_metrics(report, triplets)
    print(f"\nAgreement metrics (binary: valid vs non-valid):")
    for k, v in metrics["binary_metrics"].items():
        print(f"  {k}: {v}")
    print(f"\nConfusion matrix (expected_predicted):")
    for k, v in sorted(metrics["confusion_matrix"].items()):
        if v > 0:
            print(f"  {k}: {v}")

    # Save full results
    results = {
        "model": model,
        "test_set_size": len(triplets),
        "critic_summary": report.summary(),
        "agreement_metrics": metrics,
        "verdicts": [
            {
                "subject": v.subject[:80],
                "predicate": v.predicate,
                "object": v.object[:80],
                "evidence": v.evidence[:200],
                "critic_verdict": v.verdict,
                "critic_rationale": v.rationale,
                "expected_label": triplets[i].get("expected_label", ""),
                "source_file": triplets[i].get("source_file", ""),
                "latency_s": v.latency_seconds,
            }
            for i, v in enumerate(report.verdicts)
        ],
    }
    out_path = out_dir / "critic_evaluation_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
