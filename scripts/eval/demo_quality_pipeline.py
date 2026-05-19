#!/usr/bin/env python3
"""Демонстрация работы пайплайна качества.

Запускает извлечение триплетов на встроенном мини-корпусе дважды:
1) Baseline — без оценочной функции (score ≈ const, нельзя отличить хорошие от плохих)
2) С оценочной функцией — калиброванный score ∈ [0,1], ранжирование работает

Затем запускает независимую оценку (LLM-критик) на экспертных траекториях.

Требования: ollama + qwen2.5:7b-instruct
Запуск:
    source .venv/bin/activate
    python scripts/eval/demo_quality_pipeline.py
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

OLLAMA_MODEL = "qwen2.5:7b-instruct"


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def run_demo(label: str, scorer_enabled: bool, out_dir: str) -> dict:
    """Запустить demo-run через Python API и вернуть temporal_kg.json."""
    from scireason.config import settings
    from scireason.pipeline.demo import run_demo_pipeline

    settings.llm_provider = "ollama"
    settings.llm_model = OLLAMA_MODEL
    settings.assertion_scorer_enabled = scorer_enabled

    print(f"  Запуск: {label} (scorer={'ON' if scorer_enabled else 'OFF'})...")
    t0 = time.monotonic()
    try:
        run_path = run_demo_pipeline(
            query="temporal knowledge graph hypothesis",
            domain_id="science",
            edge_mode="llm_triplets",
            out_dir=Path(out_dir),
            use_llm_for_hypotheses=False,
        )
    except Exception as e:
        print(f"  ОШИБКА: {e}")
        return {}
    elapsed = time.monotonic() - t0

    kg_path = run_path / "temporal_kg.json"
    if not kg_path.exists():
        print(f"  temporal_kg.json не найден в {run_path}")
        return {}

    kg = json.loads(kg_path.read_text())
    print(f"  Готово за {elapsed:.1f}с — {len(kg.get('edges',[]))} рёбер")
    return kg


def show_comparison(baseline: dict, quality: dict) -> None:
    """Вывести сравнительную таблицу."""
    section("СРАВНЕНИЕ: Baseline vs Оценочная функция")

    for label, kg in [("Baseline (scorer OFF)", baseline), ("С оценочной функцией", quality)]:
        edges = kg.get("edges", [])
        if not edges:
            print(f"  {label}: нет данных\n")
            continue

        scores = [e.get("score", 0) for e in edges]
        has_qs = any("quality_score" in (e.get("features") or {}) for e in edges)
        causal = {"causes", "leads_to", "results_in", "improves", "reduces",
                  "increases", "prevents", "inhibits", "drives", "enables"}
        n_causal = sum(1 for e in edges if e.get("predicate") in causal)

        print(f"  {label}:")
        print(f"    Рёбер:           {len(edges)}")
        print(f"    Score диапазон:  [{min(scores):.3f} .. {max(scores):.3f}]")
        print(f"    Score среднее:   {sum(scores)/len(scores):.3f}")
        print(f"    Quality score:   {'есть' if has_qs else 'нет'}")
        print(f"    Каузальных:      {n_causal}/{len(edges)}")
        print()

        print(f"    Рёбра (ранжированы по score):")
        for e in sorted(edges, key=lambda x: -x.get("score", 0)):
            qs = (e.get("features") or {}).get("quality_score")
            qs_s = f"  quality={qs:.3f}" if qs is not None else ""
            print(f"      {e['score']:.3f}{qs_s}  [{e['predicate']}] {e['source'][:35]} → {e['target'][:35]}")
        print()


def run_critic() -> None:
    """Запустить независимую оценку LLM-критиком."""
    section("НЕЗАВИСИМАЯ ОЦЕНКА (LLM-as-Judge)")

    from scireason.eval.critic_agent import (
        build_test_set_from_trajectories,
        compute_agreement_metrics,
        evaluate_triplets,
    )

    traj_dir = str(ROOT / "data" / "experts" / "trajectories")
    triplets = build_test_set_from_trajectories(traj_dir, n_noise=15)
    n_valid = sum(1 for t in triplets if t["expected_label"] == "valid")
    n_noise = sum(1 for t in triplets if t["expected_label"] == "noise")
    print(f"  Тестовый набор: {len(triplets)} триплетов ({n_valid} экспертных + {n_noise} синтетический шум)")
    print(f"  Модель-критик: {OLLAMA_MODEL}")
    print(f"  Запуск (может занять ~2 мин)...\n")

    report = evaluate_triplets(triplets, model=OLLAMA_MODEL)
    metrics = compute_agreement_metrics(report, triplets)

    print(f"  Результаты критика:")
    print(f"    valid:      {report.n_valid}")
    print(f"    noise:      {report.n_noise}")
    print(f"    borderline: {report.n_borderline}")
    print()
    print(f"  Метрики (binary: valid vs non-valid):")
    for k, v in metrics["binary_metrics"].items():
        print(f"    {k}: {v}")
    print()
    print(f"  Confusion matrix:")
    for k, v in sorted(metrics["confusion_matrix"].items()):
        if v > 0:
            print(f"    {k}: {v}")
    print()
    print(f"  Время: {report.total_time_seconds:.0f}с")


def main() -> None:
    section("ДЕМОНСТРАЦИЯ ПАЙПЛАЙНА КАЧЕСТВА")
    print(f"  Корпус: встроенный мини-набор (3 статьи)")
    print(f"  LLM: ollama / {OLLAMA_MODEL}")

    # 1. Baseline
    baseline = run_demo("Baseline", scorer_enabled=False, out_dir=str(ROOT / "runs" / "demo_baseline"))

    # 2. С оценочной функцией
    quality = run_demo("Quality", scorer_enabled=True, out_dir=str(ROOT / "runs" / "demo_quality"))

    # 3. Сравнение
    if baseline and quality:
        show_comparison(baseline, quality)

    # 4. Независимая оценка
    run_critic()

    section("ГОТОВО")


if __name__ == "__main__":
    main()
