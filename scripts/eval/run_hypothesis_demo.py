#!/usr/bin/env python3
"""Прогон генерации гипотез по фиксированному набору кейсов на ТЕКУЩЕЙ версии кода.

Используется как одна половина side-by-side демонстрации: один и тот же кейс-файл
запускается дважды (на разных ветках), результаты складываются в `runs/hypothesis_demo/<label>/`,
а затем `render_hypothesis_ab.py` рендерит их рядом.

Пример:
    python scripts/eval/run_hypothesis_demo.py \
        --cases data/demo/hypothesis_demo_cases.json \
        --label main \
        --out-root runs/hypothesis_demo
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return ""


def _run_one_case(case: dict, *, out_dir: Path, llm_provider: str | None, llm_model: str | None) -> dict:
    from scireason.pipeline.task3_hypothesis_generation import prepare_task3_hypothesis_bundle

    case_id = str(case["id"])
    query = str(case.get("query") or "").strip()
    if not query:
        raise ValueError(f"case {case_id}: пустой query")

    kwargs = dict(
        query=query,
        out_dir=out_dir / case_id,
        top_hypotheses=int(case.get("top_hypotheses", 5)),
        run_vlm=False,
    )
    if "top_papers" in case:
        kwargs["top_papers"] = int(case["top_papers"])
    if llm_provider:
        kwargs["llm_provider"] = llm_provider
    if llm_model:
        kwargs["llm_model"] = llm_model
    if case.get("identifiers"):
        kwargs["identifiers"] = list(case["identifiers"])
    if case.get("processed_dir"):
        kwargs["processed_dir"] = Path(case["processed_dir"])

    t0 = time.monotonic()
    result = prepare_task3_hypothesis_bundle(**kwargs)
    elapsed = time.monotonic() - t0

    return {
        "id": case_id,
        "query": query,
        "bundle_dir": str(result.bundle_dir),
        "hypotheses_path": str(result.hypotheses_path),
        "hypotheses_markdown": str(result.bundle_dir / "hypotheses_ranked.md"),
        "manifest_path": str(result.manifest_path),
        "elapsed_seconds": round(elapsed, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=Path, required=True, help="JSON-файл с полем cases: [...]")
    ap.add_argument("--label", type=str, required=True, help="Метка прогона (обычно имя ветки: main / feature/...)")
    ap.add_argument("--out-root", type=Path, default=ROOT / "runs" / "hypothesis_demo")
    ap.add_argument("--llm-provider", type=str, default=None)
    ap.add_argument("--llm-model", type=str, default=None)
    args = ap.parse_args()

    cases_doc = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = cases_doc.get("cases") or []
    if not cases:
        raise SystemExit(f"в {args.cases} нет поля 'cases' или оно пустое")

    label_slug = args.label.replace("/", "__")
    out_dir = args.out_root / label_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{args.label}] прогон {len(cases)} кейсов → {out_dir}")
    rows: list[dict] = []
    for i, case in enumerate(cases, start=1):
        print(f"  [{i}/{len(cases)}] {case.get('id')}: {case.get('query')!r}")
        try:
            rows.append(_run_one_case(case, out_dir=out_dir, llm_provider=args.llm_provider, llm_model=args.llm_model))
        except Exception as e:
            print(f"    ОШИБКА: {e}")
            traceback.print_exc()
            rows.append({"id": case.get("id"), "query": case.get("query"), "error": str(e)})

    manifest = {
        "label": args.label,
        "label_slug": label_slug,
        "generated_at": _utc_now(),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "cases_source": str(args.cases),
        "llm_provider": args.llm_provider,
        "llm_model": args.llm_model,
        "cases": rows,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{args.label}] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
