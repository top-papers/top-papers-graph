#!/usr/bin/env python3
"""Side-by-side рендер двух прогонов генерации гипотез.

Никаких метрик и скоринга — просто кладёт результат версии A и версии B рядом по каждому
кейсу, чтобы человек глазами видел разницу.

Пример:
    python scripts/eval/render_hypothesis_ab.py \
        --left  runs/hypothesis_demo/main/manifest.json \
        --right runs/hypothesis_demo/feature__quality-pipeline-v2/manifest.json \
        --out   runs/hypothesis_demo/report
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_markdown(case: dict) -> str:
    if "error" in case:
        return f"_ошибка прогона: {case['error']}_"
    md_path = Path(case.get("hypotheses_markdown") or "")
    if not md_path.exists():
        return f"_не найден файл: {md_path}_"
    return md_path.read_text(encoding="utf-8")


def _index_cases(manifest: dict) -> dict[str, dict]:
    return {str(c.get("id")): c for c in (manifest.get("cases") or [])}


def _render_markdown(left: dict, right: dict) -> str:
    left_cases = _index_cases(left)
    right_cases = _index_cases(right)
    all_ids = sorted(set(left_cases) | set(right_cases))

    lines: list[str] = []
    lines.append(f"# A/B демонстрация генерации гипотез")
    lines.append("")
    lines.append(f"- **Слева ({left.get('label')}):** sha `{left.get('git_sha','')[:10]}`, ветка `{left.get('git_branch','')}`, время {left.get('generated_at')}")
    lines.append(f"- **Справа ({right.get('label')}):** sha `{right.get('git_sha','')[:10]}`, ветка `{right.get('git_branch','')}`, время {right.get('generated_at')}")
    lines.append("")
    lines.append(f"Всего кейсов: {len(all_ids)}")
    lines.append("")

    for cid in all_ids:
        lc = left_cases.get(cid) or {}
        rc = right_cases.get(cid) or {}
        query = lc.get("query") or rc.get("query") or ""
        lines.append(f"---")
        lines.append("")
        lines.append(f"## {cid}")
        lines.append("")
        lines.append(f"**Запрос:** {query}")
        lines.append("")
        lines.append(f"### {left.get('label')}")
        lines.append("")
        lines.append(_read_markdown(lc))
        lines.append("")
        lines.append(f"### {right.get('label')}")
        lines.append("")
        lines.append(_read_markdown(rc))
        lines.append("")
    return "\n".join(lines)


_HTML_TEMPLATE = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>A/B демонстрация генерации гипотез</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1a1a1a; }}
  h1 {{ border-bottom: 2px solid #333; padding-bottom: 8px; }}
  .meta {{ background: #f6f6f6; padding: 10px 14px; border-radius: 6px; margin-bottom: 18px; font-size: 14px; }}
  .case {{ margin-top: 28px; border-top: 1px solid #ccc; padding-top: 14px; }}
  .case h2 {{ margin-bottom: 4px; }}
  .query {{ color: #444; font-style: italic; margin-bottom: 12px; }}
  .ab {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .col {{ background: #fafafa; border: 1px solid #ddd; border-radius: 6px; padding: 12px 16px; }}
  .col h3 {{ margin-top: 0; color: #2a4d7a; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; }}
  .col pre {{ white-space: pre-wrap; word-wrap: break-word; font-family: ui-monospace, monospace; font-size: 13px; line-height: 1.45; }}
  .err {{ color: #a33; font-style: italic; }}
</style>
</head>
<body>
<h1>A/B демонстрация генерации гипотез</h1>
<div class="meta">
  <div><b>Слева ({left_label}):</b> sha <code>{left_sha}</code>, ветка <code>{left_branch}</code>, {left_time}</div>
  <div><b>Справа ({right_label}):</b> sha <code>{right_sha}</code>, ветка <code>{right_branch}</code>, {right_time}</div>
  <div>Всего кейсов: {n_cases}</div>
</div>
{cases_html}
</body>
</html>
"""


def _render_html(left: dict, right: dict) -> str:
    left_cases = _index_cases(left)
    right_cases = _index_cases(right)
    all_ids = sorted(set(left_cases) | set(right_cases))

    parts: list[str] = []
    for cid in all_ids:
        lc = left_cases.get(cid) or {}
        rc = right_cases.get(cid) or {}
        query = lc.get("query") or rc.get("query") or ""
        parts.append(
            f'<div class="case">'
            f'<h2>{html.escape(cid)}</h2>'
            f'<div class="query">Запрос: {html.escape(query)}</div>'
            f'<div class="ab">'
            f'<div class="col"><h3>{html.escape(str(left.get("label")))}</h3>'
            f'<pre>{html.escape(_read_markdown(lc))}</pre></div>'
            f'<div class="col"><h3>{html.escape(str(right.get("label")))}</h3>'
            f'<pre>{html.escape(_read_markdown(rc))}</pre></div>'
            f'</div></div>'
        )
    return _HTML_TEMPLATE.format(
        left_label=html.escape(str(left.get("label"))),
        right_label=html.escape(str(right.get("label"))),
        left_sha=html.escape((left.get("git_sha") or "")[:10]),
        right_sha=html.escape((right.get("git_sha") or "")[:10]),
        left_branch=html.escape(left.get("git_branch") or ""),
        right_branch=html.escape(right.get("git_branch") or ""),
        left_time=html.escape(left.get("generated_at") or ""),
        right_time=html.escape(right.get("generated_at") or ""),
        n_cases=len(all_ids),
        cases_html="\n".join(parts),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--left", type=Path, required=True, help="manifest.json первого прогона (обычно main)")
    ap.add_argument("--right", type=Path, required=True, help="manifest.json второго прогона (наши доработки)")
    ap.add_argument("--out", type=Path, required=True, help="Папка, куда сложить report.html и report.md")
    args = ap.parse_args()

    left = _load_manifest(args.left)
    right = _load_manifest(args.right)

    args.out.mkdir(parents=True, exist_ok=True)
    md_path = args.out / "report.md"
    html_path = args.out / "report.html"
    md_path.write_text(_render_markdown(left, right), encoding="utf-8")
    html_path.write_text(_render_html(left, right), encoding="utf-8")
    print(f"report.md:   {md_path}")
    print(f"report.html: {html_path}")


if __name__ == "__main__":
    main()
