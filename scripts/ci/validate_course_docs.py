#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Validate the course documentation.

Checks that the course stays coherent as it grows: every week module exists
and is reachable from the syllabus, every relative link resolves, and the
machine-readable syllabus agrees with the human-readable one.

Exit code 1 on any problem, so CI fails loudly rather than letting the
course silently drift out of sync.

Usage:
    python3 scripts/ci/validate_course_docs.py [--json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COURSE_DIR = REPO_ROOT / "course"
DOCS_COURSE = REPO_ROOT / "docs" / "course"
SYLLABUS_MD = DOCS_COURSE / "syllabus.md"
SYLLABUS_JSON = DOCS_COURSE / "syllabus.json"

LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
SPDX_RE = re.compile(r"SPDX-License-Identifier:")
WEEKS = 12

problems: list[str] = []
checks_run = 0


def fail(msg: str) -> None:
    problems.append(msg)


def check(name: str, condition: bool, message: str) -> None:
    """Record a check and its failure message."""
    global checks_run
    checks_run += 1
    if not condition:
        fail(f"{name}: {message}")


def iter_course_docs() -> list[Path]:
    docs = []
    for base in (COURSE_DIR, DOCS_COURSE):
        if base.is_dir():
            docs.extend(sorted(base.rglob("*.md")))
    return docs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    # ── 1. syllabus exists and is complete ────────────────────────────────
    check("syllabus.md exists", SYLLABUS_MD.is_file(),
          f"{SYLLABUS_MD.relative_to(REPO_ROOT)} is missing")
    check("syllabus.json exists", SYLLABUS_JSON.is_file(),
          f"{SYLLABUS_JSON.relative_to(REPO_ROOT)} is missing")

    syllabus_text = SYLLABUS_MD.read_text(encoding="utf-8") if SYLLABUS_MD.is_file() else ""
    syllabus_data: dict = {}
    if SYLLABUS_JSON.is_file():
        try:
            syllabus_data = json.loads(SYLLABUS_JSON.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            fail(f"syllabus.json is not valid JSON: {e}")

    # ── 2. every week module exists and is linked from the syllabus ───────
    for week in range(1, WEEKS + 1):
        module = COURSE_DIR / "weeks" / f"week{week:02d}.md"
        check(f"week{week:02d} module exists", module.is_file(),
              f"course/weeks/week{week:02d}.md is missing")
        if module.is_file() and syllabus_text:
            name = f"week{week:02d}.md"
            check(f"week{week:02d} is linked from the syllabus", name in syllabus_text,
                  f"{name} is never referenced by the syllabus")

    # ── 3. syllabus.json agrees with the files on disk ────────────────────
    json_weeks = syllabus_data.get("weeks", [])
    check("syllabus.json lists 12 weeks", len(json_weeks) == WEEKS,
          f"found {len(json_weeks)} entries")
    for entry in json_weeks:
        for key in ("module", "agenda"):
            rel = entry.get(key)
            if not rel:
                continue
            check(f"syllabus.json {key} path resolves", (REPO_ROOT / rel).exists(),
                  f"{rel} does not exist")

    # ── 4. every week module has the required sections ────────────────────
    required = ("## Результат недели", "## Проверьте себя", "## Источники")
    for week in range(1, WEEKS + 1):
        module = COURSE_DIR / "weeks" / f"week{week:02d}.md"
        if not module.is_file():
            continue
        text = module.read_text(encoding="utf-8")
        for section in required:
            check(f"week{week:02d} has {section!r}", section in text,
                  f"missing section {section!r}")

    # ── 5. no broken relative links anywhere in the course docs ───────────
    total_links = 0
    for md in iter_course_docs():
        text = md.read_text(encoding="utf-8", errors="replace")
        for m in LINK_RE.finditer(text):
            target = m.group(2).split("#")[0].strip()
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            total_links += 1
            resolved = (md.parent / target).resolve()
            check("relative link resolves", resolved.exists(),
                  f"{md.relative_to(REPO_ROOT)} -> {target}")

    # ── 6. SPDX headers on course markdown ────────────────────────────────
    for md in iter_course_docs():
        if md.name in ("README.md",) and md.parent == DOCS_COURSE.parent:
            continue
        text = md.read_text(encoding="utf-8", errors="replace")
        if not SPDX_RE.search(text):
            fail(f"{md.relative_to(REPO_ROOT)} has no SPDX license header")

    # ── report ────────────────────────────────────────────────────────────
    if args.json:
        print(json.dumps({
            "checks_run": checks_run,
            "problems": problems,
            "relative_links_checked": total_links,
            "ok": not problems,
        }, ensure_ascii=False, indent=2))
    else:
        print(f"Course documentation validation: {checks_run} checks, "
              f"{total_links} relative links")
        if problems:
            print(f"\n{len(problems)} problem(s):")
            for p in problems:
                print(f"  - {p}")
        else:
            print("All checks passed.")

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
