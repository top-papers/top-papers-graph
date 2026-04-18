"""Light-weight JSON / JSONL I/O helpers + jsonschema validation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

try:
    import jsonschema  # type: ignore
except ImportError:  # pragma: no cover - jsonschema is in requirements
    jsonschema = None  # type: ignore


def validate(doc: Any, schema: dict) -> list[str]:
    """Return a list of human-readable validation errors (empty = valid)."""
    if jsonschema is None:
        return ["jsonschema not installed; skipping validation"]
    validator = jsonschema.Draft202012Validator(schema)
    return [
        f"{'/'.join(str(p) for p in err.absolute_path) or '<root>'}: {err.message}"
        for err in validator.iter_errors(doc)
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_jsonl(path: Path, rows: Iterable[Any]) -> int:
    """Write an iterable of dicts / pydantic-dumped rows as JSONL.

    Returns the number of rows written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for _ in fh)


__all__ = ["validate", "read_json", "write_json", "write_jsonl", "count_lines"]
