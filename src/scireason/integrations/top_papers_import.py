from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json

from ..ingest.store import stable_id


def load_top_papers_json(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected a list of papers (JSON array).")
    return data


def to_scireason_meta(p: Dict[str, Any]) -> Dict[str, Any]:
    # Prefer DOI or paperId as stable ID
    pid = p.get("doi") or p.get("paperId") or stable_id(p.get("title", "unknown"))
    return {
        "id": pid,
        "title": p.get("title", ""),
        "year": _year_from_date(p.get("publication_date", "")),
        "source": p.get("source", ""),
        "url": p.get("url", ""),
        "abstract": p.get("abstract", ""),
        "authors": p.get("authors", ""),
        "external": {"doi": p.get("doi", ""), "paperId": p.get("paperId", "")},
    }


def _year_from_date(s: str) -> int | None:
    if not s:
        return None
    # formats: YYYY-MM-DD, YYYY, etc.
    try:
        return int(str(s)[:4])
    except Exception:
        return None


def export_meta_files(top_papers_json: Path, out_dir: Path) -> List[Path]:
    papers = load_top_papers_json(top_papers_json)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_files = []
    for p in papers:
        meta = to_scireason_meta(p)
        fpath = out_dir / f"{meta['id']}.meta.json"
        fpath.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        out_files.append(fpath)
    return out_files
