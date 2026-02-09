from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json
import hashlib


def stable_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def save_paper(output_dir: Path, meta: Dict[str, Any], chunks: List[str]) -> Path:
    """Сохраняет paper как папку: meta.json + chunks.jsonl"""
    output_dir.mkdir(parents=True, exist_ok=True)
    pid = meta.get("id") or stable_id(meta.get("title", "unknown"))
    paper_dir = output_dir / pid
    paper_dir.mkdir(parents=True, exist_ok=True)

    (paper_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    with (paper_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for idx, c in enumerate(chunks):
            rec = {"chunk_id": f"{pid}:{idx}", "text": c}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return paper_dir
