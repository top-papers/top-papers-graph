from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List
from datetime import date, datetime
import json
import hashlib

from ..contracts import ChunkRecord


def stable_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def _json_default(o: Any) -> Any:
    # Make dumps robust to common non-JSON types.
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _safe_path_component(value: str, *, max_len: int = 160) -> str:
    """Return a filesystem-safe single path component.

    Important: paper ids may contain `/` (for DOI suffixes) and `:`. Those values
    must not be used directly as directory names, otherwise `Path(output_dir) / pid`
    creates nested directories like `processed_papers/doi:10.1070/...`, which breaks
    later scans that expect each paper to live under exactly one direct child folder.
    """

    raw = str(value or "").strip()
    if not raw:
        return "unknown"

    safe = raw.replace("/", "_").replace(":", "_").replace("\\", "_")
    safe = "".join(ch for ch in safe if ch.isalnum() or ch in {"_", "-", "."})
    safe = safe.strip("._") or "unknown"
    if len(safe) <= max_len:
        return safe

    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return safe[: max_len - 11].rstrip("._") + "_" + digest


def _normalize_chunk_records(pid: str, chunks: Iterable[str | Dict[str, Any] | ChunkRecord]) -> List[ChunkRecord]:
    out: List[ChunkRecord] = []
    for idx, item in enumerate(chunks):
        if isinstance(item, ChunkRecord):
            rec = item
        elif isinstance(item, dict):
            data = dict(item)
            data.setdefault("chunk_id", f"{pid}:{idx}")
            data.setdefault("paper_id", pid)
            rec = ChunkRecord.model_validate(data)
        else:
            rec = ChunkRecord(
                chunk_id=f"{pid}:{idx}",
                paper_id=pid,
                text=str(item),
                source_backend="legacy_text",
            )
        if not rec.paper_id:
            rec.paper_id = pid
        out.append(rec)
    return out


def save_paper(output_dir: Path, meta: Dict[str, Any], chunks: List[str | Dict[str, Any] | ChunkRecord]) -> Path:
    """Сохраняет paper как папку: meta.json + chunks.jsonl.

    Backward compatible with the historical `List[str]` representation while also supporting the
    richer `ChunkRecord` contract used by OCR/layout-aware ingestion.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    pid = meta.get("id") or stable_id(meta.get("title", "unknown"))
    paper_dir = output_dir / _safe_path_component(str(pid))
    paper_dir.mkdir(parents=True, exist_ok=True)

    (paper_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    chunk_records = _normalize_chunk_records(str(pid), chunks)
    with (paper_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for rec in chunk_records:
            f.write(json.dumps(rec.model_dump(mode="json"), ensure_ascii=False, default=_json_default) + "\n")
    return paper_dir
