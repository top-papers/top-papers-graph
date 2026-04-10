from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class AnnoyBundle:
    index_dir: Path
    manifest_path: Path
    index_path: Optional[Path]
    ids_path: Path
    metadata_path: Path
    vectors_path: Path
    size: int
    dim: int
    metric: str
    n_trees: int
    backend: str


_DEFAULT_INDEX_NAME = "chunks.ann"
_DEFAULT_IDS_NAME = "item_ids.json"
_DEFAULT_METADATA_NAME = "item_metadata.jsonl"
_DEFAULT_VECTORS_NAME = "vectors.npy"
_DEFAULT_MANIFEST_NAME = "annoy_manifest.json"


def annoy_available() -> bool:
    return importlib.util.find_spec("annoy") is not None


def _coerce_metric(metric: str) -> str:
    value = str(metric or "angular").strip().lower()
    return value if value in {"angular", "euclidean", "manhattan", "dot"} else "angular"


def _normalize_vector(vec: Sequence[float], dim: int) -> np.ndarray:
    arr = np.asarray(list(vec), dtype=np.float32).reshape(-1)
    if arr.size == dim:
        return arr
    out = np.zeros((dim,), dtype=np.float32)
    if dim <= 0:
        return out
    n = min(dim, int(arr.size))
    if n > 0:
        out[:n] = arr[:n]
    return out


def _matrix_from_vectors(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)
    dim = max(len(list(v)) for v in vectors)
    if dim <= 0:
        return np.zeros((len(vectors), 0), dtype=np.float32)
    rows = [_normalize_vector(v, dim) for v in vectors]
    return np.vstack(rows).astype(np.float32, copy=False)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    b_norm = np.linalg.norm(b, axis=1)
    a_norm = float(np.linalg.norm(a))
    denom = np.maximum(b_norm * max(a_norm, 1e-12), 1e-12)
    return (b @ a) / denom


def _score_rows(query: np.ndarray, matrix: np.ndarray, metric: str) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    metric = _coerce_metric(metric)
    if metric == "angular":
        return _cosine_similarity(query, matrix)
    if metric == "dot":
        return (matrix @ query).astype(np.float32, copy=False)
    if metric == "euclidean":
        return -np.linalg.norm(matrix - query.reshape(1, -1), axis=1)
    if metric == "manhattan":
        return -np.abs(matrix - query.reshape(1, -1)).sum(axis=1)
    return _cosine_similarity(query, matrix)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def build_annoy_index(
    vectors: Sequence[Sequence[float]],
    item_ids: Sequence[str],
    out_dir: Path,
    *,
    metric: str = "angular",
    n_trees: int = 32,
    item_payloads: Optional[Sequence[Dict[str, Any]]] = None,
    index_name: str = _DEFAULT_INDEX_NAME,
) -> AnnoyBundle:
    """Build a file-backed Annoy index when the dependency is available.

    The function always persists a NumPy matrix sidecar so retrieval can still work in
    environments where the optional `annoy` package is not installed.
    """

    if len(item_ids) != len(vectors):
        raise ValueError("item_ids and vectors must have the same length")
    if item_payloads is not None and len(item_payloads) != len(vectors):
        raise ValueError("item_payloads and vectors must have the same length")

    out_dir.mkdir(parents=True, exist_ok=True)
    metric = _coerce_metric(metric)
    n_trees = max(1, int(n_trees or 1))

    matrix = _matrix_from_vectors(vectors)
    dim = int(matrix.shape[1]) if matrix.ndim == 2 else 0
    size = int(matrix.shape[0]) if matrix.ndim == 2 else 0

    ids_path = out_dir / _DEFAULT_IDS_NAME
    metadata_path = out_dir / _DEFAULT_METADATA_NAME
    vectors_path = out_dir / _DEFAULT_VECTORS_NAME
    manifest_path = out_dir / _DEFAULT_MANIFEST_NAME
    index_path = out_dir / str(index_name or _DEFAULT_INDEX_NAME)

    ids_path.write_text(json.dumps([str(x) for x in item_ids], ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(vectors_path, matrix)

    with metadata_path.open("w", encoding="utf-8") as fh:
        for idx, item_id in enumerate(item_ids):
            payload = dict((item_payloads[idx] if item_payloads is not None else {}) or {})
            payload.setdefault("item_id", str(item_id))
            payload.setdefault("item_index", idx)
            fh.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")

    backend = "numpy_fallback"
    if size > 0 and dim > 0 and annoy_available():
        try:
            from annoy import AnnoyIndex  # type: ignore

            index = AnnoyIndex(dim, metric)
            for idx in range(size):
                index.add_item(idx, matrix[idx].tolist())
            index.build(n_trees)
            index.save(str(index_path))
            backend = "annoy"
        except Exception:
            backend = "numpy_fallback"
            try:
                if index_path.exists():
                    index_path.unlink()
            except Exception:
                pass

    manifest = {
        "backend": backend,
        "metric": metric,
        "n_trees": n_trees,
        "size": size,
        "dim": dim,
        "index_path": str(index_path) if backend == "annoy" else "",
        "ids_path": str(ids_path),
        "metadata_path": str(metadata_path),
        "vectors_path": str(vectors_path),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return AnnoyBundle(
        index_dir=out_dir,
        manifest_path=manifest_path,
        index_path=index_path if backend == "annoy" else None,
        ids_path=ids_path,
        metadata_path=metadata_path,
        vectors_path=vectors_path,
        size=size,
        dim=dim,
        metric=metric,
        n_trees=n_trees,
        backend=backend,
    )


def _read_metadata(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            row = {"raw": line}
        if isinstance(row, dict):
            out.append(row)
        else:
            out.append({"value": row})
    return out


def _resolve_bundle(index_dir_or_bundle: Path | AnnoyBundle) -> AnnoyBundle:
    if isinstance(index_dir_or_bundle, AnnoyBundle):
        return index_dir_or_bundle
    root = Path(index_dir_or_bundle)
    manifest_path = root / _DEFAULT_MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Annoy manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    index_path = Path(str(payload.get("index_path") or "")) if payload.get("index_path") else None
    return AnnoyBundle(
        index_dir=root,
        manifest_path=manifest_path,
        index_path=index_path,
        ids_path=Path(str(payload.get("ids_path") or root / _DEFAULT_IDS_NAME)),
        metadata_path=Path(str(payload.get("metadata_path") or root / _DEFAULT_METADATA_NAME)),
        vectors_path=Path(str(payload.get("vectors_path") or root / _DEFAULT_VECTORS_NAME)),
        size=int(payload.get("size") or 0),
        dim=int(payload.get("dim") or 0),
        metric=_coerce_metric(str(payload.get("metric") or "angular")),
        n_trees=max(1, int(payload.get("n_trees") or 1)),
        backend=str(payload.get("backend") or "numpy_fallback"),
    )


def search_annoy_index(
    index_dir_or_bundle: Path | AnnoyBundle,
    query_vector: Sequence[float],
    *,
    top_k: int = 5,
    search_k: int = -1,
) -> List[Dict[str, Any]]:
    bundle = _resolve_bundle(index_dir_or_bundle)
    if bundle.size <= 0 or bundle.dim <= 0 or top_k <= 0:
        return []

    item_ids = json.loads(bundle.ids_path.read_text(encoding="utf-8")) if bundle.ids_path.exists() else []
    metadata = _read_metadata(bundle.metadata_path)
    query = _normalize_vector(query_vector, bundle.dim)

    rows: List[Dict[str, Any]] = []
    if bundle.backend == "annoy" and bundle.index_path is not None and bundle.index_path.exists() and annoy_available():
        try:
            from annoy import AnnoyIndex  # type: ignore

            index = AnnoyIndex(bundle.dim, bundle.metric)
            index.load(str(bundle.index_path))
            item_indexes, distances = index.get_nns_by_vector(query.tolist(), int(top_k), search_k=search_k, include_distances=True)
            for item_index, distance in zip(item_indexes, distances):
                payload = dict(metadata[item_index]) if item_index < len(metadata) else {}
                payload.setdefault("item_id", item_ids[item_index] if item_index < len(item_ids) else str(item_index))
                payload.setdefault("item_index", int(item_index))
                payload["distance"] = float(distance)
                rows.append(payload)
            return rows
        except Exception:
            pass

    matrix = np.load(bundle.vectors_path)
    scores = _score_rows(query, matrix, bundle.metric)
    if scores.size <= 0:
        return []
    best = np.argsort(scores)[::-1][: int(top_k)]
    for item_index in best.tolist():
        payload = dict(metadata[item_index]) if item_index < len(metadata) else {}
        payload.setdefault("item_id", item_ids[item_index] if item_index < len(item_ids) else str(item_index))
        payload.setdefault("item_index", int(item_index))
        payload["score"] = float(scores[item_index])
        rows.append(payload)
    return rows
