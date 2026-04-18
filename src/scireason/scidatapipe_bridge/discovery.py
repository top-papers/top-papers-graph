from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

TASK1_EXTENSIONS = {".yaml", ".yml"}
TASK2_ARCHIVE_EXTENSIONS = {".zip"}
TASK2_MARKERS = (
    "edge_reviews.json",
    "review_templates",
    "task2_notebook_manifest.json",
    "review_state_latest.json",
    "temporal_corrections.json",
)


def discover_task1_files(paths: Sequence[Path] = (), dirs: Sequence[Path] = (), *, recursive: bool = True) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        path = Path(path)
        if path.is_file() and path.suffix.lower() in TASK1_EXTENSIONS:
            _append_unique(out, seen, path.resolve())
        elif path.is_dir():
            for candidate in _iter_task1_in_dir(path, recursive=recursive):
                _append_unique(out, seen, candidate.resolve())
    for root in dirs:
        for candidate in _iter_task1_in_dir(Path(root), recursive=recursive):
            _append_unique(out, seen, candidate.resolve())
    return out


def discover_task2_inputs(paths: Sequence[Path] = (), dirs: Sequence[Path] = (), *, recursive: bool = True) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        path = Path(path)
        if path.is_file() and path.suffix.lower() in TASK2_ARCHIVE_EXTENSIONS:
            _append_unique(out, seen, path.resolve())
        elif path.is_dir():
            if is_task2_bundle_dir(path):
                _append_unique(out, seen, path.resolve())
            else:
                for candidate in _iter_task2_in_dir(path, recursive=recursive):
                    _append_unique(out, seen, candidate.resolve())
    for root in dirs:
        for candidate in _iter_task2_in_dir(Path(root), recursive=recursive):
            _append_unique(out, seen, candidate.resolve())
    return out


def discover_mixed_inputs(input_dirs: Sequence[Path] = (), *, recursive: bool = True) -> tuple[list[Path], list[Path]]:
    task1: list[Path] = []
    task2: list[Path] = []
    seen_task1: set[str] = set()
    seen_task2: set[str] = set()
    for root in input_dirs:
        root = Path(root)
        if not root.exists():
            continue
        if root.is_file():
            if root.suffix.lower() in TASK1_EXTENSIONS:
                _append_unique(task1, seen_task1, root.resolve())
            elif root.suffix.lower() in TASK2_ARCHIVE_EXTENSIONS:
                _append_unique(task2, seen_task2, root.resolve())
            continue
        if is_task2_bundle_dir(root):
            _append_unique(task2, seen_task2, root.resolve())
            continue
        _scan_mixed_dir(root, task1, seen_task1, task2, seen_task2, recursive=recursive)
    return task1, task2


def is_task2_bundle_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any((path / marker).exists() for marker in TASK2_MARKERS)


def _iter_task1_in_dir(root: Path, *, recursive: bool) -> Iterable[Path]:
    if not root.exists() or not root.is_dir():
        return []
    pattern = "**/*" if recursive else "*"
    candidates = sorted(p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in TASK1_EXTENSIONS)
    return candidates


def _iter_task2_in_dir(root: Path, *, recursive: bool) -> Iterable[Path]:
    if not root.exists() or not root.is_dir():
        return []
    out: list[Path] = []
    pattern = "**/*" if recursive else "*"
    for candidate in sorted(root.glob(pattern)):
        if candidate.is_file() and candidate.suffix.lower() in TASK2_ARCHIVE_EXTENSIONS:
            out.append(candidate)
        elif candidate.is_dir() and is_task2_bundle_dir(candidate):
            out.append(candidate)
    return out


def _scan_mixed_dir(
    root: Path,
    task1: list[Path],
    seen_task1: set[str],
    task2: list[Path],
    seen_task2: set[str],
    *,
    recursive: bool,
) -> None:
    try:
        children = sorted(root.iterdir())
    except Exception:
        return
    for child in children:
        if child.is_file():
            suffix = child.suffix.lower()
            if suffix in TASK1_EXTENSIONS:
                _append_unique(task1, seen_task1, child.resolve())
            elif suffix in TASK2_ARCHIVE_EXTENSIONS:
                _append_unique(task2, seen_task2, child.resolve())
            continue
        if not child.is_dir():
            continue
        if is_task2_bundle_dir(child):
            _append_unique(task2, seen_task2, child.resolve())
            continue
        if recursive:
            _scan_mixed_dir(child, task1, seen_task1, task2, seen_task2, recursive=recursive)


def _append_unique(items: list[Path], seen: set[str], path: Path) -> None:
    key = str(path)
    if key in seen:
        return
    seen.add(key)
    items.append(path)
