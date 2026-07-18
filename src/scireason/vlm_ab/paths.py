# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Safe local paths with explicit support for Hugging Face cache symlinks."""

from __future__ import annotations

import os
from pathlib import Path


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _allowed_targets(root: Path) -> list[Path]:
    allowed = [root.resolve(strict=True)]
    # HF snapshots contain trusted links into the repository-local sibling blobs/.
    if root.parent.name == "snapshots":
        blobs = root.parent.parent / "blobs"
        if blobs.is_dir():
            allowed.append(blobs.resolve(strict=True))
    return allowed


def resolve_dataset_file(root: Path, candidate: Path) -> Path:
    """Resolve a file lexically under root, allowing only HF snapshot blob links."""

    root = root.resolve(strict=True)
    joined = candidate if candidate.is_absolute() else root / candidate
    lexical = _lexical_absolute(joined)
    try:
        lexical.relative_to(_lexical_absolute(root))
    except ValueError as exc:
        raise ValueError(f"path escapes dataset root: {candidate}") from exc
    try:
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FileNotFoundError(f"file does not exist: {candidate}") from exc
    if not any(_is_relative_to(resolved, allowed_root) for allowed_root in _allowed_targets(root)):
        raise ValueError(f"path resolves outside the dataset or trusted HF blob store: {candidate}")
    if not resolved.is_file():
        raise FileNotFoundError(f"path is not a file: {candidate}")
    return resolved


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


__all__ = ["resolve_dataset_file"]
