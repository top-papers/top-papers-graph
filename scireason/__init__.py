"""Compatibility shim for running notebooks and scripts from a source checkout.

This makes ``import scireason`` work even before ``pip install -e .`` completes,
by extending the package search path to the real implementation under ``src/``.
The editable install remains the canonical installation path.
"""
from __future__ import annotations

from pathlib import Path

_pkg_root = Path(__file__).resolve().parent
_src_pkg = _pkg_root.parent / "src" / "scireason"

if _src_pkg.is_dir():
    _src_str = str(_src_pkg)
    if _src_str not in __path__:
        __path__.append(_src_str)
    _src_init = _src_pkg / "__init__.py"
    if _src_init.exists():
        exec(compile(_src_init.read_text(encoding="utf-8"), str(_src_init), "exec"))
