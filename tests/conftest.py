"""Pytest configuration.

This repository is frequently executed in two modes:
1) installed as an editable package (`pip install -e .`)
2) executed straight from the source tree.

To make `pytest -q` work in both modes, we add `src/` to `sys.path` when needed.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
