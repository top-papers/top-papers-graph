#!/usr/bin/env python3
"""CLI wrapper for building the Task 3 Hugging Face VLM benchmark dataset."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scireason.task3_hf_benchmark import main

if __name__ == "__main__":
    raise SystemExit(main())
