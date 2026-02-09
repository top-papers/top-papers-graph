#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import json

from scireason.demos.exporter import index_demos_from_experts


def main() -> None:
    ap = argparse.ArgumentParser(description="Build and index retrieval few-shot demo store from data/experts artifacts.")
    ap.add_argument("--experts-dir", type=str, default="data/experts", help="Root dir with expert artifacts.")
    args = ap.parse_args()

    counts = index_demos_from_experts(Path(args.experts_dir))
    print(json.dumps({"indexed": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
