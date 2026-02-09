from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict, List

from pydantic import TypeAdapter

from ..schemas import HypothesisDraft, CritiqueReview


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_hypothesis(path: Path) -> HypothesisDraft:
    return TypeAdapter(HypothesisDraft).validate_python(load_json(path))


def validate_critique(path: Path) -> CritiqueReview:
    return TypeAdapter(CritiqueReview).validate_python(load_json(path))
