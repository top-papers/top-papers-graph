from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .schemas import DemoExample


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def render_demos_block(
    items: List[Dict[str, Any]],
    *,
    max_total_chars: int = 3500,
    title: str = "Эталонные примеры",
) -> str:
    """Render retrieved demos as a compact few-shot block for the prompt."""
    if not items:
        return ""

    parts: List[str] = [f"{title} (few-shot):"]
    budget = max_total_chars
    for i, it in enumerate(items, start=1):
        demo: DemoExample = it["demo"]
        inp = demo.input
        out = demo.output
        block = f"\nПример {i}\nINPUT:\n{_json(inp)}\nOUTPUT:\n{_json(out)}\n"
        if len(block) > budget:
            # stop when budget exhausted
            break
        parts.append(block)
        budget -= len(block)

    if len(parts) == 1:
        return ""
    return "\n".join(parts).strip() + "\n\n"
