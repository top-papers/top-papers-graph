from __future__ import annotations

from scireason.llm import _json_loads_best_effort


def test_json_loads_best_effort_accepts_literal_newlines_inside_strings() -> None:
    payload = '{"triplets": [{"subject": "Dusty plasma\nhexatic phase", "predicate": "supports", "object": "KTHNY", "confidence": 0.9}]}'

    data = _json_loads_best_effort(payload)

    assert data["triplets"][0]["subject"] == "Dusty plasma\nhexatic phase"
    assert data["triplets"][0]["object"] == "KTHNY"



def test_json_loads_best_effort_accepts_markdown_wrapped_json_with_control_chars() -> None:
    payload = '```json\n{"triplets": [{"subject": "Nefedov\t1997", "predicate": "observes", "object": "intermediate state\nwith orientational order", "confidence": 0.8}]}\n```'

    data = _json_loads_best_effort(payload)

    assert data["triplets"][0]["subject"] == "Nefedov\t1997"
    assert data["triplets"][0]["object"] == "intermediate state\nwith orientational order"
