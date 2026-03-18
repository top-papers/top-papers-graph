from __future__ import annotations

from typing import List
from pydantic import TypeAdapter

from ..llm import chat_json
from ..schemas import Triplet


TRIPLET_SCHEMA_HINT = """Ожидается JSON массив объектов:
[{ "subject": "...", "predicate": "...", "object": "...", "confidence": 0.0-1.0, "polarity": "supports|contradicts|unknown"}]
Правила:
- subject/object: короткие сущности (термины)
- predicate: короткий глагол/связка (например "inhibits", "causes", "correlates_with")
- НЕ выдумывай факты. Если не уверен — polarity="unknown" и confidence <= 0.5.
"""


def extract_triplets(domain: str, chunk_text: str) -> List[Triplet]:
    system = f"""Ты — помощник исследователя в области {domain}.
Твоя задача — извлечь из фрагмента текста научные утверждения в виде триплетов (S-P-O).
"""
    user = f"""Фрагмент:
{chunk_text}

Извлеки 3-10 самых важных триплетов."""
    data = chat_json(system=system, user=user, schema_hint=TRIPLET_SCHEMA_HINT, temperature=0.0)
    adapter = TypeAdapter(List[Triplet])
    return adapter.validate_python(data)
