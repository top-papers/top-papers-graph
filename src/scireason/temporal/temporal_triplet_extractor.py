from __future__ import annotations

from typing import List, Optional
from pydantic import TypeAdapter

from ..config import settings
from ..llm import chat_json
from ..demos.retriever import retrieve_demos
from ..demos.render import render_demos_block
from .schemas import TemporalTriplet
from .time_parse import default_time_from_paper_year


TEMPORAL_TRIPLET_SCHEMA_HINT = """Ожидается JSON массив объектов:
[
  {
    "subject": "...",
    "predicate": "...",
    "object": "...",
    "confidence": 0.0-1.0,
    "polarity": "supports|contradicts|unknown",
    "evidence_quote": "короткая цитата из фрагмента (<=200 символов)",
    "time": {"start": "YYYY or YYYY-MM or YYYY-MM-DD", "end": "...", "granularity": "year|month|day"} | null
  }
]
Правила:
- НЕ выдумывай факты. Если не уверен — polarity="unknown" и confidence <= 0.5.
- Время: если в фрагменте явно указан период — заполни time.
  Если времени нет в тексте — оставь null (мы подставим год публикации из meta).
- evidence_quote: возьми дословный кусок из фрагмента (можно укоротить), не придумывай.
"""


def extract_temporal_triplets(
    domain: str,
    chunk_text: str,
    paper_year: Optional[int] = None,
    *,
    use_demos: Optional[bool] = None,
) -> List[TemporalTriplet]:
    """Extract temporal triplets from a chunk.

    If use_demos is True (or settings.demo_enabled), the function injects retrieval-few-shot
    examples from Qdrant demo store (task=temporal_triplets) to improve format and accuracy.
    """
    system = f"""Ты — помощник исследователя в области {domain}.
Твоя задача — извлечь из фрагмента текста научные утверждения в виде триплетов (S-P-O) и (если возможно) временные метки.
"""

    enabled = getattr(settings, "demo_enabled", True) if use_demos is None else use_demos
    demo_block = ""
    if enabled:
        k = int(getattr(settings, "demo_top_k_triplets", 3))
        demos = retrieve_demos(task="temporal_triplets", domain=domain, query=chunk_text, k=k)
        demo_block = render_demos_block(
            demos,
            max_total_chars=int(getattr(settings, "demo_max_chars_total", 3500)),
            title="Эталонные примеры извлечения триплетов",
        )

    user = f"""{demo_block}Фрагмент:
{chunk_text}

Извлеки 3-10 самых важных утверждений."""

    data = chat_json(system=system, user=user, schema_hint=TEMPORAL_TRIPLET_SCHEMA_HINT, temperature=0.0)
    adapter = TypeAdapter(List[TemporalTriplet])
    triplets = adapter.validate_python(data)

    # если time не найден, подставляем год публикации как «суррогат» времени
    fallback = default_time_from_paper_year(paper_year)
    if fallback:
        for t in triplets:
            if t.time is None:
                t.time = fallback
    return triplets
