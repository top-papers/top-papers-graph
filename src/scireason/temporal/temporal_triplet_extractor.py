from __future__ import annotations

from typing import List, Optional
import re
from pydantic import TypeAdapter

from ..config import settings
from ..llm import chat_json, _resolve_auto_provider
from ..demos.render import render_demos_block
from .schemas import TemporalTriplet
from .time_parse import default_time_from_paper_year




_STOPWORDS = {
    'the','a','an','this','that','these','those','we','our','their','his','her','its','and','or','but','with','for','from','into','onto','than','then','using','use','used','show','shows','showed','shown','observe','observed','result','results','study','paper','method','methods','approach','approaches','data','dataset','datasets','experiment','experiments','analysis','model','models','in','on','at','to','of','by','as','is','are','was','were','be','being','been','can','may','might','could','should','would','will','during','under','across','between','among','via','through'
}


def _clean_entity(text: str) -> str:
    s = ' '.join((text or '').replace('\n', ' ').split()).strip(' ,;:-')
    s = re.sub(r'^(?:the|a|an)\s+', '', s, flags=re.I)
    s = re.sub(r'\s+', ' ', s).strip()
    words = [w for w in s.split() if w.lower() not in _STOPWORDS]
    if len(words) > 8:
        words = words[:8]
    return ' '.join(words).strip()


def _sentence_spans(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r'(?<=[\.!?])\s+', text or '') if p and p.strip()]
    return parts if parts else [text]


def _time_from_sentence(sentence: str, paper_year: Optional[int]):
    m = re.search(r'(19\d{2}|20\d{2}|2100)', sentence)
    year = m.group(1) if m else (str(paper_year) if paper_year else None)
    if not year:
        return None
    return {'start': year, 'end': year, 'granularity': 'year'}


def _rule_based_triplets(chunk_text: str, paper_year: Optional[int] = None) -> List[TemporalTriplet]:
    patterns = [
        (r'(?P<subject>[A-Za-z][A-Za-z0-9_\- /]{2,80}?)\s+(?:significantly\s+|strongly\s+|consistently\s+)?(?P<verb>improves?|enhances?|boosts?|increases?)\s+(?P<object>[A-Za-z][A-Za-z0-9_\- /]{2,80}?)(?:[\.,;:]|$)', 'supports'),
        (r'(?P<subject>[A-Za-z][A-Za-z0-9_\- /]{2,80}?)\s+(?P<verb>reduces?|decreases?|suppresses?|mitigates?)\s+(?P<object>[A-Za-z][A-Za-z0-9_\- /]{2,80}?)(?:[\.,;:]|$)', 'supports'),
        (r'(?P<subject>[A-Za-z][A-Za-z0-9_\- /]{2,80}?)\s+(?:is|was|are|were)\s+(?P<verb>associated with|correlated with|linked to|related to)\s+(?P<object>[A-Za-z][A-Za-z0-9_\- /]{2,80}?)(?:[\.,;:]|$)', 'unknown'),
        (r'(?P<subject>[A-Za-z][A-Za-z0-9_\- /]{2,80}?)\s+(?P<verb>causes?|induces?|drives?|promotes?)\s+(?P<object>[A-Za-z][A-Za-z0-9_\- /]{2,80}?)(?:[\.,;:]|$)', 'supports'),
        (r'(?P<subject>[A-Za-z][A-Za-z0-9_\- /]{2,80}?)\s+(?P<verb>contradicts?|fails to improve|does not improve|does not increase|does not reduce|worsens?)\s+(?P<object>[A-Za-z][A-Za-z0-9_\- /]{2,80}?)(?:[\.,;:]|$)', 'contradicts'),
    ]
    out: List[TemporalTriplet] = []
    seen: set[tuple[str, str, str]] = set()
    for sentence in _sentence_spans(chunk_text):
        sent = ' '.join(sentence.split())
        for pattern, polarity in patterns:
            for m in re.finditer(pattern, sent, flags=re.I):
                subj = _clean_entity(m.group('subject'))
                pred = _clean_entity(m.group('verb')).lower().replace(' ', '_')
                obj = _clean_entity(m.group('object'))
                if not subj or not obj or subj.lower() == obj.lower():
                    continue
                if len(subj) < 3 or len(obj) < 3:
                    continue
                key = (subj.lower(), pred, obj.lower())
                if key in seen:
                    continue
                seen.add(key)
                time = _time_from_sentence(sent, paper_year)
                out.append(TemporalTriplet(subject=subj, predicate=pred, object=obj, confidence=0.72 if polarity == 'supports' else 0.58, polarity=polarity, evidence_quote=sent[:200], time=time))
                if len(out) >= 10:
                    return out
    if out:
        return out

    # Best-effort co-occurrence fallback from capitalized / technical phrases in the first sentences.
    candidates: list[str] = []
    for sentence in _sentence_spans(chunk_text)[:4]:
        for phrase in re.findall(r'[A-Za-z][A-Za-z0-9_\-]{2,}(?:\s+[A-Za-z0-9_\-]{2,}){0,3}', sentence):
            ent = _clean_entity(phrase)
            if not ent or ent.lower() in _STOPWORDS:
                continue
            if ent.lower() not in {c.lower() for c in candidates}:
                candidates.append(ent)
            if len(candidates) >= 6:
                break
        if len(candidates) >= 6:
            break
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            s = candidates[i]
            o = candidates[j]
            if s.lower() == o.lower():
                continue
            time = _time_from_sentence(chunk_text, paper_year)
            out.append(TemporalTriplet(subject=s, predicate='relates_to', object=o, confidence=0.41, polarity='unknown', evidence_quote=' '.join(_sentence_spans(chunk_text)[:1])[:200], time=time))
            if len(out) >= 5:
                return out
    return out


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
        try:
            # Lazy import: demos require Qdrant client/service, but the extractor itself shouldn't.
            from ..demos.retriever import retrieve_demos  # type: ignore

            demos = retrieve_demos(task="temporal_triplets", domain=domain, query=chunk_text, k=k)
            demo_block = render_demos_block(
                demos,
                max_total_chars=int(getattr(settings, "demo_max_chars_total", 3500)),
                title="Эталонные примеры извлечения триплетов",
            )
        except Exception:
            demo_block = ""

    provider = (settings.llm_provider or '').lower().strip() or 'auto'
    if provider == 'auto':
        provider = _resolve_auto_provider()

    user = f"""{demo_block}Фрагмент:
{chunk_text}

Извлеки 3-10 самых важных утверждений."""

    if provider == 'mock':
        triplets = _rule_based_triplets(chunk_text=chunk_text, paper_year=paper_year)
    else:
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
