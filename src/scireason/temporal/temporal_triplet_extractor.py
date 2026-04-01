from __future__ import annotations

from typing import List, Optional
import re
from pydantic import TypeAdapter

from ..config import settings
from ..llm import chat_json, _resolve_auto_provider, temporary_llm_selection
from ..demos.render import render_demos_block
from .schemas import TemporalTriplet, normalize_granularity
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
    m = re.search(r'(19\d{2}|20\d{2}|2100)(?:-(\d{2})(?:-(\d{2}))?)?', sentence)
    if m:
        year = m.group(1)
        month = m.group(2)
        day = m.group(3)
        if day:
            value = f'{year}-{month}-{day}'
            granularity = 'day'
        elif month:
            value = f'{year}-{month}'
            granularity = 'month'
        else:
            value = year
            granularity = 'year'
        return {'start': value, 'end': value, 'granularity': granularity}
    if paper_year:
        return {'start': str(paper_year), 'end': str(paper_year), 'granularity': 'year'}
    return None


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
                out.append(
                    TemporalTriplet(
                        subject=subj,
                        predicate=pred,
                        object=obj,
                        confidence=0.72 if polarity == 'supports' else 0.58,
                        polarity=polarity,
                        evidence_quote=sent[:200],
                        time=time,
                        time_source='extracted' if time and str(time.get('start') or '') != str(paper_year or '') else 'paper_year_fallback',
                    )
                )
                if len(out) >= 10:
                    return out
    if out:
        return out

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
            out.append(
                TemporalTriplet(
                    subject=s,
                    predicate='relates_to',
                    object=o,
                    confidence=0.41,
                    polarity='unknown',
                    evidence_quote=' '.join(_sentence_spans(chunk_text)[:1])[:200],
                    time=time,
                    time_source='extracted' if time and str(time.get('start') or '') != str(paper_year or '') else 'paper_year_fallback',
                )
            )
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
    "time": {"start": "YYYY or YYYY-MM or YYYY-MM-DD", "end": "...", "granularity": "year|month|day|interval"} | null
  }
]
Правила:
- НЕ выдумывай факты. Если не уверен — polarity="unknown" и confidence <= 0.5.
- Если в тексте есть точная дата/месяц/период — обязательно сохрани её с максимально доступной гранулярностью.
- Для диапазонов и периодов используй granularity="interval".
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
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> List[TemporalTriplet]:
    """Extract temporal triplets from a chunk.

    If use_demos is True (or settings.demo_enabled), the function injects retrieval-few-shot
    examples from Qdrant demo store (task=temporal_triplets) to improve format and accuracy.

    `llm_provider` / `llm_model` are explicit overrides for this extraction call. This makes
    notebook/CLI overrides deterministic instead of relying on ambient repo defaults.
    """
    system = f"""Ты — помощник исследователя в области {domain}.
Твоя задача — извлечь из фрагмента текста научные утверждения в виде триплетов (S-P-O) и сохранить временные метки с максимально возможной точностью.
"""

    enabled = getattr(settings, "demo_enabled", True) if use_demos is None else use_demos
    demo_block = ""
    if enabled:
        k = int(getattr(settings, "demo_top_k_triplets", 3))
        try:
            from ..demos.retriever import retrieve_demos  # type: ignore

            demos = retrieve_demos(task="temporal_triplets", domain=domain, query=chunk_text, k=k)
            demo_block = render_demos_block(
                demos,
                max_total_chars=int(getattr(settings, "demo_max_chars_total", 3500)),
                title="Эталонные примеры извлечения триплетов",
            )
        except Exception:
            demo_block = ""

    user = f"""{demo_block}Фрагмент:
{chunk_text}

Извлеки 3-10 самых важных утверждений."""

    with temporary_llm_selection(llm_provider=llm_provider, llm_model=llm_model):
        provider = (settings.llm_provider or '').lower().strip() or 'auto'
        if provider == 'auto':
            provider = _resolve_auto_provider()

        if provider == 'mock':
            triplets = _rule_based_triplets(chunk_text=chunk_text, paper_year=paper_year)
        else:
            data = chat_json(system=system, user=user, schema_hint=TEMPORAL_TRIPLET_SCHEMA_HINT, temperature=0.0)
            if isinstance(data, list):
                normalized = []
                for item in data:
                    if not isinstance(item, dict):
                        normalized.append(item)
                        continue
                    row = dict(item)
                    time_obj = row.get("time")
                    if isinstance(time_obj, dict):
                        time_payload = dict(time_obj)
                        time_payload["granularity"] = normalize_granularity(
                            time_payload.get("granularity"),
                            start=time_payload.get("start"),
                            end=time_payload.get("end"),
                            default="year",
                        )
                        row["time"] = time_payload
                    normalized.append(row)
                data = normalized
            adapter = TypeAdapter(List[TemporalTriplet])
            triplets = adapter.validate_python(data)

    fallback = default_time_from_paper_year(paper_year)
    if fallback:
        for t in triplets:
            if t.time is None:
                t.time = fallback
                t.time_source = 'paper_year_fallback'
            elif not getattr(t, 'time_source', None):
                t.time_source = 'extracted'
    return triplets
