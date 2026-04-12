from __future__ import annotations

import asyncio
import json
import re
import threading
from typing import List, Optional


from ..config import settings
from ..demos.render import render_demos_block
from ..llm import (
    _resolve_auto_provider,
    _resolve_llm_selection,
    chat_json,
    chat_json_async,
    temporary_llm_selection,
)
from .schemas import TemporalTriplet, normalize_granularity
from .extraction_contracts import (
    EXTRACTION_SCHEMA_HINT,
    validate_extraction_payload,
)
from .time_parse import default_time_from_paper_year


def _run_with_timeout(timeout_seconds: float, fn, /, *args, **kwargs):
    """Run a blocking extraction call in a daemon thread and fail fast on timeout.

    This protects notebook/CLI execution from hanging forever on flaky remote text
    LLM providers during temporal triplet extraction.
    """

    timeout_seconds = float(timeout_seconds or 0)
    if timeout_seconds <= 0:
        return fn(*args, **kwargs)

    result: dict[str, object] = {}

    def _target() -> None:
        try:
            result["value"] = fn(*args, **kwargs)
        except BaseException as exc:  # pragma: no cover - forwarded to caller
            result["error"] = exc

    thread = threading.Thread(target=_target, name=f"temporal-triplets-timeout:{getattr(fn, '__name__', 'call')}", daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        raise TimeoutError(f"Temporal triplet extraction exceeded {timeout_seconds:g}s")
    if "error" in result:
        raise result["error"]  # type: ignore[misc]
    return result.get("value")


_STOPWORDS = {
    'the','a','an','this','that','these','those','we','our','their','his','her','its','and','or','but','with','for','from','into','onto','than','then','using','use','used','show','shows','showed','shown','observe','observed','result','results','study','paper','method','methods','approach','approaches','data','dataset','datasets','experiment','experiments','analysis','model','models','in','on','at','to','of','by','as','is','are','was','were','be','being','been','can','may','might','could','should','would','will','during','under','across','between','among','via','through'
}

_ENTITY_JUNK = (
    'copyright',
    'all rights reserved',
    'supplementary',
    'figure',
    'table',
    'equation',
    'section',
    'appendix',
    'references',
)

_PREDICATE_ALIASES = {
    'associated with': 'associated_with',
    'associated_with': 'associated_with',
    'correlated with': 'associated_with',
    'linked to': 'associated_with',
    'related to': 'associated_with',
    'depends on': 'depends_on',
    'results in': 'results_in',
    'result in': 'results_in',
    'resulting in': 'results_in',
    'leads to': 'leads_to',
    'lead to': 'leads_to',
    'gives rise to': 'leads_to',
    'causes': 'causes',
    'caused': 'causes',
    'cause': 'causes',
    'induces': 'induces',
    'induce': 'induces',
    'induced': 'induces',
    'triggers': 'triggers',
    'trigger': 'triggers',
    'triggered': 'triggers',
    'drives': 'drives',
    'drive': 'drives',
    'driven': 'drives',
    'promotes': 'promotes',
    'promote': 'promotes',
    'promoted': 'promotes',
    'enhances': 'improves',
    'boosts': 'improves',
    'improves': 'improves',
    'improve': 'improves',
    'improved': 'improves',
    'increases': 'increases',
    'increase': 'increases',
    'increased': 'increases',
    'reduces': 'reduces',
    'reduce': 'reduces',
    'reduced': 'reduces',
    'decreases': 'reduces',
    'suppresses': 'suppresses',
    'inhibits': 'inhibits',
    'prevents': 'prevents',
    'mitigates': 'mitigates',
    'predicts': 'predicts',
    'predict': 'predicts',
    'predicted': 'predicts',
    'precedes': 'precedes',
    'before': 'precedes',
    'follows': 'follows',
    'after': 'follows',
    'occurs before': 'precedes',
    'occurs after': 'follows',
    'is followed by': 'precedes',
    'followed by': 'precedes',
}

_WEAK_PREDICATES = {
    'show', 'shows', 'state', 'states', 'describes', 'describe', 'mentions', 'section', 'displays', 'defined_as'
}

_STRONG_SIGNAL_PREDICATES = {
    'causes', 'induces', 'triggers', 'drives', 'promotes', 'leads_to', 'results_in', 'prevents',
    'inhibits', 'suppresses', 'mitigates', 'improves', 'increases', 'reduces', 'predicts', 'precedes', 'follows'
}

_RELATION_PATTERNS: list[tuple[str, str, str, float]] = [
    (r'(?P<subject>[A-Za-z][A-Za-z0-9_\-()/,%+ ]{2,90}?)\s+(?:directly\s+|strongly\s+|significantly\s+|consistently\s+)?(?P<verb>causes?|induces?|triggers?|drives?|promotes?)\s+(?P<object>[A-Za-z][A-Za-z0-9_\-()/,%+ ]{2,90}?)(?:[\.,;:]|$)', 'supports', 'causal', 0.9),
    (r'(?P<subject>[A-Za-z][A-Za-z0-9_\-()/,%+ ]{2,90}?)\s+(?:directly\s+|significantly\s+|consistently\s+)?(?P<verb>leads to|results in|gives rise to|depends on|predicts?)\s+(?P<object>[A-Za-z][A-Za-z0-9_\-()/,%+ ]{2,90}?)(?:[\.,;:]|$)', 'supports', 'causal', 0.87),
    (r'(?P<subject>[A-Za-z][A-Za-z0-9_\-()/,%+ ]{2,90}?)\s+(?:directly\s+|significantly\s+|strongly\s+)?(?P<verb>improves?|enhances?|boosts?|increases?)\s+(?P<object>[A-Za-z][A-Za-z0-9_\-()/,%+ ]{2,90}?)(?:[\.,;:]|$)', 'supports', 'directional', 0.82),
    (r'(?P<subject>[A-Za-z][A-Za-z0-9_\-()/,%+ ]{2,90}?)\s+(?:significantly\s+|consistently\s+)?(?P<verb>reduces?|decreases?|suppresses?|inhibits?|prevents?|mitigates?)\s+(?P<object>[A-Za-z][A-Za-z0-9_\-()/,%+ ]{2,90}?)(?:[\.,;:]|$)', 'supports', 'directional', 0.84),
    (r'(?P<subject>[A-Za-z][A-Za-z0-9_\-()/,%+ ]{2,90}?)\s+(?:occurs?\s+|happens?\s+|appears?\s+)?(?P<verb>before|precedes?)\s+(?P<object>[A-Za-z][A-Za-z0-9_\-()/,%+ ]{2,90}?)(?:[\.,;:]|$)', 'supports', 'temporal', 0.83),
    (r'(?P<subject>[A-Za-z][A-Za-z0-9_\-()/,%+ ]{2,90}?)\s+(?:occurs?\s+|happens?\s+|appears?\s+)?(?P<verb>after|follows?)\s+(?P<object>[A-Za-z][A-Za-z0-9_\-()/,%+ ]{2,90}?)(?:[\.,;:]|$)', 'supports', 'temporal', 0.83),
    (r'(?P<subject>[A-Za-z][A-Za-z0-9_\-()/,%+ ]{2,90}?)\s+(?P<verb>is followed by|followed by)\s+(?P<object>[A-Za-z][A-Za-z0-9_\-()/,%+ ]{2,90}?)(?:[\.,;:]|$)', 'supports', 'temporal', 0.8),
    (r'(?P<subject>[A-Za-z][A-Za-z0-9_\-()/,%+ ]{2,90}?)\s+(?:is|was|are|were)\s+(?P<verb>associated with|correlated with|linked to|related to)\s+(?P<object>[A-Za-z][A-Za-z0-9_\-()/,%+ ]{2,90}?)(?:[\.,;:]|$)', 'unknown', 'association', 0.56),
    (r'(?P<subject>[A-Za-z][A-Za-z0-9_\-()/,%+ ]{2,90}?)\s+(?P<verb>contradicts?|fails to improve|does not improve|does not increase|does not reduce|worsens?)\s+(?P<object>[A-Za-z][A-Za-z0-9_\-()/,%+ ]{2,90}?)(?:[\.,;:]|$)', 'contradicts', 'negated', 0.74),
]


def _clean_entity(text: str) -> str:
    s = ' '.join((text or '').replace('\n', ' ').split()).strip(' ,;:-')
    s = re.sub(r'^(?:the|a|an)\s+', '', s, flags=re.I)
    s = re.sub(r'\([^)]*\b(?:fig|figure|table|eq|equation)\.?\s*\d+[^)]*\)', '', s, flags=re.I)
    s = re.sub(r'\[[^\]]+\]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    raw_words = s.split()
    words = [w for w in raw_words if w.lower() not in _STOPWORDS]
    if not words:
        words = raw_words
    if len(words) > 10:
        words = words[:10]
    return ' '.join(words).strip(' ,;:-')


def _canonicalize_predicate(text) -> str:
    if isinstance(text, (bytes, bytearray)):
        s = text.decode('utf-8', errors='replace')
    else:
        s = str(text or '')
    s = ' '.join(s.strip().lower().split())
    if not s:
        return ''
    s = s.replace('-', ' ')
    return _PREDICATE_ALIASES.get(s, s.replace(' ', '_'))


def _predicate_strength(predicate: str) -> float:
    pred = _canonicalize_predicate(predicate)
    if pred in {'causes', 'induces', 'triggers', 'drives', 'promotes', 'leads_to', 'results_in'}:
        return 1.0
    if pred in {'prevents', 'inhibits', 'suppresses', 'mitigates', 'improves', 'increases', 'reduces', 'predicts'}:
        return 0.9
    if pred in {'precedes', 'follows'}:
        return 0.85
    if pred == 'associated_with':
        return 0.45
    if pred == 'relates_to':
        return 0.25
    return 0.3


def _is_informative_entity(text: str) -> bool:
    s = _clean_entity(text)
    if not s:
        return False
    low = s.lower()
    if len(low) < 3:
        return False
    if any(junk in low for junk in _ENTITY_JUNK):
        return False
    raw_tokens = re.findall(r'[A-Za-z][A-Za-z0-9_\-]*', low)
    tokens = [tok for tok in raw_tokens if tok not in _STOPWORDS]
    if not tokens:
        tokens = raw_tokens
    if not tokens:
        return False
    if len(tokens) > 12:
        return False
    return True


def _sentence_spans(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r'(?<=[\.\!?])\s+', text or '') if p and p.strip()]
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


def _quote_from_sentence(sentence: str) -> str:
    quote = ' '.join((sentence or '').split())
    return quote[:200]


def _make_triplet(*, subject: str, predicate: str, object: str, confidence: float, polarity: str, sentence: str, paper_year: Optional[int]) -> TemporalTriplet:
    predicate_norm = _canonicalize_predicate(predicate)
    time = _time_from_sentence(sentence, paper_year)
    return TemporalTriplet(
        subject=subject,
        predicate=predicate_norm,
        object=object,
        confidence=max(0.0, min(1.0, float(confidence))),
        polarity=polarity,
        evidence_quote=_quote_from_sentence(sentence),
        time=time,
        time_source='extracted' if time and str(time.get('start') or '') != str(paper_year or '') else 'paper_year_fallback',
    )


def _fallback_entity_candidates(chunk_text: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for sentence in _sentence_spans(chunk_text)[:5]:
        for phrase in re.findall(r'[A-Za-z][A-Za-z0-9_\-]{2,}(?:\s+[A-Za-z0-9_\-]{2,}){0,4}', sentence):
            ent = _clean_entity(phrase)
            key = ent.lower()
            if not _is_informative_entity(ent):
                continue
            if key in seen:
                continue
            seen.add(key)
            candidates.append(ent)
            if len(candidates) >= 8:
                return candidates
    return candidates


def _filter_triplets(triplets: List[TemporalTriplet]) -> List[TemporalTriplet]:
    normalized: list[TemporalTriplet] = []
    seen: set[tuple[str, str, str, str]] = set()
    for triplet in triplets:
        triplet.subject = _clean_entity(triplet.subject)
        triplet.object = _clean_entity(triplet.object)
        triplet.predicate = _canonicalize_predicate(triplet.predicate)
        if not _is_informative_entity(triplet.subject) or not _is_informative_entity(triplet.object):
            continue
        if triplet.subject.lower() == triplet.object.lower():
            continue
        if not triplet.predicate:
            continue
        if triplet.predicate in _WEAK_PREDICATES:
            continue
        key = (
            triplet.subject.lower(),
            triplet.predicate,
            triplet.object.lower(),
            str((triplet.time.start if triplet.time else '') or ''),
        )
        if key in seen:
            continue
        seen.add(key)
        normalized.append(triplet)

    strong = [t for t in normalized if t.predicate in _STRONG_SIGNAL_PREDICATES]
    weak = [t for t in normalized if t.predicate not in _STRONG_SIGNAL_PREDICATES]
    strong.sort(key=lambda t: (_predicate_strength(t.predicate), float(t.confidence), len(str(t.evidence_quote or ''))), reverse=True)
    weak.sort(key=lambda t: (_predicate_strength(t.predicate), float(t.confidence), len(str(t.evidence_quote or ''))), reverse=True)

    if strong:
        kept = strong + weak[: max(0, 10 - len(strong))]
    else:
        kept = weak
    return kept[:10]


def _rule_based_triplets(chunk_text: str, paper_year: Optional[int] = None) -> List[TemporalTriplet]:
    out: List[TemporalTriplet] = []
    for sentence in _sentence_spans(chunk_text):
        sent = ' '.join(sentence.split())
        if not sent:
            continue
        for pattern, polarity, _family, confidence in _RELATION_PATTERNS:
            for match in re.finditer(pattern, sent, flags=re.I):
                subject = _clean_entity(match.group('subject'))
                predicate = _canonicalize_predicate(match.group('verb'))
                obj = _clean_entity(match.group('object'))
                if not _is_informative_entity(subject) or not _is_informative_entity(obj):
                    continue
                if predicate == 'follows':
                    triplet = _make_triplet(subject=subject, predicate=predicate, object=obj, confidence=confidence, polarity=polarity, sentence=sent, paper_year=paper_year)
                elif predicate == 'precedes' and match.group('verb').strip().lower() in {'is followed by', 'followed by'}:
                    triplet = _make_triplet(subject=subject, predicate='precedes', object=obj, confidence=confidence, polarity=polarity, sentence=sent, paper_year=paper_year)
                else:
                    triplet = _make_triplet(subject=subject, predicate=predicate, object=obj, confidence=confidence, polarity=polarity, sentence=sent, paper_year=paper_year)
                out.append(triplet)

    filtered = _filter_triplets(out)
    if filtered:
        return filtered

    candidates = _fallback_entity_candidates(chunk_text)
    out = []
    for sentence in _sentence_spans(chunk_text)[:3]:
        sent = ' '.join(sentence.split())
        if not sent:
            continue
        sentence_entities = [ent for ent in candidates if ent.lower() in sent.lower()]
        if len(sentence_entities) < 2:
            continue
        for i in range(len(sentence_entities)):
            for j in range(i + 1, len(sentence_entities)):
                s = sentence_entities[i]
                o = sentence_entities[j]
                if s.lower() == o.lower():
                    continue
                cue_match = re.search(r'\b(improv(?:e|es|ed)|increase(?:s|d)?|reduce(?:s|d)?|decrease(?:s|d)?|lead(?:s)? to|result(?:s)? in|cause(?:s|d)?|prevent(?:s|ed)?|before|after|follow(?:s|ed)?|predict(?:s|ed)?)\b', sent, flags=re.I)
                predicate = _canonicalize_predicate(cue_match.group(1)) if cue_match else 'associated_with'
                confidence = 0.62 if cue_match else 0.46
                polarity = 'supports' if cue_match else 'unknown'
                out.append(_make_triplet(subject=s, predicate=predicate, object=o, confidence=confidence, polarity=polarity, sentence=sent, paper_year=paper_year))
                if len(out) >= 8:
                    return _filter_triplets(out)
    return _filter_triplets(out)


TEMPORAL_TRIPLET_SCHEMA_HINT = EXTRACTION_SCHEMA_HINT



def _build_temporal_triplet_prompt_parts(domain: str, chunk_text: str, use_demos: Optional[bool]) -> tuple[str, str]:
    system = f"""Ты — помощник исследователя в области {domain}.
Твоя задача — извлечь из фрагмента текста научные утверждения в виде триплетов (S-P-O) и сохранить временные метки с максимально возможной точностью.
Возвращай только связи, которые действительно выражают направленное влияние, изменение, причинность, зависимость, прогноз или временной порядок. Не подменяй это co-occurrence или общим тематическим соседством.
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

Извлеки 3-10 самых важных утверждений. Для каждого триплета используй компактные сущности и канонический предикат в snake_case. Верни объект с ключом triplets по строгой JSON schema."""
    return system, user


def _validate_triplet_payload(data) -> List[TemporalTriplet]:
    rows = validate_extraction_payload(data)
    normalized: list[dict] = []
    for item in rows:
        row = item.model_dump(mode="json")
        row["predicate"] = _canonicalize_predicate(row.get("predicate") or "")
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
    return [TemporalTriplet.model_validate(row) for row in normalized]



def _finalize_triplets(triplets: List[TemporalTriplet], paper_year: Optional[int]) -> List[TemporalTriplet]:
    fallback = default_time_from_paper_year(paper_year)
    kept = _filter_triplets([
        t for t in triplets
        if t.subject.strip() and t.predicate.strip() and t.object.strip()
    ])
    if fallback:
        for t in kept:
            if t.time is None:
                t.time = fallback
                t.time_source = 'paper_year_fallback'
            elif not getattr(t, 'time_source', None):
                t.time_source = 'extracted'
    return kept


def extract_temporal_triplets_localized_fallback(
    chunk_text: str,
    paper_year: Optional[int] = None,
) -> List[TemporalTriplet]:
    return _finalize_triplets(_rule_based_triplets(chunk_text=chunk_text, paper_year=paper_year), paper_year)


async def extract_temporal_triplets_async(
    domain: str,
    chunk_text: str,
    paper_year: Optional[int] = None,
    *,
    use_demos: Optional[bool] = None,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> List[TemporalTriplet]:
    """Async extractor used for high-throughput g4f triplet extraction."""

    system, user = _build_temporal_triplet_prompt_parts(domain=domain, chunk_text=chunk_text, use_demos=use_demos)
    provider, model = _resolve_llm_selection(llm_provider=llm_provider, llm_model=llm_model)

    if provider == 'mock':
        return extract_temporal_triplets_localized_fallback(chunk_text=chunk_text, paper_year=paper_year)

    if provider != 'g4f' or not bool(getattr(settings, 'g4f_async_enabled', True)):
        return await asyncio.to_thread(
            extract_temporal_triplets,
            domain,
            chunk_text,
            paper_year,
            use_demos=use_demos,
            llm_provider=provider,
            llm_model=model,
        )

    async def _call() -> List[TemporalTriplet]:
        try:
            data = await chat_json_async(
                system=system,
                user=user,
                schema_hint=TEMPORAL_TRIPLET_SCHEMA_HINT,
                temperature=0.0,
                llm_provider=provider,
                llm_model=model,
            )
            triplets = _finalize_triplets(_validate_triplet_payload(data), paper_year)
            if triplets:
                return triplets
        except TimeoutError:
            raise
        except Exception:
            pass
        return extract_temporal_triplets_localized_fallback(chunk_text=chunk_text, paper_year=paper_year)

    if semaphore is None:
        return await _call()
    async with semaphore:
        return await _call()


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
    system, user = _build_temporal_triplet_prompt_parts(domain=domain, chunk_text=chunk_text, use_demos=use_demos)

    with temporary_llm_selection(llm_provider=llm_provider, llm_model=llm_model):
        provider = (settings.llm_provider or '').lower().strip() or 'auto'
        if provider == 'auto':
            provider = _resolve_auto_provider()

        if provider == 'mock':
            triplets = extract_temporal_triplets_localized_fallback(chunk_text=chunk_text, paper_year=paper_year)
        else:
            timeout_seconds = float(getattr(settings, "llm_request_timeout_seconds", 25) or 25)
            try:
                data = _run_with_timeout(
                    timeout_seconds,
                    chat_json,
                    system=system,
                    user=user,
                    schema_hint=TEMPORAL_TRIPLET_SCHEMA_HINT,
                    temperature=0.0,
                )
                triplets = _finalize_triplets(_validate_triplet_payload(data), paper_year)
                if not triplets:
                    triplets = extract_temporal_triplets_localized_fallback(chunk_text=chunk_text, paper_year=paper_year)
            except TimeoutError:
                raise
            except Exception:
                triplets = extract_temporal_triplets_localized_fallback(chunk_text=chunk_text, paper_year=paper_year)

    return _finalize_triplets(triplets, paper_year)
