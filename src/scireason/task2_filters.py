from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
from urllib.parse import unquote

import yaml  # type: ignore

STOPWORDS = {
    'the','and','for','with','from','that','this','these','those','into','onto','about','under','over','between','among',
    'study','paper','article','result','results','method','methods','approach','approaches','using','used','use','based',
    'model','models','data','analysis','task','graph','graphs','knowledge','temporal','validation','expert','experts',
    'как','что','это','для','при','под','над','между','после','перед','если','только','также','статья','статьи','метод','методы',
    'подход','подходы','данные','анализ','граф','графы','задача','задачи','эксперт','эксперты','knowledge','temporal',
}


def _norm_ws(text: str) -> str:
    return ' '.join((text or '').split())


def norm_title(text: str) -> str:
    s = _norm_ws(text).lower()
    s = re.sub(r'[^\w\s]+', ' ', s, flags=re.UNICODE)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def tokenize(text: str) -> List[str]:
    base = norm_title(text)
    tokens = [t for t in re.findall(r'[\w\-]{3,}', base, flags=re.UNICODE) if t not in STOPWORDS]
    return tokens


def _coerce_list(value: Any) -> List[str]:
    if value in (None, '', []):
        return []
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for item in value:
            txt = str(item or '').strip()
            if txt:
                out.append(txt)
        return out
    txt = str(value).strip()
    return [txt] if txt else []


def load_yaml_like(spec: Any) -> Dict[str, Any]:
    if spec is None:
        return {}
    if isinstance(spec, Mapping):
        return dict(spec)
    if isinstance(spec, (str, Path)):
        path = Path(spec)
        if path.exists():
            doc = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
            return doc if isinstance(doc, dict) else {}
        try:
            doc = yaml.safe_load(str(spec)) or {}
            return doc if isinstance(doc, dict) else {}
        except Exception:
            return {}
    return {}


def normalize_exclusion_spec(spec: Any) -> Dict[str, Any]:
    doc = load_yaml_like(spec)
    if not isinstance(doc, dict):
        return {
            'paper_ids': set(),
            'titles': set(),
            'source_refs': set(),
            'match_substrings': [],
            'url_substrings': [],
            'max_year': None,
        }

    nested = doc.get('task2_exclusions') or doc.get('exclusions') or doc.get('exclude_articles') or doc
    if isinstance(nested, list):
        nested = {'paper_ids': nested}
    if not isinstance(nested, dict):
        nested = {}

    paper_ids = set()
    titles = set()
    source_refs = set()
    match_substrings: List[str] = []
    url_substrings: List[str] = []

    for key in ('paper_ids', 'paper_id', 'papers', 'articles', 'ids'):
        for item in _coerce_list(nested.get(key)):
            paper_ids.add(item.lower())

    for key in ('titles', 'paper_titles', 'article_titles'):
        for item in _coerce_list(nested.get(key)):
            titles.add(norm_title(item))

    for key in ('source_refs', 'source_ref', 'sources', 'urls', 'pdf_urls'):
        for item in _coerce_list(nested.get(key)):
            source_refs.add(item.strip().lower())

    for key in ('match_substrings', 'contains', 'substrings', 'paper_id_contains', 'title_contains'):
        for item in _coerce_list(nested.get(key)):
            item = item.strip().lower()
            if item:
                match_substrings.append(item)

    for key in ('url_substrings', 'source_contains'):
        for item in _coerce_list(nested.get(key)):
            item = item.strip().lower()
            if item:
                url_substrings.append(item)

    year_cfg = nested.get('years') if isinstance(nested.get('years'), Mapping) else {}
    max_year = nested.get('max_year', year_cfg.get('max'))
    try:
        max_year = int(max_year) if max_year not in (None, '') else None
    except Exception:
        max_year = None

    return {
        'paper_ids': paper_ids,
        'titles': titles,
        'source_refs': source_refs,
        'match_substrings': sorted(set(match_substrings)),
        'url_substrings': sorted(set(url_substrings)),
        'max_year': max_year,
    }


def serialize_exclusion_spec(spec: Any) -> Dict[str, Any]:
    s = normalize_exclusion_spec(spec)
    return {
        'paper_ids': sorted(s['paper_ids']),
        'titles': sorted(s['titles']),
        'source_refs': sorted(s['source_refs']),
        'match_substrings': list(s['match_substrings']),
        'url_substrings': list(s['url_substrings']),
        'max_year': s['max_year'],
    }


def exclusion_spec_is_empty(spec: Any) -> bool:
    s = normalize_exclusion_spec(spec)
    return not any([
        s['paper_ids'], s['titles'], s['source_refs'], s['match_substrings'], s['url_substrings'], s['max_year'] is not None,
    ])


def _paper_year(entity: Any) -> Optional[int]:
    raw = None
    if isinstance(entity, Mapping):
        raw = entity.get('year')
    else:
        raw = getattr(entity, 'year', None)
    try:
        return int(raw) if raw not in (None, '') else None
    except Exception:
        return None


def _entity_values(entity: Any) -> List[str]:
    if isinstance(entity, Mapping):
        values = [
            entity.get('id'), entity.get('paper_id'), entity.get('title'), entity.get('url'), entity.get('pdf_url'),
            entity.get('source'), entity.get('papers_text'), entity.get('evidence_text'), entity.get('submission_id'),
        ]
        evidence = entity.get('evidence')
        if isinstance(evidence, Mapping):
            values.extend([evidence.get('paper_id'), evidence.get('source'), evidence.get('snippet_or_summary')])
        papers = entity.get('papers')
        if isinstance(papers, (list, tuple, set)):
            values.extend(list(papers))
        raw = entity.get('raw')
        if isinstance(raw, Mapping):
            values.extend([raw.get('url'), raw.get('pdf_url')])
    else:
        values = [
            getattr(entity, 'id', None), getattr(entity, 'paper_id', None), getattr(entity, 'title', None),
            getattr(entity, 'url', None), getattr(entity, 'pdf_url', None), getattr(entity, 'source', None),
        ]
        raw = getattr(entity, 'raw', None)
        if isinstance(raw, Mapping):
            values.extend([raw.get('url'), raw.get('pdf_url')])
    out: List[str] = []
    for value in values:
        text = str(value or '').strip()
        if text:
            out.append(unquote(text))
    return out


def entity_matches_exclusion(entity: Any, spec: Any) -> bool:
    rules = normalize_exclusion_spec(spec)
    if exclusion_spec_is_empty(rules):
        return False

    values = _entity_values(entity)
    lowered = [v.lower() for v in values]
    title_norms = {norm_title(v) for v in values}
    year = _paper_year(entity)

    if rules['max_year'] is not None and year is not None and year > int(rules['max_year']):
        return True

    if any(v in rules['paper_ids'] for v in lowered):
        return True
    if any(v in rules['source_refs'] for v in lowered):
        return True
    if any(t in rules['titles'] for t in title_norms if t):
        return True

    for probe in lowered:
        if any(sub in probe for sub in rules['match_substrings']):
            return True
        if probe.startswith(('http://', 'https://')) and any(sub in probe for sub in rules['url_substrings']):
            return True
    return False


def topic_profile_from_doc(doc: Mapping[str, Any]) -> Dict[str, Any]:
    pieces = [str(doc.get('topic') or ''), str(doc.get('domain') or '')]
    for step in doc.get('steps', []) or []:
        if not isinstance(step, Mapping):
            continue
        pieces.extend([
            str(step.get('claim') or ''),
            str(step.get('inference') or ''),
            str(step.get('next_question') or ''),
        ])
    for paper in doc.get('papers', []) or []:
        if isinstance(paper, Mapping):
            pieces.append(str(paper.get('title') or ''))
    text = ' '.join(pieces)
    tokens = tokenize(text)
    counts: Dict[str, int] = {}
    for tok in tokens:
        counts[tok] = counts.get(tok, 0) + 1
    return {
        'text': text,
        'tokens': set(tokens),
        'counts': counts,
        'cutoff_year': doc.get('cutoff_year'),
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ''):
            return default
        return float(value)
    except Exception:
        return default


def _coverage(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a))


def score_triplet_importance(row: Mapping[str, Any], profile: Mapping[str, Any], *, graph_metrics: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    text = ' '.join([
        str(row.get('subject') or ''),
        str(row.get('predicate') or ''),
        str(row.get('object') or ''),
        str(row.get('evidence_text') or ''),
        str(row.get('papers_text') or ''),
    ])
    row_tokens = set(tokenize(text))
    topic_tokens = set(profile.get('tokens') or set())

    overlap = _coverage(row_tokens, topic_tokens)
    topic_counts = profile.get('counts') or {}
    weighted_overlap = 0.0
    if row_tokens and topic_counts:
        common_weight = sum(float(topic_counts.get(tok, 0)) for tok in (row_tokens & topic_tokens))
        total_weight = sum(float(topic_counts.get(tok, 0)) for tok in topic_tokens) or 1.0
        weighted_overlap = common_weight / total_weight

    conf = _safe_float(row.get('mean_confidence'), _safe_float(row.get('score'), 0.0))
    conf = max(0.0, min(1.0, conf))

    papers_value = row.get('papers') or row.get('papers_text') or []
    if isinstance(papers_value, str):
        paper_support_n = len([x for x in re.split(r'[;,]', papers_value) if x.strip()])
    elif isinstance(papers_value, (list, tuple, set)):
        paper_support_n = len([x for x in papers_value if str(x or '').strip()])
    else:
        paper_support_n = 0
    paper_support = min(1.0, math.log1p(paper_support_n) / math.log(5.0)) if paper_support_n > 0 else 0.0

    evidence_len = len(str(row.get('evidence_text') or ''))
    evidence_richness = min(1.0, evidence_len / 220.0)

    time_source = str(row.get('time_source') or '').lower()
    time_specificity = 0.15 if time_source == 'triplet_extractor' else (0.08 if time_source in {'paper_year_fallback', 'metadata'} else 0.04)
    granularity = str(row.get('time_granularity') or '').lower()
    if granularity in {'day', 'month'}:
        time_specificity += 0.05

    centrality_bonus = 0.0
    if graph_metrics:
        subject = str(row.get('subject') or '')
        obj = str(row.get('object') or '')
        node_metrics = graph_metrics.get('node_metrics') or {}
        for node_id in (subject, obj):
            nm = node_metrics.get(node_id) or node_metrics.get(node_id.lower()) or {}
            centrality_bonus += min(0.05, max(0.0, _safe_float(nm.get('pagerank'), 0.0) * 2.5))
            centrality_bonus += min(0.03, max(0.0, _safe_float(nm.get('betweenness'), 0.0) * 1.5))
    centrality_bonus = min(0.1, centrality_bonus)

    score = (
        0.44 * overlap +
        0.18 * weighted_overlap +
        0.14 * conf +
        0.10 * paper_support +
        0.08 * evidence_richness +
        time_specificity +
        centrality_bonus
    )
    score = max(0.0, min(1.0, round(score, 4)))

    reasons: List[str] = []
    common_tokens = sorted(list(row_tokens & topic_tokens))[:8]
    if common_tokens:
        reasons.append('topic_overlap=' + ', '.join(common_tokens))
    if conf > 0:
        reasons.append(f'confidence={conf:.3f}')
    if paper_support_n > 0:
        reasons.append(f'papers={paper_support_n}')
    if time_source:
        reasons.append(f'time_source={time_source}')
    if centrality_bonus > 0:
        reasons.append(f'centrality_bonus={centrality_bonus:.3f}')

    return {
        'importance_score': score,
        'importance_model': 'topic_aware_relevance_v1',
        'importance_reasons': reasons,
        'topic_overlap_tokens': common_tokens,
    }
