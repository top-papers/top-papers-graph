from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple
from urllib.parse import unquote, urlparse

import yaml  # type: ignore

STOPWORDS = {
    'the','and','for','with','from','that','this','these','those','into','onto','about','under','over','between','among',
    'study','paper','article','result','results','method','methods','approach','approaches','using','used','use','based',
    'model','models','data','analysis','task','graph','graphs','knowledge','temporal','validation','expert','experts',
    'как','что','это','для','при','под','над','между','после','перед','если','только','также','статья','статьи','метод','методы',
    'подход','подходы','данные','анализ','граф','графы','задача','задачи','эксперт','эксперты','knowledge','temporal',
}

DOI_RE = re.compile(r'10\.\d{4,9}/[-._;()/:A-Z0-9]+', re.IGNORECASE)
ID_PREFIXES = {'doi', 'pmid', 'pmcid', 'arxiv', 'openalex'}


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



def _looks_like_url(text: str) -> bool:
    raw = str(text or '').strip()
    if not raw:
        return False
    try:
        parsed = urlparse(raw)
    except Exception:
        return False
    return bool(parsed.scheme and parsed.netloc)



def _extract_doi(text: Any) -> Optional[str]:
    raw = unquote(str(text or '').strip())
    if not raw:
        return None
    match = DOI_RE.search(raw)
    if not match:
        return None
    return match.group(0).rstrip(').,;]').lower()



def _normalize_url(text: Any, *, strip_pdf_suffix: bool = False) -> str:
    raw = unquote(str(text or '').strip())
    if not raw:
        return ''
    try:
        parsed = urlparse(raw)
    except Exception:
        return ''
    if not (parsed.scheme and parsed.netloc):
        return ''
    path = (parsed.path or '').rstrip('/')
    if strip_pdf_suffix and path.lower().endswith('.pdf'):
        path = path[:-4]
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{path}"



def _identifier_variants(value: Any) -> Set[str]:
    raw = unquote(str(value or '').strip())
    if not raw:
        return set()
    lowered = raw.lower()
    variants: Set[str] = {lowered}

    doi = _extract_doi(raw)
    if doi:
        variants.update({doi, f'doi:{doi}', f'https://doi.org/{doi}'})

    if _looks_like_url(raw):
        normalized = _normalize_url(raw, strip_pdf_suffix=False)
        normalized_without_pdf = _normalize_url(raw, strip_pdf_suffix=True)
        for candidate in (normalized, normalized_without_pdf):
            if candidate:
                variants.add(candidate)
        tail = normalized_without_pdf.rsplit('/', 1)[-1] if normalized_without_pdf else ''
        if tail:
            variants.add(tail)

    if ':' in lowered:
        prefix, remainder = lowered.split(':', 1)
        if prefix in ID_PREFIXES and remainder:
            variants.add(remainder)
            if prefix == 'doi':
                variants.add(f'https://doi.org/{remainder}')

    return {item for item in variants if item}



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

    paper_ids: Set[str] = set()
    titles: Set[str] = set()
    source_refs: Set[str] = set()
    match_substrings: List[str] = []
    url_substrings: List[str] = []

    for key in ('paper_ids', 'paper_id', 'papers', 'articles', 'ids', 'identifier', 'identifiers', 'doi', 'dois'):
        for item in _coerce_list(nested.get(key)):
            paper_ids.update(_identifier_variants(item))

    for key in ('titles', 'paper_titles', 'article_titles'):
        for item in _coerce_list(nested.get(key)):
            normalized = norm_title(item)
            if normalized:
                titles.add(normalized)

    for key in ('source_refs', 'source_ref', 'sources', 'urls', 'pdf_urls', 'links'):
        for item in _coerce_list(nested.get(key)):
            source_refs.update(_identifier_variants(item))

    for key in ('match_substrings', 'contains', 'substrings', 'paper_id_contains', 'title_contains'):
        for item in _coerce_list(nested.get(key)):
            item = item.strip().lower()
            if item:
                match_substrings.append(item)

    for key in ('url_substrings', 'source_contains', 'link_contains'):
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



def _collect_entity_fields(entity: Any) -> Dict[str, List[str]]:
    buckets: Dict[str, List[str]] = {
        'ids': [],
        'titles': [],
        'source_refs': [],
        'general': [],
    }
    seen: Set[Tuple[str, str]] = set()

    def _push(bucket: str, value: Any) -> None:
        if value is None:
            return
        text = unquote(str(value).strip())
        if not text:
            return
        key = (bucket, text)
        if key in seen:
            return
        seen.add(key)
        buckets[bucket].append(text)

    def _visit(node: Any, *, hint: str = 'general') -> None:
        if node is None:
            return
        if isinstance(node, Mapping):
            for key, value in node.items():
                lower_key = str(key or '').strip().lower()
                if value is None:
                    continue
                if isinstance(value, (Mapping, list, tuple, set)):
                    if lower_key in {'raw', 'metadata', 'evidence', 'papers', 'paper', 'sources'}:
                        _visit(value, hint=hint)
                    elif lower_key in {'paper_ids', 'paper_id', 'id', 'doi', 'pmid', 'pmcid', 'arxiv', 'openalex', 'ids', 'identifiers'}:
                        _visit(value, hint='ids')
                    elif lower_key in {'paper_titles', 'paper_title', 'title', 'titles', 'name'}:
                        _visit(value, hint='titles')
                    elif lower_key in {'paper_source_refs', 'paper_source_ref', 'source_ref', 'source_refs', 'url', 'urls', 'pdf_url', 'pdf_urls', 'landing_page', 'source'}:
                        _visit(value, hint='source_refs')
                    else:
                        _visit(value, hint=hint)
                    continue

                if lower_key in {'paper_ids', 'paper_id', 'id', 'doi', 'pmid', 'pmcid', 'arxiv', 'openalex', 'ids', 'identifiers'}:
                    _push('ids', value)
                elif lower_key in {'paper_titles', 'paper_title', 'title', 'titles', 'name'}:
                    _push('titles', value)
                elif lower_key in {'paper_source_refs', 'paper_source_ref', 'source_ref', 'source_refs', 'url', 'urls', 'pdf_url', 'pdf_urls', 'landing_page', 'source'}:
                    _push('source_refs', value)
                else:
                    _push('general', value)
            return

        if isinstance(node, (list, tuple, set)):
            for item in node:
                _visit(item, hint=hint)
            return

        _push(hint, node)

    if isinstance(entity, Mapping):
        _visit(entity)
    else:
        _visit({
            'id': getattr(entity, 'id', None),
            'paper_id': getattr(entity, 'paper_id', None),
            'title': getattr(entity, 'title', None),
            'url': getattr(entity, 'url', None),
            'pdf_url': getattr(entity, 'pdf_url', None),
            'source': getattr(entity, 'source', None),
            'raw': getattr(entity, 'raw', None),
        })
    return buckets



def entity_matches_exclusion(entity: Any, spec: Any) -> bool:
    rules = normalize_exclusion_spec(spec)
    if exclusion_spec_is_empty(rules):
        return False

    fields = _collect_entity_fields(entity)
    id_variants: Set[str] = set()
    for value in fields['ids'] + fields['source_refs']:
        id_variants.update(_identifier_variants(value))
    source_variants: Set[str] = set()
    for value in fields['source_refs'] + fields['ids']:
        source_variants.update(_identifier_variants(value))

    title_norms = {norm_title(v) for v in fields['titles'] if norm_title(v)}
    haystack_parts = fields['ids'] + fields['titles'] + fields['source_refs'] + fields['general']
    haystack = ' '.join(haystack_parts).lower()
    year = _paper_year(entity)

    if rules['max_year'] is not None and year is not None and year > int(rules['max_year']):
        return True

    if rules['paper_ids'] and (rules['paper_ids'] & id_variants):
        return True
    if rules['source_refs'] and (rules['source_refs'] & source_variants):
        return True
    if any(title in rules['titles'] for title in title_norms):
        return True

    if any(token in haystack for token in rules['paper_ids']):
        return True
    if any(token in haystack for token in rules['titles']):
        return True
    if any(token in haystack for token in rules['source_refs']):
        return True
    if any(sub in haystack for sub in rules['match_substrings']):
        return True
    if any(sub in haystack for sub in rules['url_substrings']):
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



def _normalized_edge_key(subject: Any, predicate: Any, object_: Any) -> str:
    return ' | '.join([
        _norm_ws(str(subject or '')).lower(),
        _norm_ws(str(predicate or '')).lower(),
        _norm_ws(str(object_ or '')).lower(),
    ])



def _max_norm(value: Any, max_value: Any) -> float:
    raw = max(0.0, _safe_float(value, 0.0))
    ceiling = max(0.0, _safe_float(max_value, 0.0))
    if ceiling <= 0.0:
        return 0.0
    return max(0.0, min(1.0, raw / ceiling))




def is_cooccurrence_triplet(entity: Any) -> bool:
    fields = _collect_entity_fields(entity)
    text_parts: list[str] = []
    for bucket in ('ids', 'titles', 'source_refs', 'general'):
        text_parts.extend(fields.get(bucket) or [])
    haystack = ' '.join(str(part or '').lower() for part in text_parts)
    return any(token in haystack for token in ('cooccurs_with', 'cooccur', 'co-occur', ':cooccurrence', 'source_kind": "cooccurrence', 'source_kind": "text:cooccurrence'))


def is_likely_causal_triplet(entity: Any) -> bool:
    fields = _collect_entity_fields(entity)
    text_parts: list[str] = []
    for bucket in ('ids', 'titles', 'source_refs', 'general'):
        text_parts.extend(fields.get(bucket) or [])
    haystack = ' '.join(str(part or '').lower() for part in text_parts)
    return any(token in haystack for token in (
        'cause', 'causes', 'causal', 'lead to', 'leads to', 'drives', 'drive', 'induces', 'induce',
        'promotes', 'promote', 'inhibits', 'inhibit', 'mediates', 'mediate', 'results in', 'trigger', 'triggered by',
    ))


def cooccurrence_strength_score(entity: Any) -> float:
    candidates = []
    if isinstance(entity, Mapping):
        candidates = [
            entity.get('cooccurrence_strength'),
            entity.get('importance_score'),
            entity.get('graph_support_strength'),
            entity.get('mean_confidence'),
            entity.get('score'),
        ]
    else:
        candidates = [
            getattr(entity, 'cooccurrence_strength', None),
            getattr(entity, 'importance_score', None),
            getattr(entity, 'graph_support_strength', None),
            getattr(entity, 'mean_confidence', None),
            getattr(entity, 'score', None),
        ]
    for value in candidates:
        score = _safe_float(value, -1.0)
        if 0.0 <= score <= 1.0:
            return round(score, 6)
    return 0.0


def should_filter_cooccurrence(entity: Any, *, mode: str = 'all', weak_threshold: float = 0.45) -> bool:
    normalized_mode = str(mode or 'all').strip().lower()
    if normalized_mode == 'all':
        return False
    is_coocc = is_cooccurrence_triplet(entity)
    if not is_coocc:
        return False
    if normalized_mode in {'exclude_all', 'drop_all', 'only_non_cooccurrence'}:
        return True
    if normalized_mode not in {'hide_weak', 'weak_only'}:
        return False
    if is_likely_causal_triplet(entity):
        return False
    return cooccurrence_strength_score(entity) <= max(0.0, min(1.0, _safe_float(weak_threshold, 0.45)))

def score_triplet_importance(row: Mapping[str, Any], profile: Mapping[str, Any], *, graph_metrics: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    del profile  # compatibility: scoring is intentionally topology-only now.

    edge_key = _normalized_edge_key(row.get('subject'), row.get('predicate'), row.get('object'))
    edge_metrics = ((graph_metrics or {}).get('edge_metrics') or {}).get(edge_key) or {}
    maxima = (graph_metrics or {}).get('edge_metric_max') or {}

    edge_betweenness = _max_norm(edge_metrics.get('edge_betweenness'), maxima.get('edge_betweenness'))
    endpoint_pagerank = _max_norm(edge_metrics.get('pagerank_mean'), maxima.get('pagerank_mean'))
    endpoint_betweenness = _max_norm(edge_metrics.get('node_betweenness_mean'), maxima.get('node_betweenness_mean'))
    core_cohesion = _max_norm(edge_metrics.get('core_mean'), maxima.get('core_mean'))
    directional_flow = _max_norm(edge_metrics.get('directional_flow'), maxima.get('directional_flow'))

    support_strength = (
        _max_norm(edge_metrics.get('total_count'), maxima.get('total_count'))
        + _max_norm(edge_metrics.get('papers_count'), maxima.get('papers_count'))
        + _max_norm(edge_metrics.get('evidence_count'), maxima.get('evidence_count'))
    ) / 3.0

    temporal_persistence = (
        _max_norm(edge_metrics.get('active_years_count'), maxima.get('active_years_count'))
        + _max_norm(edge_metrics.get('temporal_span_years'), maxima.get('temporal_span_years'))
        + _max_norm(edge_metrics.get('temporal_density'), maxima.get('temporal_density'))
    ) / 3.0

    cross_community = 1.0 if bool(edge_metrics.get('is_cross_community')) else 0.0

    score = (
        0.28 * edge_betweenness
        + 0.18 * endpoint_pagerank
        + 0.14 * endpoint_betweenness
        + 0.12 * core_cohesion
        + 0.10 * directional_flow
        + 0.10 * support_strength
        + 0.05 * temporal_persistence
        + 0.03 * cross_community
    )
    score = max(0.0, min(1.0, round(score, 4)))

    reasons: List[str] = []
    if edge_metrics:
        reasons.append(f"edge_betweenness={edge_metrics.get('edge_betweenness', 0):.4f}")
        reasons.append(f"pagerank_mean={edge_metrics.get('pagerank_mean', 0):.4f}")
        reasons.append(f"node_betweenness_mean={edge_metrics.get('node_betweenness_mean', 0):.4f}")
        reasons.append(f"core_mean={edge_metrics.get('core_mean', 0):.4f}")
        reasons.append(f"directional_flow={edge_metrics.get('directional_flow', 0):.4f}")
        if edge_metrics.get('total_count') not in (None, ''):
            reasons.append(f"total_count={edge_metrics.get('total_count')}")
        if edge_metrics.get('papers_count') not in (None, ''):
            reasons.append(f"papers={edge_metrics.get('papers_count')}")
        if edge_metrics.get('active_years_count') not in (None, ''):
            reasons.append(f"active_years={edge_metrics.get('active_years_count')}")
        if edge_metrics.get('is_cross_community'):
            reasons.append('cross_community_bridge=1')
    else:
        reasons.append('topology_metrics_missing')

    return {
        'importance_score': score,
        'importance_model': 'temporal_graph_topology_v2',
        'importance_reasons': reasons,
        'topic_overlap_tokens': [],
        'graph_edge_betweenness': round(edge_betweenness, 6),
        'graph_pagerank_mean': round(endpoint_pagerank, 6),
        'graph_node_betweenness_mean': round(endpoint_betweenness, 6),
        'graph_core_mean': round(core_cohesion, 6),
        'graph_directional_flow': round(directional_flow, 6),
        'graph_support_strength': round(support_strength, 6),
        'graph_temporal_persistence': round(temporal_persistence, 6),
        'graph_cross_community': round(cross_community, 6),
    }
