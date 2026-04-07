import json
from pathlib import Path

from scireason.task2_filters import (
    cooccurrence_strength_score,
    entity_matches_exclusion,
    is_cooccurrence_triplet,
    is_likely_causal_triplet,
    normalize_exclusion_spec,
    score_triplet_importance,
    should_filter_cooccurrence,
)
from scireason.task2_graph_viz import compute_graph_analytics, write_graph_html_variants


def test_exclusion_spec_matches_by_any_identifier_title_or_link() -> None:
    spec = normalize_exclusion_spec(
        {
            'identifier': ['10.1000/demo'],
            'titles': ['A Demo Paper About Temporal Graphs'],
            'links': ['https://example.org/papers/demo.pdf'],
        }
    )

    row = {
        'subject': 'Temporal graph',
        'predicate': 'supported_by',
        'object': 'Demo result',
        'paper_ids': ['doi:10.1000/demo'],
        'paper_titles': ['A Demo Paper About Temporal Graphs'],
        'paper_source_refs': ['https://example.org/papers/demo.pdf'],
        'raw_record_json': json.dumps(
            {
                'papers': [
                    {
                        'id': 'doi:10.1000/demo',
                        'title': 'A Demo Paper About Temporal Graphs',
                        'url': 'https://example.org/papers/demo.pdf',
                    }
                ]
            },
            ensure_ascii=False,
        ),
    }

    assert entity_matches_exclusion(row, spec) is True
    assert entity_matches_exclusion(row, {'titles': ['A Demo Paper About Temporal Graphs']}) is True
    assert entity_matches_exclusion(row, {'doi': ['10.1000/demo']}) is True
    assert entity_matches_exclusion(row, {'links': ['https://example.org/papers/demo.pdf']}) is True


def test_topology_importance_uses_graph_metrics_only() -> None:
    payload = {
        'nodes': [
            {'id': 'A', 'label': 'A', 'type': 'term'},
            {'id': 'B', 'label': 'B', 'type': 'term'},
            {'id': 'C', 'label': 'C', 'type': 'term'},
        ],
        'edges': [
            {
                'source': 'A',
                'target': 'B',
                'predicate': 'relates_to',
                'papers': ['doi:10.1000/demo'],
                'evidence_quotes': ['quote'],
                'yearly_count': {'2020': 1, '2021': 1},
                'total_count': 3,
            },
            {
                'source': 'B',
                'target': 'C',
                'predicate': 'relates_to',
                'papers': ['doi:10.1000/other'],
                'evidence_quotes': ['quote'],
                'yearly_count': {'2021': 1},
                'total_count': 1,
            },
        ],
    }
    analytics = compute_graph_analytics(payload)
    row = {
        'subject': 'A',
        'predicate': 'relates_to',
        'object': 'B',
        'papers': ['doi:10.1000/demo'],
        'evidence_quotes': ['quote'],
        'yearly_count': {'2020': 1, '2021': 1},
        'total_count': 3,
    }

    scored = score_triplet_importance(row, {'topic': 'ignored'}, graph_metrics=analytics)

    assert scored['importance_model'] == 'temporal_graph_topology_v2'
    assert 0.0 <= float(scored['importance_score']) <= 1.0
    assert float(scored['graph_edge_betweenness']) >= 0.0
    assert float(scored['graph_pagerank_mean']) >= 0.0


def test_write_graph_html_variants_creates_full_and_light_files(tmp_path: Path) -> None:
    payload = {
        'nodes': [
            {'id': 'A', 'label': 'A', 'type': 'term'},
            {'id': 'B', 'label': 'B', 'type': 'term'},
        ],
        'edges': [
            {
                'source': 'A',
                'target': 'B',
                'predicate': 'relates_to',
                'papers': ['doi:10.1000/demo'],
                'yearly_count': {'2024': 1},
                'total_count': 1,
            }
        ],
    }
    graph_json = tmp_path / 'graph.json'
    full_html = tmp_path / 'graph.html'
    light_html = tmp_path / 'graph_light.html'
    analytics_json = tmp_path / 'graph_analytics.json'
    graph_json.write_text(json.dumps(payload), encoding='utf-8')

    outputs = write_graph_html_variants(
        graph_json,
        full_html,
        analytics_path=analytics_json,
        light_html_path=light_html,
    )

    assert outputs['full'].exists()
    assert outputs['light'].exists()
    assert analytics_json.exists()
    analytics = json.loads(analytics_json.read_text(encoding='utf-8'))
    assert analytics['edge_metrics']
    assert 'edge_betweenness' in next(iter(analytics['edge_metrics'].values()))
    assert 'облегченная версия' in outputs['light'].read_text(encoding='utf-8').lower()


def test_weak_cooccurrence_filter_prefers_non_cooccurrence_or_stronger_edges() -> None:
    weak_row = {
        'predicate': 'cooccurs_with',
        'importance_score': 0.22,
        'evidence_text': 'A and B co-occur in one sentence',
    }
    strong_row = {
        'predicate': 'cooccurs_with',
        'importance_score': 0.71,
        'evidence_text': 'A and B co-occur repeatedly across the graph',
    }
    causal_row = {
        'predicate': 'causes',
        'importance_score': 0.18,
        'evidence_text': 'A causes B in the described mechanism',
    }

    assert is_cooccurrence_triplet(weak_row) is True
    assert is_likely_causal_triplet(causal_row) is True
    assert cooccurrence_strength_score(weak_row) == 0.22
    assert should_filter_cooccurrence(weak_row, mode='hide_weak', weak_threshold=0.45) is True
    assert should_filter_cooccurrence(strong_row, mode='hide_weak', weak_threshold=0.45) is False
    assert should_filter_cooccurrence(causal_row, mode='hide_weak', weak_threshold=0.45) is False
    assert should_filter_cooccurrence(weak_row, mode='exclude_all', weak_threshold=0.45) is True
