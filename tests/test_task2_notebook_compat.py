from __future__ import annotations

import json
from pathlib import Path

import nbformat
import pytest

from scireason.pipeline import task2_validation as pipeline
from scireason.task2_validation import build_task2_validation_bundle


def _sample_yaml() -> Path:
    return sorted((Path(__file__).resolve().parents[1] / 'examples' / 'task2_validation_inputs').glob('*.yaml'))[0]


def test_resolve_papers_from_trajectory_offline_first_does_not_call_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def _boom(*args, **kwargs):
        calls.append('remote')
        raise AssertionError('remote lookup should not be used when disabled')

    monkeypatch.setattr(pipeline, 'search_papers', _boom)
    monkeypatch.setattr(pipeline, 'get_paper_by_doi', _boom)

    doc = json.loads(json.dumps({'papers': [{'id': 'doi:10.1234/demo', 'title': 'Demo paper', 'year': 2024}]}))
    resolved = pipeline.resolve_papers_from_trajectory(doc, enable_remote_lookup=False)

    assert calls == []
    assert len(resolved) == 1
    assert resolved[0].id == 'doi:10.1234/demo'
    assert resolved[0].title == 'Demo paper'
    assert resolved[0].year == 2024


def test_prepare_task2_bundle_offline_first_creates_artifacts(tmp_path: Path) -> None:
    traj = _sample_yaml()
    result = build_task2_validation_bundle(
        traj,
        out_dir=tmp_path,
        include_auto_pipeline=True,
        multimodal=False,
        enable_reference_scout=True,
        enable_remote_lookup=False,
        max_papers=1,
    )

    bundle_dir = result.bundle_dir
    assert (bundle_dir / 'task2_notebook_manifest.json').exists()
    assert (bundle_dir / 'manifest.json').exists()
    assert (bundle_dir / 'reference_graph.json').exists()
    assert (bundle_dir / 'reference_triplets.csv').exists()
    assert (bundle_dir / 'automatic_graph' / 'temporal_kg.json').exists()
    assert (bundle_dir / 'automatic_triplets.csv').exists()
    assert (bundle_dir / 'automatic_triplets_thresholded.json').exists()
    assert (bundle_dir / 'comparison_summary.json').exists()
    assert (bundle_dir / 'scout' / 'suggested_links.json').exists()
    assert (bundle_dir / 'expert_validation' / 'offline_review' / 'task2_expert_validation_offline.html').exists()

    notebook_manifest = json.loads((bundle_dir / 'task2_notebook_manifest.json').read_text(encoding='utf-8'))
    assert notebook_manifest['offline_review_html'].endswith('task2_expert_validation_offline.html')

    manifest = json.loads((bundle_dir / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['remote_lookup_enabled'] is False
    assert manifest['resolved_papers'] >= 1
    assert 'task2_controls' in manifest
    assert 'default_importance_threshold' in manifest

    scout = json.loads((bundle_dir / 'scout' / 'suggested_links.json').read_text(encoding='utf-8'))
    assert scout == []



def test_task2_notebook_exposes_exclusion_and_importance_controls() -> None:
    nb = nbformat.read("notebooks/task2_temporal_graph_validation_colab.ipynb", as_version=4)
    source = nb.cells[4].source
    assert "excluded_paper_ids = W.Textarea" in source
    assert "excluded_paper_titles = W.Textarea" in source
    assert "excluded_title_contains = W.Textarea" in source
    assert "min_importance = W.FloatSlider" in source
    assert "'excluded_paper_ids': [x.strip()" in source
    assert "'min_importance': float(min_importance.value or 0.0)" in source
