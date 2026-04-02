import json
from pathlib import Path

from scireason.task2_graph_viz import compute_graph_analytics, write_graph_html


def test_compute_graph_analytics_tolerates_self_loops(tmp_path: Path):
    payload = {
        "nodes": [
            {"id": "A", "label": "A", "type": "term"},
            {"id": "B", "label": "B", "type": "term"},
        ],
        "edges": [
            {"source": "A", "target": "A", "predicate": "self_ref"},
            {"source": "A", "target": "B", "predicate": "relates_to"},
        ],
    }

    analytics = compute_graph_analytics(payload)

    assert analytics["summary"]["node_count"] == 2
    assert analytics["summary"]["edge_count"] == 2
    assert analytics["summary"]["self_loop_count"] == 1
    assert analytics["node_metrics"]["A"]["core_number"] >= 0

    graph_json = tmp_path / "graph.json"
    html_path = tmp_path / "graph.html"
    analytics_path = tmp_path / "analytics.json"
    graph_json.write_text(json.dumps(payload), encoding="utf-8")

    write_graph_html(graph_json, html_path, analytics_path=analytics_path)

    assert html_path.exists()
    assert analytics_path.exists()
