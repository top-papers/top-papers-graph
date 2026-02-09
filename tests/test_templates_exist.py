from pathlib import Path

def test_templates_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "data/experts/trajectories/_template.yaml").exists()
    assert (root / "data/experts/graph_reviews/_template.json").exists()
    assert (root / "data/experts/hypothesis_reviews/_template.json").exists()
