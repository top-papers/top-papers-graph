from __future__ import annotations

import types
from pathlib import Path

import pytest


def _find_single_run_dir(out_dir: Path) -> Path:
    run_dirs = [p for p in out_dir.iterdir() if p.is_dir()]
    assert run_dirs, f"No run dirs created in {out_dir}"
    # pick the most recently modified
    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return run_dirs[0]


@pytest.mark.integration
def test_demo_run_smolagents_mock(tmp_path: Path) -> None:
    pytest.importorskip("smolagents")

    from typer.testing import CliRunner

    from scireason.cli import app

    out_dir = tmp_path / "runs"
    runner = CliRunner()

    res = runner.invoke(
        app,
        [
            "demo-run",
            "--query",
            "demo smoke",
            "--edge-mode",
            "cooccurrence",
            "--out-dir",
            str(out_dir),
            "--no-llm-hypotheses",
            "--agent-backend",
            "smolagents",
            "--llm-provider",
            "mock",
            "--smol-model-backend",
            "scireason",
        ],
    )
    assert res.exit_code == 0, res.output

    rp = _find_single_run_dir(out_dir)
    for f in ["paper_records.json", "temporal_kg.json", "hypotheses.json"]:
        assert (rp / f).exists(), f"Missing {f} in {rp}"


@pytest.mark.integration
def test_demo_run_smolagents_g4f_mocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("smolagents")
    g4f = pytest.importorskip("g4f")

    # Patch scireason.llm g4f client so CI doesn't need network.
    import scireason.llm as llm

    class _Msg:
        def __init__(self, content: str):
            self.content = content

    class _Choice:
        def __init__(self, content: str):
            self.message = _Msg(content)

    class _Resp:
        def __init__(self, content: str):
            self.choices = [_Choice(content)]

    class _Chat:
        class completions:
            @staticmethod
            def create(**kwargs):
                # Return deterministic python that produces a list of candidates.
                content = (
                    "G = build_graph()\n"
                    "comms = communities(G, method='greedy', max_communities=6)\n"
                    "bridges = cross_bridges(G, comms, top_k=10)\n"
                    "cands = []\n"
                    "for u,v,s in (bridges or [])[:8]:\n"
                    "    cands.append({'kind':'cross_bridge','source':str(u),'target':str(v),'predicate':'may_relate_to','score':float(s),'graph_signals':{'bridge_score':float(s)}})\n"
                    "if not cands:\n"
                    "    lp = link_prediction(G, method='adamic_adar', k=10)\n"
                    "    for u,v,s in (lp or [])[:8]:\n"
                    "        cands.append({'kind':'link_prediction','source':str(u),'target':str(v),'predicate':'may_relate_to','score':float(s),'graph_signals':{'adamic_adar':float(s)}})\n"
                    "final_answer(cands)\n"
                )
                return _Resp(content)

    class _Client:
        def __init__(self):
            self.chat = _Chat()

    monkeypatch.setattr(llm, "_g4f_client", lambda: _Client())
    monkeypatch.setattr(llm, "_g4f_model_candidates", lambda: ["stub-model"])

    from typer.testing import CliRunner

    from scireason.cli import app

    out_dir = tmp_path / "runs"
    runner = CliRunner()

    res = runner.invoke(
        app,
        [
            "demo-run",
            "--query",
            "demo g4f",
            "--edge-mode",
            "cooccurrence",
            "--out-dir",
            str(out_dir),
            "--no-llm-hypotheses",
            "--agent-backend",
            "smolagents",
            "--llm-provider",
            "mock",
            "--smol-model-backend",
            "g4f",
        ],
    )
    assert res.exit_code == 0, res.output

    rp = _find_single_run_dir(out_dir)
    for f in ["paper_records.json", "temporal_kg.json", "hypotheses.json"]:
        assert (rp / f).exists(), f"Missing {f} in {rp}"
