import os
import pytest

from scireason.agents.debate_graph import run_debate

@pytest.mark.skipif(os.getenv("RUN_LLM_TESTS") != "1", reason="Set RUN_LLM_TESTS=1 to run LLM-dependent tests")
def test_debate_runs():
    res = run_debate(domain="Test", context="Paper: X inhibits Y.", max_rounds=1)
    assert res.rounds >= 1
