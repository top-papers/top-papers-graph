"""Integration with Hugging Face `smolagents`.

This project includes a lightweight internal code-writing agent (see `scireason.agentic`),
but we also provide an *optional* backend using the open-source `smolagents` framework.

Why this exists:
- students can compare a minimal, transparent agent implementation with an industry-style one
  (tool schemas, telemetry hooks, sandboxed execution options)
- smolagents supports multiple model backends, including local Transformers and API-based models

Notes:
- `smolagents` is an optional dependency (install with `pip install -e '.[agents]'`).
- Local HF models require `pip install -e '.[agents_hf]'` (or `smolagents[transformers]`).
"""

from __future__ import annotations

from .models import create_smol_model
from .runner import run_code_agent

__all__ = ["create_smol_model", "run_code_agent"]
