from __future__ import annotations

"""Helpers to run Hugging Face `smolagents` CodeAgent inside this repository.

We keep this small and explicit, so students can understand how the external framework is
wired into our pipeline.
"""

import importlib.util
from typing import Any, Dict, List, Optional

from ..config import settings
from .models import create_smol_model


def _require_smolagents() -> None:
    if importlib.util.find_spec("smolagents") is None:
        raise RuntimeError(
            "smolagents не установлен. Установите зависимости: pip install -e '.[agents]'"
        )


def run_code_agent(
    *,
    task: str,
    tools: List[Any],
    system_prompt: Optional[str] = None,
    context: Optional[str] = None,
    max_steps: Optional[int] = None,
    executor_type: Optional[str] = None,
    additional_authorized_imports: Optional[List[str]] = None,
    additional_args: Optional[Dict[str, Any]] = None,
) -> Any:
    """Run a smolagents.CodeAgent and return its final answer.

    smolagents expects:
    - `model`: a callable returning an object with `.content`
    - `tools`: list of `Tool` objects

    We embed `context` inside the prompt to keep it portable.
    """

    _require_smolagents()

    from smolagents import CodeAgent  # type: ignore
    from smolagents.monitoring import LogLevel  # type: ignore

    model = create_smol_model()
    verbosity = LogLevel.INFO if getattr(settings, "smol_print_steps", False) else LogLevel.ERROR

    agent = CodeAgent(
        tools=tools,
        model=model,
        max_steps=int(max_steps or getattr(settings, "hyp_agent_max_steps", 4) or 4),
        executor_type=(executor_type or getattr(settings, "smol_executor", "local") or "local"),
        additional_authorized_imports=additional_authorized_imports,
        verbosity_level=verbosity,
    )

    prompt = ""
    if system_prompt:
        prompt += f"SYSTEM:\n{system_prompt.strip()}\n\n"
    if context:
        prompt += f"CONTEXT:\n{context.strip()}\n\n"
    prompt += task.strip()

    # `additional_args` are injected as variables in the execution environment.
    return agent.run(prompt, additional_args=additional_args or None)
