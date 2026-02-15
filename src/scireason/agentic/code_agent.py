from __future__ import annotations

"""A tiny code-writing agent inspired by Hugging Face smolagents.

Goal
----
Provide an educational, dependency-light "CodeAgent" that:
- asks an LLM to write Python code to solve a task
- executes the generated code in a constrained local sandbox
- iterates for a few steps until `final_answer` is produced

It is purposely small and transparent for classroom use.

Security note
-------------
Running LLM-generated code is inherently risky. This sandbox blocks `import` statements
and most dangerous builtins. It is *not* a perfect security boundary (consistent with
smolagents' own guidance). Use additional isolation (Docker / VM) if running untrusted
prompts.
"""

import ast
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from ..llm import chat_text


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    func: Callable[..., Any]


class SandboxError(RuntimeError):
    pass


class _NoImportVisitor(ast.NodeVisitor):
    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        raise SandboxError("`import` is not allowed in agent code. Use provided tools/modules.")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        raise SandboxError("`from ... import ...` is not allowed in agent code. Use provided tools/modules.")

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        # block dunder attribute access (common escape hatch)
        if isinstance(node.attr, str) and node.attr.startswith("__"):
            raise SandboxError("Access to dunder attributes is blocked in agent code.")
        self.generic_visit(node)


def _safe_builtins() -> Dict[str, Any]:
    allowed = {
        "len": len,
        "min": min,
        "max": max,
        "sum": sum,
        "sorted": sorted,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "float": float,
        "int": int,
        "str": str,
        "bool": bool,
        "print": print,
        "abs": abs,
        "round": round,
        "all": all,
        "any": any,
        "isinstance": isinstance,
        "getattr": getattr,
        "hasattr": hasattr,

        # Common exception classes (needed for basic try/except patterns)
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "KeyError": KeyError,
    }
    return allowed


def run_in_sandbox(
    code: str,
    *,
    tools: Mapping[str, Callable[..., Any]],
    globals_extra: Optional[Dict[str, Any]] = None,
    timeout_s: int = 20,
) -> Dict[str, Any]:
    """Execute code with minimal builtins and injected tools.

    The code must end by setting a variable named `final_answer`.
    """

    code = (code or "").strip()
    if not code:
        raise SandboxError("Empty code")

    # 1) AST validation
    try:
        tree = ast.parse(code)
        _NoImportVisitor().visit(tree)
    except SandboxError:
        raise
    except Exception as e:
        raise SandboxError(f"Invalid python code: {e}") from e

    # 2) Build execution namespace
    env: Dict[str, Any] = {
        "__builtins__": _safe_builtins(),
        "tools": dict(tools),
    }
    # Expose tool call shorthand: e.g. `kg = build_graph(...)` instead of `tools['build_graph'](...)`
    env.update({name: fn for name, fn in tools.items()})

    if globals_extra:
        env.update(globals_extra)

    # 3) Execute with a soft timeout
    start = time.time()
    exec(compile(tree, filename="<agent>", mode="exec"), env, env)  # noqa: S102
    if time.time() - start > float(timeout_s):
        raise SandboxError("Execution timed out")

    if "final_answer" not in env:
        raise SandboxError("Agent code did not set `final_answer`")

    return env


class CodeAgent:
    """Minimal iterative code agent."""

    def __init__(
        self,
        *,
        tools: Iterable[Tool],
        system_prompt: str,
        max_steps: int = 4,
        timeout_s: int = 20,
    ) -> None:
        self.tools: List[Tool] = list(tools)
        self.system_prompt = system_prompt
        self.max_steps = max(1, int(max_steps))
        self.timeout_s = max(5, int(timeout_s))

    def _tools_block(self) -> str:
        lines = []
        for t in self.tools:
            lines.append(f"- {t.name}: {t.description}")
        return "\n".join(lines)

    def run(self, task: str, *, context: Optional[str] = None) -> Any:
        tools_map = {t.name: t.func for t in self.tools}
        sys = self.system_prompt.strip() + "\n\nAvailable tools:\n" + self._tools_block()

        history: List[Dict[str, str]] = []
        obs = ""

        for step in range(1, self.max_steps + 1):
            user = (
                f"Task:\n{task.strip()}\n\n"
                + (f"Context:\n{context.strip()}\n\n" if context else "")
                + (f"Previous observation:\n{obs}\n\n" if obs else "")
                + "Write ONLY Python code. Rules:\n"
                + "1) Do NOT use import statements. Use provided tools and built-ins only.\n"
                + "2) End by setting `final_answer` to a JSON-serializable object.\n"
                + "3) Keep code short and robust; handle missing data.\n"
            )

            code = chat_text(sys, user, temperature=0.2)

            # If the model wrapped code in fences, strip them.
            code = code.strip()
            if code.startswith("```"):
                code = code.split("\n", 1)[1]
                if code.rstrip().endswith("```"):
                    code = code.rsplit("```", 1)[0]

            try:
                env = run_in_sandbox(
                    code,
                    tools=tools_map,
                    timeout_s=self.timeout_s,
                    globals_extra={"json": json},
                )
                return env["final_answer"]
            except Exception as e:
                obs = f"Step {step} failed: {type(e).__name__}: {e}"
                history.append({"code": code, "error": obs})
                continue

        raise RuntimeError(f"CodeAgent failed after {self.max_steps} steps. Last: {obs}")
