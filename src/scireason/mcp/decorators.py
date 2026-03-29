from __future__ import annotations

from typing import Any, Callable, TypeVar

from .specs import MCP_TOOL_ATTR, MCPToolMarker


F = TypeVar("F", bound=Callable[..., Any])


def scireason_mcp_tool(
    *,
    toolset: str,
    name: str | None = None,
    title: str | None = None,
    description: str | None = None,
    meta: dict[str, Any] | None = None,
    structured_output: bool | None = None,
    read_only: bool = True,
    enabled_by_default: bool = True,
) -> Callable[[F], F]:
    """Mark a function for MCP auto-registration.

    The intended flow is simple:
    1) place the function in ``mcp/toolsets``
    2) decorate it here
    3) keep type hints and a usable docstring
    """

    marker = MCPToolMarker(
        toolset=toolset,
        name=name,
        title=title,
        description=description,
        meta=dict(meta or {}),
        structured_output=structured_output,
        read_only=read_only,
        enabled_by_default=enabled_by_default,
    )

    def _decorator(func: F) -> F:
        # The registry looks for this marker during module discovery.
        setattr(func, MCP_TOOL_ATTR, marker)
        return func

    return _decorator
