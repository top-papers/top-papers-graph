from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


MCP_TOOL_ATTR = "__scireason_mcp_tool_marker__"


@dataclass(frozen=True)
class MCPToolMarker:
    toolset: str
    name: str | None = None
    title: str | None = None
    description: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    structured_output: bool | None = None
    read_only: bool = True
    enabled_by_default: bool = True


@dataclass(frozen=True)
class MCPToolSpec:
    func: Callable[..., Any]
    toolset: str
    name: str
    title: str
    description: str
    meta: dict[str, Any] = field(default_factory=dict)
    structured_output: bool | None = None
    read_only: bool = True
    enabled_by_default: bool = True
    origin_module: str = ""
