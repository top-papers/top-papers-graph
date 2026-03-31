from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal


Transport = Literal["stdio", "sse", "streamable-http"]


def _csv_env(name: str) -> frozenset[str]:
    # Small helper for the env knobs used by the MCP bootstrap.
    raw = os.environ.get(name, "")
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return frozenset(items)


@dataclass(frozen=True)
class MCPServerConfig:
    toolsets: frozenset[str]
    tools: frozenset[str]
    disabled_tools: frozenset[str]
    extra_modules: tuple[str, ...]
    read_only: bool
    transport: Transport
    mount_path: str | None

    @property
    def has_selection_filters(self) -> bool:
        return bool(self.toolsets or self.tools)


def load_mcp_server_config() -> MCPServerConfig:
    # Keep the first version env-driven so the setup stays easy to demo locally.
    extra_modules = tuple(sorted(_csv_env("MCP_EXTRA_MODULES")))
    transport = (os.environ.get("MCP_TRANSPORT") or "stdio").strip().lower()
    if transport not in {"stdio", "sse", "streamable-http"}:
        transport = "stdio"

    mount_path = (os.environ.get("MCP_MOUNT_PATH") or "").strip() or None
    read_only = (os.environ.get("MCP_READ_ONLY") or "").strip().lower() in {"1", "true", "yes", "on"}

    return MCPServerConfig(
        toolsets=_csv_env("MCP_TOOLSETS"),
        tools=_csv_env("MCP_TOOLS"),
        disabled_tools=_csv_env("MCP_DISABLE_TOOLS"),
        extra_modules=extra_modules,
        read_only=read_only,
        transport=transport,  # type: ignore[arg-type]
        mount_path=mount_path,
    )
