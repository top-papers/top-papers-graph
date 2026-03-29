from __future__ import annotations

import importlib.util
from typing import Any

from .config import MCPServerConfig
from .registry import MCPToolRegistry


def register_manual_tools(mcp: Any, *, registry: MCPToolRegistry, config: MCPServerConfig) -> list[str]:
    """Register a couple of tools manually to show the non-decorator path."""

    def doctor_tool() -> dict[str, Any]:
        """Return the current MCP config and the list of discovered tools."""
        data = registry.summary(config)
        data.update(
            {
                "transport": config.transport,
                "mount_path": config.mount_path,
                "manual_tools": ["doctor_tool", "health_tool"],
                "extra_modules": list(config.extra_modules),
            }
        )
        return data

    def health_tool() -> dict[str, Any]:
        """Return a small health payload and optional dependency availability."""
        packages = ["mcp", "fastapi", "neo4j", "qdrant_client", "smolagents"]
        return {
            "ok": True,
            "transport": config.transport,
            "read_only_mode": config.read_only,
            "optional_packages": {
                package_name: bool(importlib.util.find_spec(package_name))
                for package_name in packages
            },
        }

    # Register these directly so both registration styles stay visible in the repo.
    mcp.tool(
        name="doctor_tool",
        title="Doctor Tool",
        description="Inspect MCP configuration and selected tools.",
        meta={"toolset": "system", "read_only": True, "registration_mode": "manual"},
    )(doctor_tool)

    mcp.tool(
        name="health_tool",
        title="Health Tool",
        description="Return lightweight runtime health and dependency availability.",
        meta={"toolset": "system", "read_only": True, "registration_mode": "manual"},
    )(health_tool)

    return ["doctor_tool", "health_tool"]
