from __future__ import annotations

from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "MCP extra is not installed. Install with: pip install -e '.[mcp]'"
    ) from e

from .config import MCPServerConfig, load_mcp_server_config
from .manual import register_manual_tools
from .registry import MCPToolRegistry


def _register_auto_tools(mcp: FastMCP, registry: MCPToolRegistry, config: MCPServerConfig) -> list[str]:
    # Register only the auto tools that survived config-based filtering.
    selected_specs = registry.selected_specs(config)
    for spec in selected_specs:
        mcp.tool(
            name=spec.name,
            title=spec.title,
            description=spec.description,
            meta=spec.meta,
            structured_output=spec.structured_output,
        )(spec.func)
    return [spec.name for spec in selected_specs]


def create_mcp_server(name: str = "top-papers-graph") -> FastMCP:
    # Server bootstrap:
    # 1) load env config
    # 2) discover decorated tools
    # 3) add a couple of manual system tools
    config = load_mcp_server_config()
    registry = MCPToolRegistry.discover(extra_modules=config.extra_modules)
    registry.validate_selection(config)

    mcp = FastMCP(name)
    auto_tools = _register_auto_tools(mcp, registry, config)
    manual_tools = register_manual_tools(mcp, registry=registry, config=config)

    setattr(mcp, "_scireason_registry", registry)
    setattr(mcp, "_scireason_config", config)
    setattr(mcp, "_scireason_auto_tools", auto_tools)
    setattr(mcp, "_scireason_manual_tools", manual_tools)
    return mcp


def run_mcp_server(mcp: FastMCP) -> None:
    # stdio is the current default; HTTP transport is left for later experiments.
    config = getattr(mcp, "_scireason_config")
    mcp.run(transport=config.transport, mount_path=config.mount_path)
