# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from .decorators import scireason_mcp_tool
from .runtime import create_mcp_server

__all__ = ["create_mcp_server", "mcp", "scireason_mcp_tool"]


def __getattr__(name: str):
    if name == "mcp":
        from .server import mcp

        return mcp
    raise AttributeError(name)
