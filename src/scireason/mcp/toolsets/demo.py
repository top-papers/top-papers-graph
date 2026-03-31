from __future__ import annotations

import re

from ..decorators import scireason_mcp_tool


@scireason_mcp_tool(toolset="demo")
def demo_echo_tool(message: str, uppercase: bool = False) -> dict[str, str]:
    """Return the incoming text, optionally uppercased."""
    text = message.upper() if uppercase else message
    return {"message": text}


@scireason_mcp_tool(toolset="demo")
def demo_keywords_tool(text: str, limit: int = 5) -> list[str]:
    """Extract a few simple keyword-like tokens without external dependencies."""
    tokens = re.findall(r"[a-zA-Z0-9_-]{3,}", text.lower())
    uniq: list[str] = []
    for token in tokens:
        if token not in uniq:
            uniq.append(token)
        if len(uniq) >= max(1, limit):
            break
    return uniq
