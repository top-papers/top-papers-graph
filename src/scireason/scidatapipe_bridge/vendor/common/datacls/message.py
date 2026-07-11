# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Chat message models (system / user / assistant)."""
from __future__ import annotations

from typing import List, Literal, Union

from pydantic import BaseModel

from scireason.scidatapipe_bridge.vendor.common.datacls.content import ImageContent, TextContent


class BaseMessage(BaseModel):
    role: str
    content: List[Union[TextContent, ImageContent]]


class SystemMessage(BaseMessage):
    role: Literal["system"] = "system"
    content: List[TextContent]


class UserMessage(BaseMessage):
    role: Literal["user"] = "user"
    content: List[Union[TextContent, ImageContent]]


class AssistantMessage(BaseMessage):
    role: Literal["assistant"] = "assistant"
    content: List[TextContent]


Message = Union[SystemMessage, UserMessage, AssistantMessage]

__all__ = [
    "BaseMessage",
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "Message",
]
