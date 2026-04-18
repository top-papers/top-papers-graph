"""Chat aggregate — an ordered list of messages."""
from __future__ import annotations

from typing import List

from pydantic import BaseModel

from scireason.scidatapipe_bridge.vendor.common.datacls.message import Message


class Chat(BaseModel):
    """Ordered list of chat messages used as training input."""

    messages: List[Message]


__all__ = ["Chat"]
