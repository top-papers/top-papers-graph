"""Content blocks inside a chat message.

A message carries a list of content blocks. For the pipeline MVP we only
need two kinds — ``text`` and ``image`` — but the union form keeps the
door open for audio/video later without breaking the wire format.
"""
from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel


class BaseContent(BaseModel):
    """Shared envelope for every content block."""

    trainable: bool = False
    """Whether this block should contribute to the training loss."""

    meta: Optional[Dict[str, Any]] = None
    """Free-form metadata (caption, paper_id, figure number, ...)."""

    def get_content(self) -> Any:
        raise NotImplementedError


class TextContent(BaseContent):
    """Plain-text content block."""

    type: Literal["text"] = "text"
    text: str

    def get_content(self) -> str:
        return self.text


class ImageContent(BaseContent):
    """Image reference (local path or URL)."""

    type: Literal["image"] = "image"
    image: str
    """Path or URL pointing at the image asset."""

    def get_content(self) -> str:
        return self.image


__all__ = ["BaseContent", "TextContent", "ImageContent"]
