"""SFT sample pydantic model.

One JSONL row corresponds to one :class:`SFTSample`. Two task families are
emitted today:

* ``trajectory_reasoning`` — one sample per Task 1 step,
* ``assertion_reconstruction`` — one sample per Task 2 gold assertion.

The wire format is the ``model_dump()`` of this class (``exclude_none=True``).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from scireason.scidatapipe_bridge.vendor.common.datacls.chat import Chat


class SFTMetadata(BaseModel):
    submission_id: str
    step_id: Optional[int] = None
    assertion_id: Optional[str] = None
    cutoff_year: Optional[int] = None
    importance: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    time_source: Optional[str] = None
    graph_kind: Optional[str] = None
    importance_score: Optional[float] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class SFTSample(BaseModel):
    """A single supervised fine-tuning sample."""

    id: str
    task_family: str
    domain: str = ""
    topic: str = ""
    expert_key: str = ""
    source_file: str = ""
    chat: Chat
    metadata: SFTMetadata


__all__ = ["SFTSample", "SFTMetadata"]
