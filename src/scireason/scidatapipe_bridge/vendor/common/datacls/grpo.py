"""GRPO / RL sample pydantic model.

Only Task 2 auto-assertions that received a non-empty expert verdict are
promoted to GRPO samples.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from scireason.scidatapipe_bridge.vendor.common.datacls.chat import Chat


class ExpertSignals(BaseModel):
    semantic_correctness: str = ""
    evidence_sufficiency: str = ""
    scope_match: str = ""
    hypothesis_role: str = ""
    causal_status: str = ""
    severity: str = ""
    leakage_risk: str = ""
    time_confidence: str = ""
    mm_verdict: str = ""


class GRPOMetadata(BaseModel):
    submission_id: str
    assertion_id: str
    importance_score: Optional[float] = None
    expert: ExpertSignals = Field(default_factory=ExpertSignals)
    extra: Dict[str, Any] = Field(default_factory=dict)


class GRPOSample(BaseModel):
    """A single GRPO/RL training sample."""

    id: str
    sample_id: str
    task_family: str = "assertion_review_rl"
    reward_task: str = "assertion_review"
    domain: str = ""
    topic: str = ""
    expert_key: str = ""
    source_file: str = ""
    prompt_chat: Chat
    """System + user messages the model sees."""

    reference_json: str
    """JSON-encoded reference verdict + rationale (what the reward model scores against)."""

    reference_assertions_json: str
    """JSON-encoded list of the assertions under review."""

    reference_temporal_json: str
    """JSON-encoded temporal window attached to the assertion."""

    expected_verdict: str
    evidence_text: str = ""
    metadata: GRPOMetadata


__all__ = ["GRPOSample", "GRPOMetadata", "ExpertSignals"]
