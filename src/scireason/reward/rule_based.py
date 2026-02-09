from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RewardBreakdown:
    score: float
    reasons: List[str]


class RuleBasedReward:
    """Minimal reward model that translates expert feedback into immediate behavior changes.

    Uses overrides compiled from `data/experts/graph_reviews/*.json`.
    - accepted -> +1
    - rejected -> -1
    - needs_fix -> -0.25

    Note: for the strongest effect, have evidence.source_id contain the exact key:
        "subject|predicate|object|time_interval"
    """

    def __init__(self, overrides_path: Optional[Path] = None):
        self.overrides_path = overrides_path
        self._index: Optional[Dict[str, float]] = None

    @staticmethod
    def _key(s: Any, p: Any, o: Any, t: Any) -> str:
        return f"{s}|{p}|{o}|{t}"

    def _load(self) -> Dict[str, float]:
        if self._index is not None:
            return self._index
        idx: Dict[str, float] = {}
        if self.overrides_path and self.overrides_path.exists():
            for line in self.overrides_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                k = self._key(rec.get("subject"), rec.get("predicate"), rec.get("object"), rec.get("time_interval"))
                idx[k] = float(rec.get("weight", 0.0))
        self._index = idx
        return idx

    def score(self, hypothesis: Dict[str, Any]) -> RewardBreakdown:
        idx = self._load()
        score = 0.0
        reasons: List[str] = []

        # 1) Require explicit time/conditions scope
        time_scope = hypothesis.get("time_scope") or hypothesis.get("conditions") or ""
        if not str(time_scope).strip():
            score -= 1.0
            reasons.append("missing time/conditions scope (time_scope)")

        # 2) Evidence requirement
        ev = hypothesis.get("supporting_evidence") or []
        if not isinstance(ev, list) or len(ev) == 0:
            score -= 1.0
            reasons.append("missing supporting_evidence")
        else:
            score += min(0.5, 0.1 * len(ev))
            if len(ev) < 2:
                reasons.append("weak evidence count (<2 citations)")

        # 3) Apply expert weights if source_id encodes triplets
        for cite in ev[:10]:
            sid = str(cite.get("source_id") or "").strip()
            if "|" in sid:
                w = idx.get(sid)
                if w is not None:
                    score += w
                    if w < 0:
                        reasons.append(f"references expert-rejected assertion: {sid}")

        return RewardBreakdown(score=score, reasons=reasons)
