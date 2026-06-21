#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import importlib.util
import json
import os
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Image as HFImage, Sequence
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
try:
    from peft import PeftModel
except ImportError:  # lets lightweight regression tests stub peft without a full install
    PeftModel = None
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)

try:
    from transformers import AutoModelForImageTextToText
except ImportError:  # pragma: no cover - compatibility with older Transformers aliases
    from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:  # pragma: no cover - fallback to AutoModelForImageTextToText
    Qwen3VLForConditionalGeneration = None

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:  # pragma: no cover - fallback to AutoModelForImageTextToText
    Qwen2_5_VLForConditionalGeneration = None


def install_torch_fsdp_module_import_compat() -> bool:
    """Make newer TRL GRPO imports work with torch 2.5.x.

    Recent TRL versions import ``FSDPModule`` from ``torch.distributed.fsdp``
    while the DataSphere-compatible torch 2.5.1+cu121 build exposes the legacy
    ``FullyShardedDataParallel`` symbol instead. This job uses DDP, not FSDP;
    the alias is only needed so TRL's optional FSDP helpers can be imported.
    """
    try:
        import torch.distributed.fsdp as fsdp
    except Exception:
        return False
    if hasattr(fsdp, "FSDPModule"):
        return False
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel
    except Exception:
        return False
    setattr(fsdp, "FSDPModule", FullyShardedDataParallel)
    return True


install_torch_fsdp_module_import_compat()
from trl import GRPOConfig, GRPOTrainer

# Parsing and logging

class RewardTraceLogger:
    def __init__(self, path: Path | None = None):
        self.path = path
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                self.path.unlink()

    def append(self, component: str, trainer_state: Any, sample_ids: List[str], domains: List[str], task_families: List[str], rewards: List[Optional[float]]):
        if not self.path:
            return
        step = getattr(trainer_state, "global_step", None) if trainer_state is not None else None
        epoch = getattr(trainer_state, "epoch", None) if trainer_state is not None else None
        
        with self.path.open("a", encoding="utf-8") as f:
            for sid, domain, task_family, reward in zip(sample_ids, domains, task_families, rewards):
                f.write(json.dumps({
                    "component": component,
                    "global_step": step,
                    "epoch": epoch,
                    "sample_id": sid,
                    "domain": domain,
                    "task_family": task_family,
                    "reward": reward,
                }, ensure_ascii=False) + "\n")

# Global logger
REWARD_LOGGER = RewardTraceLogger(None)


# Task-aware reward routing.  Components that are not semantically active for a
# task family are audit-only and do not contribute to the GRPO objective.  When
# TRL supplies duplicated sample_id values for a generation group, active
# components are robustly normalized within that group so cheap saturated
# features do not dominate the group advantage.
ACTIVE_REWARD_COMPONENTS_BY_TASK: Dict[str, set[str]] = {
    "assertion_review_rl": {"schema", "evidence", "verdict", "temporal"},
    "mm_review_rl": {"schema", "evidence", "verdict"},
    "trajectory_reasoning_rl": {"schema", "evidence", "graph"},
    "image_label_rl": {"schema", "label"},
    "temporal_fix_rl": {"schema", "temporal", "verdict"},
}
COMPONENT_ROUTING_KEY: Dict[str, str] = {
    "label_exact_match": "label",
    "schema_validity": "schema",
    "temporal_consistency": "temporal",
    "graph_consistency": "graph",
    "evidence_presence": "evidence",
    "expert_override_match": "verdict",
}


def _mad(values: list[float]) -> float:
    if not values:
        return 0.0
    med = statistics.median(values)
    return statistics.median([abs(v - med) for v in values])


def _has_repeated_sample_id(sample_ids: list[str]) -> bool:
    seen: set[str] = set()
    for sid in sample_ids:
        sid = str(sid or "")
        if sid in seen:
            return True
        seen.add(sid)
    return False


def _groupwise_robust_norm(sample_ids: list[str], values: list[float], *, clip: float = 2.5) -> list[float]:
    by_group: Dict[str, list[tuple[int, float]]] = {}
    for idx, (sid, value) in enumerate(zip(sample_ids, values)):
        by_group.setdefault(str(sid or ""), []).append((idx, float(value)))
    out = [float(v) for v in values]
    for items in by_group.values():
        if len(items) < 2:
            continue
        vals = [v for _, v in items]
        med = statistics.median(vals)
        scale = max(_mad(vals) * 1.4826, 1e-4)
        for idx, value in items:
            z = (value - med) / scale
            out[idx] = max(-clip, min(clip, z))
    return out


def apply_task_aware_reward_processing(
    component_name: str,
    rewards: list[float],
    *,
    sample_id: list[str] | None,
    task_family: list[str] | None,
    normalize: bool,
    clip: float = 2.5,
    temperature: float = 1.5,
) -> list[float]:
    key = COMPONENT_ROUTING_KEY.get(component_name)
    families = task_family or [""] * len(rewards)
    routed: list[float] = []
    active_idx: list[int] = []
    for idx, (reward, tf) in enumerate(zip(rewards, families)):
        active = key is None or key in ACTIVE_REWARD_COMPONENTS_BY_TASK.get(str(tf or ""), {key})
        if active:
            routed.append(float(reward))
            active_idx.append(idx)
        else:
            routed.append(0.0)
    if not normalize or not sample_id or not _has_repeated_sample_id([str(x or "") for x in sample_id]):
        return routed
    active_ids = [str(sample_id[i] or "") for i in active_idx]
    active_values = [routed[i] for i in active_idx]
    norm_values = _groupwise_robust_norm(active_ids, active_values, clip=clip)
    for pos, idx in enumerate(active_idx):
        routed[idx] = math.tanh(norm_values[pos] / max(float(temperature), 1e-6))
    return routed


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return mean, var ** 0.5


def audit_reward_trace_file(path: Path, *, min_reward_std: float, max_zero_std_frac: float) -> dict[str, Any]:
    """Summarize component reward variance after a GRPO run.

    The previous audit showed `eval_frac_reward_zero_std ~= 0.988`, which makes
    group-relative advantages collapse.  This lightweight post-run audit catches
    the same failure mode even when TRL's metric names change.
    """
    if not path.exists():
        return {"status": "missing", "path": str(path), "weak_reward": True, "reason": "reward trace file was not created"}
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            reward = obj.get("reward")
            if isinstance(reward, (int, float)):
                rows.append(obj)
    by_component: dict[str, list[float]] = {}
    by_group: dict[tuple[Any, str, str], list[float]] = {}
    for obj in rows:
        component = str(obj.get("component") or "unknown")
        reward = float(obj["reward"])
        by_component.setdefault(component, []).append(reward)
        by_group.setdefault((obj.get("global_step"), component, str(obj.get("sample_id") or "")), []).append(reward)
    component_stats = {}
    weak_components = []
    for component, values in sorted(by_component.items()):
        mean, std = _mean_std(values)
        component_stats[component] = {"count": len(values), "mean": round(mean, 6), "std": round(std, 6), "min": min(values), "max": max(values)}
        if (
            len(values) >= 8
            and std < min_reward_std
            and component not in {"label_exact_match"}
            and not (abs(min(values)) < 1e-12 and abs(max(values)) < 1e-12)
        ):
            weak_components.append(component)
    group_stds = []
    active_group_stds = []
    for values in by_group.values():
        if len(values) > 1:
            std = _mean_std(values)[1]
            group_stds.append(std)
            # Ignore exactly all-zero task-inapplicable reward groups. They are
            # still reported in group_zero_std_fraction, but only active groups
            # should decide whether the useful GRPO reward signal collapsed.
            if any(abs(float(v)) > 1e-12 for v in values):
                active_group_stds.append(std)
    zero_std_frac = (sum(1 for x in group_stds if x < 1e-9) / len(group_stds)) if group_stds else 1.0
    active_zero_std_frac = (
        sum(1 for x in active_group_stds if x < 1e-9) / len(active_group_stds)
        if active_group_stds else 1.0
    )
    weak_reward = bool(weak_components) or active_zero_std_frac > max_zero_std_frac
    return {
        "status": "fail" if weak_reward else "pass",
        "path": str(path),
        "rows": len(rows),
        "component_stats": component_stats,
        "weak_components": weak_components,
        "group_count": len(group_stds),
        "group_zero_std_fraction": round(zero_std_frac, 6),
        "active_group_zero_std_fraction": round(active_zero_std_frac, 6),
        "active_group_count": len(active_group_stds),
        "thresholds": {"min_reward_std": min_reward_std, "max_zero_std_frac": max_zero_std_frac},
        "weak_reward": weak_reward,
    }

def norm_text(value: Any) -> str:
    return " ".join(str(value).strip().lower().split()) if value is not None else ""

def first_nonempty(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return default

def compact_dict(obj: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in obj.items():
        if isinstance(v, dict):
            vv = compact_dict(v)
            if vv: out[k] = vv
        elif isinstance(v, list):
            vv = [x for x in v if x not in (None, "", [], {})]
            if vv: out[k] = vv
        elif v not in (None, "", [], {}):
            out[k] = v
    return out

def compact_temporal(row: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    return compact_dict({
        "start_date": row.get(prefix + "start_date"),
        "end_date": row.get(prefix + "end_date"),
        "valid_from": row.get(prefix + "valid_from"),
        "valid_to": row.get(prefix + "valid_to"),
        "time_source": row.get(prefix + "time_source"),
    })

def flatten_assertions(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, dict):
        if isinstance(obj.get("assertions"), list):
            return [x for x in obj["assertions"] if isinstance(x, dict)]
        if isinstance(obj.get("extracted_assertions"), list):
            return [x for x in obj["extracted_assertions"] if isinstance(x, dict)]
        if all(k in obj for k in ("subject", "predicate", "object")):
            return [obj]
        if "prediction" in obj:
            return flatten_assertions(obj["prediction"])
    return []

def triple_key(x: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    time_obj = x.get("time") or {}
    return (
        norm_text(x.get("subject")),
        norm_text(x.get("predicate")),
        norm_text(x.get("object")),
        norm_text(first_nonempty(x.get("start_date"), time_obj.get("start"))),
        norm_text(first_nonempty(x.get("end_date"), time_obj.get("end"))),
    )

def completion_to_text(completion: Any) -> str:
    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        if "content" in completion:
            return completion_to_text(completion["content"])
        return json.dumps(completion, ensure_ascii=False)
    if isinstance(completion, list):
        return "\n".join([completion_to_text(item) for item in completion])
    return str(completion)

def parse_json_candidate(text: str) -> Optional[Any]:
    if not text:
        return None
    text = text.strip()
    
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text)
    
    for candidate in fenced + [text]:
        try:
            return json.loads(candidate.strip())
        except Exception:
            pass

    spans = []
    for opener, closer in [("{", "}"), ("[", "]")]:
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            spans.append(text[start:end+1])
    for candidate in sorted(spans, key=len, reverse=True):
        try:
            return json.loads(candidate)
        except Exception:
            pass
    return None



def parse_prediction_object(completion: Any) -> Optional[Any]:
    return parse_json_candidate(completion_to_text(completion))



VERDICT_ALIASES = {
    "accept": {"accept", "accepted", "valid", "supported", "true", "correct", "ok", "approve", "approved"},
    "reject": {"reject", "rejected", "invalid", "unsupported", "false", "incorrect", "wrong", "deny", "denied", "not supported"},
    "revise": {"revise", "revision", "needs revision", "needs_revision", "fix", "modify", "partial", "needs changes"},
}


def canonical_verdict(value: Any) -> str:
    text = norm_text(value).replace("_", " ").replace("-", " ")
    if not text:
        return ""
    # Exact alias matching first; avoid classifying "unsupported" as "accept"
    # merely because it contains the substring "supported".
    for canonical, aliases in VERDICT_ALIASES.items():
        if text == canonical or text in aliases:
            return canonical
    if text.startswith("un") and ("support" in text or "valid" in text or "correct" in text):
        return "reject"
    if "not" in text and ("support" in text or "valid" in text or "correct" in text):
        return "reject"
    for canonical, aliases in VERDICT_ALIASES.items():
        if any((" " in alias) and alias in text for alias in aliases):
            return canonical
    return text


def canonical_label(value: Any) -> str:
    obj = parse_json_candidate(completion_to_text(value)) if not isinstance(value, (dict, list)) else value
    if isinstance(obj, dict):
        for key in ("label", "class_label", "answer", "prediction", "verdict", "category"):
            if obj.get(key) not in (None, ""):
                return norm_text(obj.get(key))
    return norm_text(value)

def parse_label_candidate(completion: Any) -> str:
    text = completion_to_text(completion).strip()
    obj = parse_json_candidate(text)
    if isinstance(obj, dict):
        for key in ("label", "class_label", "answer", "prediction"):
            if obj.get(key) not in (None, ""):
                return str(obj[key]).strip()
    if isinstance(obj, str):
        return obj.strip()
    return text.splitlines()[0].strip() if text else ""


def expected_schema_keys_for_task(task_family: str) -> tuple[str, ...]:
    if task_family == "trajectory_reasoning_rl":
        return ("inference", "next_question", "extracted_assertions")
    if task_family in {"assertion_review_rl", "mm_review_rl"}:
        return ("verdict", "rationale") if task_family == "assertion_review_rl" else ("mm_verdict", "mm_rationale")
    if task_family == "temporal_fix_rl":
        return ("start_date", "end_date", "valid_from", "valid_to", "time_source")
    if task_family == "image_label_rl":
        return ("label",)
    return ()


def partial_json_progress_reward(completion: Any, task_family: str) -> float:
    """Dense shaping for truncated/near-JSON completions.

    Smoke GRPO runs used a 32-token completion cap. With strict JSON-only rewards
    every clipped candidate received exactly the same negative reward, so GRPO
    normalized each generation group to zero advantage and reported loss=0. This
    helper keeps invalid JSON negative, but gives small deterministic credit for
    schema progress so groups can have non-zero reward variance during smoke/debug
    runs.
    """
    text = completion_to_text(completion).strip()
    if not text:
        return -1.0

    lowered = text.lower()
    score = -1.0
    if "{" in text or "[" in text:
        score += 0.15
    if "}" in text or "]" in text:
        score += 0.10
    if '"' in text or "'" in text:
        score += 0.05
    if text.startswith("{") or text.startswith("[") or text.startswith("```"):
        score += 0.05

    expected_keys = expected_schema_keys_for_task(task_family)
    if expected_keys:
        matches = sum(1 for key in expected_keys if key.lower() in lowered)
        score += min(0.45, 0.15 * matches)

    verdict_words = ("accept", "reject", "revise", "yes", "no", "true", "false")
    if task_family in {"assertion_review_rl", "mm_review_rl", "temporal_fix_rl"} and any(word in lowered for word in verdict_words):
        score += 0.10

    # Keep malformed JSON below valid-object rewards.
    return max(-1.0, min(0.25, score))


def partial_field_presence_reward(completion: Any, keys: tuple[str, ...], *, empty_reward: float) -> float:
    text = completion_to_text(completion).strip().lower()
    if not text:
        return empty_reward
    if not keys:
        return empty_reward
    matches = sum(1 for key in keys if key.lower() in text)
    if matches <= 0:
        return empty_reward
    return min(-0.05, empty_reward + 0.15 * matches)


def reward_label_exact_match(prompts, completions, reference_label=None, label_text=None, sample_id=None, domain=None, task_family=None, trainer_state=None, **kwargs):
    task_family = task_family or [""] * len(completions)
    _normalize_rewards = sample_id is not None
    reference_label = reference_label or label_text or [None] * len(completions)
    sample_id = sample_id or [""] * len(completions)
    domain = domain or [""] * len(completions)

    rewards = []
    for completion, target, tf in zip(completions, reference_label, task_family):
        if tf != "image_label_rl" or not target:
            rewards.append(0.0)
            continue
        pred = canonical_label(parse_label_candidate(completion))
        target_norm = canonical_label(target)
        if pred == target_norm:
            rewards.append(1.0)
        elif target_norm and target_norm in pred:
            rewards.append(0.5)
        else:
            rewards.append(-1.0)
    rewards = apply_task_aware_reward_processing(
        "label_exact_match",
        rewards,
        sample_id=sample_id,
        task_family=task_family,
        normalize=_normalize_rewards,
    )
    REWARD_LOGGER.append("label_exact_match", trainer_state, sample_id, domain, task_family, rewards)
    return rewards


# 2. Reward functions for RL

def reward_schema_validity(prompts, completions, sample_id=None, domain=None, task_family=None, trainer_state=None, **kwargs):
    task_family = task_family or [""] * len(completions)
    _normalize_rewards = sample_id is not None
    sample_id = sample_id or [""] * len(completions)
    domain = domain or [""] * len(completions)

    rewards = []
    for completion, tf in zip(completions, task_family):
        obj = parse_prediction_object(completion)
        reward = partial_json_progress_reward(completion, tf)
        if isinstance(obj, dict):
            reward = 0.5
            if tf == "trajectory_reasoning_rl":
                if {"inference", "next_question", "extracted_assertions"}.issubset(obj.keys()):
                    reward = 1.0
            elif tf in {"assertion_review_rl", "mm_review_rl"}:
                if "verdict" in obj or "mm_verdict" in obj:
                    reward = 1.0
            elif tf == "temporal_fix_rl":
                if any(k in obj for k in ["start_date", "end_date", "valid_from", "valid_to", "time_source"]):
                    reward = 1.0
        rewards.append(reward)
    rewards = apply_task_aware_reward_processing(
        "schema_validity",
        rewards,
        sample_id=sample_id,
        task_family=task_family,
        normalize=_normalize_rewards,
    )
    REWARD_LOGGER.append("schema_validity", trainer_state, sample_id, domain, task_family, rewards)
    return rewards

def reward_graph_consistency(prompts, completions, reference_assertions_json=None, sample_id=None, domain=None, task_family=None, trainer_state=None, **kwargs):
    task_family = task_family or [""] * len(completions)
    _normalize_rewards = sample_id is not None
    reference_assertions_json = reference_assertions_json or [None] * len(completions)
    sample_id = sample_id or [""] * len(completions)
    domain = domain or [""] * len(completions)

    rewards = []
    for completion, ref_json, tf in zip(completions, reference_assertions_json, task_family):
        if tf != "trajectory_reasoning_rl":
            rewards.append(0.0)
            continue
        
        pred_obj = parse_prediction_object(completion)
        pred_keys = {triple_key(x) for x in flatten_assertions(pred_obj)}
        
        try:
            ref_assertions = json.loads(ref_json) if isinstance(ref_json, str) else (ref_json or [])
        except Exception:
            ref_assertions = []
        ref_keys = {triple_key(x) for x in ref_assertions if isinstance(x, dict)}
        
        if not pred_keys:
            # Review-style RL examples usually ask for a verdict/rationale, not
            # for reconstructing the full assertion graph. Penalize missing graph
            # output for trajectory reconstruction, but keep assertion review
            # neutral unless the model actually proposes assertions to compare.
            rewards.append(-1.0 if ref_keys else 0.0)
            continue

        if not ref_keys:
            rewards.append(-0.25)
            continue

        tp = len(pred_keys & ref_keys)
        fp, fn = len(pred_keys - ref_keys), len(ref_keys - pred_keys)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rewards.append(2.0 * f1 - 1.0)
        
    rewards = apply_task_aware_reward_processing(
        "graph_consistency",
        rewards,
        sample_id=sample_id,
        task_family=task_family,
        normalize=_normalize_rewards,
    )
    REWARD_LOGGER.append("graph_consistency", trainer_state, sample_id, domain, task_family, rewards)
    return rewards

def reward_temporal_consistency(prompts, completions, reference_temporal_json=None, sample_id=None, domain=None, task_family=None, trainer_state=None, **kwargs):
    task_family = task_family or [""] * len(completions)
    _normalize_rewards = sample_id is not None
    reference_temporal_json = reference_temporal_json or [None] * len(completions)
    sample_id = sample_id or [""] * len(completions)
    domain = domain or [""] * len(completions)

    rewards = []
    for completion, ref_json, tf in zip(completions, reference_temporal_json, task_family):
        if tf not in {"assertion_review_rl", "temporal_fix_rl"}:
            rewards.append(0.0)
            continue
            
        pred_obj = parse_prediction_object(completion)
        if not isinstance(pred_obj, dict):
            rewards.append(-1.0)
            continue
            
        pred_temporal = pred_obj.get("corrected_temporal") if isinstance(pred_obj.get("corrected_temporal"), dict) else compact_temporal(pred_obj)
        try:
            ref_temporal = json.loads(ref_json) if isinstance(ref_json, str) else (ref_json or {})
        except Exception:
            ref_temporal = {}
            
        matched, total = 0, 0
        for field in ["start_date", "end_date", "valid_from", "valid_to", "time_source"]:
            if ref_temporal.get(field) not in (None, "", "unknown"):
                total += 1
                matched += int(norm_text(pred_temporal.get(field)) == norm_text(ref_temporal.get(field)))
        rewards.append((matched / total) if total else 0.0)
        
    rewards = apply_task_aware_reward_processing(
        "temporal_consistency",
        rewards,
        sample_id=sample_id,
        task_family=task_family,
        normalize=_normalize_rewards,
    )
    REWARD_LOGGER.append("temporal_consistency", trainer_state, sample_id, domain, task_family, rewards)
    return rewards

def reward_evidence_presence(prompts, completions, evidence_text=None, sample_id=None, domain=None, task_family=None, trainer_state=None, **kwargs):
    task_family = task_family or [""] * len(completions)
    _normalize_rewards = sample_id is not None
    evidence_text = evidence_text or [None] * len(completions)
    sample_id = sample_id or [""] * len(completions)
    domain = domain or [""] * len(completions)

    rewards = []
    for completion, ev_text, tf in zip(completions, evidence_text, task_family):
        if tf == "image_label_rl":
            rewards.append(0.0)
            continue
        pred_obj = parse_prediction_object(completion)
        if not isinstance(pred_obj, dict):
            rewards.append(partial_field_presence_reward(completion, ("evidence", "rationale", "snippet", "source"), empty_reward=-0.5))
            continue
            
        if tf == "trajectory_reasoning_rl":
            assertions = flatten_assertions(pred_obj)
            has_evidence = False
            for a in assertions:
                ev = a.get("evidence") or {}
                if isinstance(ev, dict) and any(ev.get(k) not in (None, "", "unknown") for k in ["source", "locator", "snippet_or_summary"]):
                    has_evidence = True
                    break
            rewards.append(1.0 if has_evidence else -0.25)
        elif tf in {"assertion_review_rl", "mm_review_rl"}:
            rationale = completion_to_text(pred_obj.get("rationale") or pred_obj.get("mm_rationale") or "")
            rewards.append(1.0 if rationale.strip() else -0.25)
        elif tf == "temporal_fix_rl":
            comment = completion_to_text(pred_obj.get("comment") or "")
            rewards.append(0.5 if comment.strip() else 0.0)
        else:
            rewards.append(0.0)
            
    rewards = apply_task_aware_reward_processing(
        "evidence_presence",
        rewards,
        sample_id=sample_id,
        task_family=task_family,
        normalize=_normalize_rewards,
    )
    REWARD_LOGGER.append("evidence_presence", trainer_state, sample_id, domain, task_family, rewards)
    return rewards

def reward_expert_override_match(prompts, completions, expected_verdict=None, sample_id=None, domain=None, task_family=None, trainer_state=None, **kwargs):
    task_family = task_family or [""] * len(completions)
    _normalize_rewards = sample_id is not None
    expected_verdict = expected_verdict or [None] * len(completions)
    sample_id = sample_id or [""] * len(completions)
    domain = domain or [""] * len(completions)

    rewards = []
    for completion, verdict, tf in zip(completions, expected_verdict, task_family):
        if tf not in {"assertion_review_rl", "mm_review_rl", "temporal_fix_rl"}:
            rewards.append(0.0)
            continue
        verdict_norm = canonical_verdict(verdict)
        if not verdict_norm:
            rewards.append(0.0)
            continue

        pred_obj = parse_prediction_object(completion)
        if not isinstance(pred_obj, dict):
            text_norm = canonical_verdict(completion_to_text(completion))
            # Dense signal for clipped JSON/text: exact mention of the target is
            # useful but must stay below a valid parsed JSON answer.
            if text_norm == verdict_norm or verdict_norm in norm_text(completion_to_text(completion)):
                rewards.append(0.35)
            else:
                rewards.append(partial_field_presence_reward(completion, ("verdict", "mm_verdict"), empty_reward=-0.75))
            continue

        if tf == "mm_review_rl":
            pred_verdict = pred_obj.get("mm_verdict") or pred_obj.get("verdict")
        else:
            pred_verdict = pred_obj.get("verdict") or pred_obj.get("expected_verdict")

        pred_norm = canonical_verdict(pred_verdict)
        rationale = completion_to_text(
            pred_obj.get("rationale") or pred_obj.get("mm_rationale") or pred_obj.get("comment") or ""
        ).strip()
        if pred_norm == verdict_norm:
            rewards.append(1.0 if rationale else 0.75)
        elif not pred_norm:
            rewards.append(-0.9)
        elif pred_norm in {"accept", "reject", "revise"}:
            rewards.append(-0.55)
        elif verdict_norm in norm_text(completion_to_text(completion)):
            rewards.append(0.15)
        else:
            rewards.append(-0.9)
            
    rewards = apply_task_aware_reward_processing(
        "expert_override_match",
        rewards,
        sample_id=sample_id,
        task_family=task_family,
        normalize=_normalize_rewards,
    )
    REWARD_LOGGER.append("expert_override_match", trainer_state, sample_id, domain, task_family, rewards)
    return rewards


# 3. Dataset and model preparation

def get_world_size() -> int:
    try:
        return int(os.environ.get('WORLD_SIZE', '1'))
    except ValueError:
        return 1


def resolve_ddp_find_unused_parameters(args: argparse.Namespace, actual_mode: str) -> bool:
    """Return the DDP unused-parameter setting for GRPOConfig.

    VLM + LoRA/adapter training under DDP can leave some trainable branches
    unused on a rank for a particular step. Enable DDP's unused-parameter
    detection automatically for multi-process VLM runs, while preserving the
    explicit CLI override and the faster text-only/single-process default.
    """
    requested = getattr(args, 'ddp_find_unused_parameters', None)
    if requested is not None:
        return bool(requested)
    return actual_mode == 'vlm' and get_world_size() > 1


def enforce_minimum_grpo_generations(args: argparse.Namespace) -> None:
    """Coerce invalid smoke/debug values before constructing GRPOConfig.

    TRL GRPO estimates an advantage from a group of completions per prompt, so
    ``num_generations`` must be at least 2. Keeping the guard here makes CLI
    invocations robust even when an old DataSphere YAML or user override passes 1.
    """
    if getattr(args, 'num_generations', 2) < 2:
        print(
            f"[train_vlm_grpo] num_generations={args.num_generations} is invalid for GRPO; forcing 2.",
            flush=True,
        )
        args.num_generations = 2
    if getattr(args, 'num_generations_eval', 2) < 2:
        print(
            f"[train_vlm_grpo] num_generations_eval={args.num_generations_eval} is invalid for GRPO; forcing 2.",
            flush=True,
        )
        args.num_generations_eval = 2


def _supports_kwargs(callable_obj: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter config kwargs for compatibility across TRL versions and lightweight tests."""
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}



def offline_pretrained_kwargs() -> Dict[str, Any]:
    """Force local cache usage after the job switches Hugging Face into offline mode."""
    if os.environ.get('HF_HUB_OFFLINE') == '1' or os.environ.get('TRANSFORMERS_OFFLINE') == '1':
        return {'local_files_only': True}
    return {}


def _flash_attn_available() -> bool:
    return importlib.util.find_spec('flash_attn') is not None


def resolve_attn_implementation(attn_impl: str) -> str:
    if attn_impl == 'auto':
        return 'flash_attention_2' if _flash_attn_available() else 'sdpa'
    if attn_impl == 'flash_attention_2' and not _flash_attn_available():
        print('[train_vlm_grpo] flash_attn is not installed; falling back to sdpa.', flush=True)
        return 'sdpa'
    return attn_impl


def load_processor(model_id: str, min_pixels: int | None, max_pixels: int | None, trust_remote_code: bool = False):
    kwargs: Dict[str, Any] = {'trust_remote_code': trust_remote_code, **offline_pretrained_kwargs()}
    if min_pixels is not None:
        kwargs['min_pixels'] = min_pixels
    if max_pixels is not None:
        kwargs['max_pixels'] = max_pixels
    try:
        return AutoProcessor.from_pretrained(model_id, **kwargs)
    except TypeError:
        kwargs.pop('min_pixels', None)
        kwargs.pop('max_pixels', None)
        return AutoProcessor.from_pretrained(model_id, **kwargs)



def disable_trl_model_card_creation(trainer: Any, reason: str) -> None:
    """Disable TRL's optional model-card side effect in managed jobs.

    TRL may create a README/model card during checkpoint saves. In DataSphere
    jobs the locale can be ASCII, while TRL/Hugging Face templates are UTF-8;
    letting this optional side effect run can abort an otherwise valid training
    step with UnicodeDecodeError. The pipeline creates its own UTF-8 metadata.
    """
    if os.environ.get('DISABLE_TRL_MODEL_CARD', '1').lower() in {'0', 'false', 'no', 'off'}:
        return

    def _skip_model_card(*args: Any, **kwargs: Any) -> None:
        if is_main_process():
            print(f'[train_vlm_grpo] skipping TRL model card creation: {reason}', flush=True)
        return None

    try:
        trainer.create_model_card = _skip_model_card  # type: ignore[attr-defined, method-assign]
    except Exception as exc:
        if is_main_process():
            print(f'[train_vlm_grpo] warning: could not disable TRL model card creation: {exc}', flush=True)

def get_local_rank() -> int:
    return int(os.environ.get('LOCAL_RANK', '0'))


def is_main_process() -> bool:
    return int(os.environ.get('RANK', '0')) == 0


def load_qwen_model(model_id: str, qlora: bool, bf16: bool, fp16: bool, attn_impl: str, trust_remote_code: bool = False):
    torch_dtype = torch.bfloat16 if bf16 else torch.float16 if fp16 else None
    quant_config = None
    device_map = None
    
    if qlora:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
        )
        device_map = {'': get_local_rank()} if torch.cuda.is_available() else None

    kwargs = {'attn_implementation': resolve_attn_implementation(attn_impl), 'trust_remote_code': trust_remote_code, **offline_pretrained_kwargs()}
    if torch_dtype is not None: kwargs['torch_dtype'] = torch_dtype
    if quant_config is not None: kwargs['quantization_config'] = quant_config
    if device_map is not None: kwargs['device_map'] = device_map

    if 'Qwen3-VL' in model_id and Qwen3VLForConditionalGeneration is not None:
        model_cls = Qwen3VLForConditionalGeneration
    elif ('Qwen2.5-VL' in model_id or 'Qwen2-VL' in model_id) and Qwen2_5_VLForConditionalGeneration is not None:
        model_cls = Qwen2_5_VLForConditionalGeneration
    else:
        model_cls = AutoModelForImageTextToText
    try:
        return model_cls.from_pretrained(model_id, **kwargs)
    except Exception as exc:
        if kwargs.get('attn_implementation') == 'flash_attention_2':
            print(f'[train_vlm_grpo] flash_attention_2 load failed ({exc!r}); retrying with sdpa.', flush=True)
            kwargs['attn_implementation'] = 'sdpa'
            return model_cls.from_pretrained(model_id, **kwargs)
        raise

def _is_url(value: str) -> bool:
    return value.startswith('http://') or value.startswith('https://')


def _resolve_image_ref(value, base_dir: Path):
    if value in (None, '', []):
        return None
    if isinstance(value, dict):
        value = value.get('path') or value.get('image') or value.get('url') or value.get('bytes')
    if not isinstance(value, str):
        return value
    value = value.strip()
    if not value:
        return None
    if _is_url(value):
        return value
    path = Path(value)
    if not path.is_absolute():
        # Prefer paths that are already valid from the current job working
        # directory (the repository root in our DataSphere wrappers). If not
        # present there, resolve relative to the JSONL/config file directory.
        if path.exists():
            path = path.resolve()
        else:
            path = (base_dir / path).resolve()
    return str(path.as_posix())



def _safe_text(value: Any) -> str:
    """Convert arbitrary JSON-like prompt content into chat-template-safe text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _text_block(value: Any) -> Dict[str, str]:
    return {'type': 'text', 'text': _safe_text(value)}


def _append_text_block(blocks: list[Dict[str, Any]], value: Any) -> None:
    text = _safe_text(value)
    if text.strip():
        blocks.append({'type': 'text', 'text': text})


def _append_image_block(blocks: list[Dict[str, Any]], images: list[Any], image_ref: Any) -> None:
    if image_ref and image_ref not in images:
        images.append(image_ref)
    blocks.append({'type': 'image'})


def _canonicalize_content_blocks(content: Any, base_dir: Path, images: list[Any]) -> list[Dict[str, Any]]:
    """Return only Qwen/TRL-safe VLM blocks.

    Qwen3-VL's chat template checks expressions such as ``'image' in
    content`` for each content item. Raw integers or arbitrary nested objects
    in exported prompt content can therefore crash Jinja with ``TypeError``.
    This function ensures that prompt content contains only explicit text/image
    blocks before TRL calls ``apply_chat_template``.
    """
    blocks: list[Dict[str, Any]] = []
    if content in (None, ''):
        return blocks
    if isinstance(content, str):
        _append_text_block(blocks, content)
        return blocks
    if isinstance(content, dict):
        content = [content]
    elif not isinstance(content, list):
        _append_text_block(blocks, content)
        return blocks

    for item in content:
        if item in (None, ''):
            continue
        if isinstance(item, str):
            _append_text_block(blocks, item)
            continue
        if not isinstance(item, dict):
            _append_text_block(blocks, item)
            continue

        block_type = str(item.get('type') or '').lower()
        image_candidate = item.get('image') or item.get('image_url') or item.get('path') or item.get('url')
        if block_type == 'image' or image_candidate:
            image_ref = _resolve_image_ref(image_candidate, base_dir) if image_candidate else None
            _append_image_block(blocks, images, image_ref)
            continue
        if block_type == 'video' or item.get('video'):
            _append_text_block(blocks, item)
            continue
        if 'text' in item:
            _append_text_block(blocks, item.get('text'))
            continue
        _append_text_block(blocks, item)
    return blocks


def _canonicalize_messages(messages: Any, base_dir: Path):
    canonical: list[Dict[str, Any]] = []
    images: list[Any] = []
    if not isinstance(messages, list):
        if messages not in (None, '', []):
            canonical.append({'role': 'user', 'content': [_text_block(messages)]})
        return canonical, images

    for msg in messages:
        if not isinstance(msg, dict):
            if msg not in (None, '', []):
                canonical.append({'role': 'user', 'content': [_text_block(msg)]})
            continue
        role = str(msg.get('role') or msg.get('from') or 'user')
        if role == 'human':
            role = 'user'
        elif role in {'gpt', 'bot', 'model'}:
            role = 'assistant'
        content = msg.get('content', msg.get('value'))
        blocks = _canonicalize_content_blocks(content, base_dir, images)
        if blocks:
            canonical.append({'role': role, 'content': blocks})
    return canonical, images


def _normalize_image_list(value, base_dir: Path):
    if value in (None, '', []):
        return []
    values = value if isinstance(value, list) else [value]
    out = []
    for item in values:
        resolved = _resolve_image_ref(item, base_dir)
        if resolved and resolved not in out:
            out.append(resolved)
    return out


def _flatten_single_text_messages(messages):
    """Compatibility helper for older unit tests and notebook snippets."""
    flattened = []
    for msg in messages:
        content = msg.get('content')
        if (
            isinstance(content, list)
            and len(content) == 1
            and isinstance(content[0], dict)
            and content[0].get('type') == 'text'
        ):
            flattened.append({'role': msg.get('role'), 'content': content[0].get('text', '')})
        else:
            flattened.append(msg)
    return flattened


def _messages_have_image_block(messages: list[Dict[str, Any]]) -> bool:
    for msg in messages:
        content = msg.get('content')
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'image':
                    return True
    return False

def _align_image_placeholders_to_images(messages: list[Dict[str, Any]], image_count: int) -> list[Dict[str, Any]]:
    """Ensure TRL receives exactly one image placeholder per top-level image."""
    image_count = max(0, int(image_count or 0))
    kept_images = 0
    aligned: list[Dict[str, Any]] = []
    for msg in messages:
        role = str(msg.get('role') or 'user')
        content = msg.get('content')
        if not isinstance(content, list):
            aligned.append({'role': role, 'content': content})
            continue
        new_content: list[Dict[str, Any]] = []
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'image':
                if kept_images < image_count:
                    new_content.append({'type': 'image'})
                    kept_images += 1
                continue
            new_content.append(block if isinstance(block, dict) else _text_block(block))
        if new_content:
            aligned.append({'role': role, 'content': new_content})

    missing = image_count - kept_images
    if missing > 0:
        placeholders = [{'type': 'image'} for _ in range(missing)]
        for msg in aligned:
            if msg.get('role') == 'user':
                content = msg.get('content')
                if isinstance(content, str):
                    msg['content'] = placeholders + [{'type': 'text', 'text': content}]
                elif isinstance(content, list):
                    msg['content'] = placeholders + content
                else:
                    msg['content'] = placeholders
                break
        else:
            aligned.insert(0, {'role': 'user', 'content': placeholders})
    return aligned


def _ensure_image_placeholder(messages: list[Dict[str, Any]], has_images: bool) -> list[Dict[str, Any]]:
    if not has_images or _messages_have_image_block(messages):
        return messages
    for msg in messages:
        if msg.get('role') == 'user':
            content = msg.setdefault('content', [])
            if isinstance(content, str):
                msg['content'] = [{'type': 'image'}, {'type': 'text', 'text': content}]
            elif isinstance(content, list):
                content.insert(0, {'type': 'image'})
            return messages
    messages.insert(0, {'role': 'user', 'content': [{'type': 'image'}]})
    return messages


def format_grpo_keys(example, base_dir: Path | None = None):
    """Backward-compatible normalizer used by regression tests.

    The training path uses ``make_grpo_formatter`` below and keeps all prompt
    contents in TRL multimodal block format.
    """
    base_dir = base_dir or Path('.')
    source_messages = example.get('prompt')
    if not source_messages and isinstance(example.get('prompt_chat'), dict):
        source_messages = example['prompt_chat'].get('messages')
    if not source_messages and example.get('prompt_messages'):
        source_messages = example.get('prompt_messages')
    prompt, embedded_images = _canonicalize_messages(source_messages, base_dir)
    images = []
    images.extend(_normalize_image_list(example.get('images'), base_dir))
    images.extend(_normalize_image_list(example.get('image'), base_dir))
    for image_ref in embedded_images:
        if image_ref not in images:
            images.append(image_ref)
    if prompt:
        prompt = _ensure_image_placeholder(prompt, bool(images))
        prompt = _align_image_placeholders_to_images(prompt, len(images))
        example['prompt'] = _flatten_single_text_messages(prompt)
    example['images'] = images
    return example


def make_grpo_formatter(base_dir: Path):
    def format_grpo(example):
        source_messages = example.get('prompt')
        if not source_messages and isinstance(example.get('prompt_chat'), dict):
            source_messages = example['prompt_chat'].get('messages')
        if not source_messages and example.get('prompt_messages'):
            source_messages = example.get('prompt_messages')
        prompt, embedded_images = _canonicalize_messages(source_messages, base_dir)
        images = []
        images.extend(_normalize_image_list(example.get('images'), base_dir))
        images.extend(_normalize_image_list(example.get('image'), base_dir))
        for image_ref in embedded_images:
            if image_ref not in images:
                images.append(image_ref)
        if prompt:
            prompt = _ensure_image_placeholder(prompt, bool(images))
            example['prompt'] = _align_image_placeholders_to_images(prompt, len(images))
        # Keep a stable list column; VLM GRPO filters empty-image rows before
        # handing the dataset to TRL's Qwen3-VL generation path.
        example['images'] = images
        return example
    return format_grpo

def _value_has_image(value) -> bool:
    if value in (None, ''):
        return False
    if isinstance(value, list):
        return any(item not in (None, '', []) for item in value)
    return True


def _row_has_nonempty_images(example: Dict[str, Any]) -> bool:
    return _value_has_image(example.get('images')) or _value_has_image(example.get('image'))


def _dataset_has_any_images(ds) -> bool:
    if not any(col in ds.column_names for col in ('images', 'image')):
        return False
    sample = ds[: len(ds)]
    return any(
        _value_has_image(value)
        for col in ('images', 'image')
        for value in sample.get(col, [])
    )


def _filter_text_only_vlm_grpo_rows(ds, split_name: str, *, required: bool = False):
    """Drop rows that would make TRL pass images=None to Qwen3-VL.

    TRL's multimodal GRPO path batches prompts through the Qwen3-VL processor
    with an ``images=...`` argument for the whole generation batch. With mixed
    VLM/text rows, current Transformers raises ``TypeError`` when a row has no
    image payload. SFT can handle such mixed rows, but GRPO generation cannot;
    for VLM GRPO we therefore keep only rows with at least one resolved image.
    """
    before = len(ds)
    if before == 0 or not any(col in ds.column_names for col in ('images', 'image')):
        if required:
            raise ValueError(f'No image column is available for forced VLM GRPO {split_name} split.')
        return ds

    if not _dataset_has_any_images(ds):
        if required:
            raise ValueError(f'Forced VLM GRPO {split_name} split has no rows with images.')
        return ds

    filtered = ds.filter(_row_has_nonempty_images)
    dropped = before - len(filtered)
    if dropped:
        print(
            f'[train_vlm_grpo] dropped {dropped}/{before} {split_name} rows without images; '
            'current TRL multimodal GRPO calls the Qwen3-VL processor with images=None for text-only VLM rows.',
            flush=True,
        )
    return filtered



_IMAGE_SELECT_TOKEN_RE = re.compile(r"[A-Za-z0-9А-Яа-яёЁ]{3,}", re.UNICODE)


def _stable_text_for_image_selection(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)


def _image_select_tokens(value: Any) -> set[str]:
    return {m.group(0).lower() for m in _IMAGE_SELECT_TOKEN_RE.finditer(_stable_text_for_image_selection(value))}


def _score_training_image(image_ref: Any, row: Dict[str, Any], raw_index: int) -> float:
    """Deterministic evidence-aware score for the training-time memory projection.

    Full raw JSONL artifacts keep all image references.  When a g2.2-safe cap is
    needed, this prefers images whose path/basename overlaps with evidence,
    prompt, claim, metadata, figure/table/page hints, then preserves original
    order as a stable tie-breaker.
    """
    image_text = _stable_text_for_image_selection(image_ref).lower()
    basename = Path(str(image_ref)).name
    image_tokens = _image_select_tokens(basename)
    evidence_tokens = _image_select_tokens({
        "evidence": row.get("evidence"),
        "evidence_text": row.get("evidence_text"),
        "reference_assertions_json": row.get("reference_assertions_json"),
        "reference_temporal_json": row.get("reference_temporal_json"),
        "metadata": row.get("metadata"),
    })
    row_tokens = _image_select_tokens({
        "prompt": row.get("prompt"),
        "messages": row.get("messages"),
        "claim": row.get("claim"),
        "question": row.get("question"),
        "chosen": row.get("chosen"),
        "rejected": row.get("rejected"),
        "task_family": row.get("task_family"),
    })
    score = 0.0
    score += 4.0 * len(image_tokens & evidence_tokens)
    score += 1.0 * len(image_tokens & row_tokens)
    if "figure" in image_text or "fig" in image_text:
        score += 1.0
    if "table" in image_text:
        score += 1.0
    if "page" in image_text or "p_" in image_text or "p-" in image_text:
        score += 0.25
    return score - raw_index * 1e-6


def _select_training_images_for_memory(row: Dict[str, Any], images: list[Any], max_images_per_example: int) -> list[Any]:
    if max_images_per_example <= 0 or len(images) <= max_images_per_example:
        return list(images)
    scored = [(_score_training_image(image, row, idx), idx, image) for idx, image in enumerate(images)]
    chosen = sorted(scored, key=lambda item: (-item[0], item[1]))[:max_images_per_example]
    return [image for _score, _idx, image in sorted(chosen, key=lambda item: item[1])]

def cap_grpo_images_for_memory(ds, max_images_per_example: int, split_name: str):
    if max_images_per_example is None or int(max_images_per_example) <= 0:
        return ds, {"enabled": False, "max_images_per_example": int(max_images_per_example or 0), "truncated_rows": 0, "dropped_image_refs": 0, "selection_policy": "evidence_aware_top_k_then_original_order"}
    max_images_per_example = int(max_images_per_example)
    stats = {"enabled": True, "max_images_per_example": max_images_per_example, "truncated_rows": 0, "dropped_image_refs": 0, "selection_policy": "evidence_aware_top_k_then_original_order"}
    for row in ds:
        images = []
        images.extend(_normalize_image_list(row.get('images'), Path('.')))
        images.extend(_normalize_image_list(row.get('image'), Path('.')))
        # de-duplicate while preserving order
        deduped = []
        for image in images:
            if image not in deduped:
                deduped.append(image)
        extra = max(0, len(deduped) - max_images_per_example)
        if extra:
            stats["truncated_rows"] += 1
            stats["dropped_image_refs"] += extra

    if stats["truncated_rows"] == 0:
        return ds, stats

    def _cap(example: Dict[str, Any]) -> Dict[str, Any]:
        example = dict(example)
        images = []
        images.extend(_normalize_image_list(example.get('images'), Path('.')))
        images.extend(_normalize_image_list(example.get('image'), Path('.')))
        deduped = []
        for image in images:
            if image not in deduped:
                deduped.append(image)
        kept = _select_training_images_for_memory(example, deduped, max_images_per_example)
        example['images'] = kept
        if kept:
            example['image'] = kept[0]
        elif 'image' in example:
            example['image'] = None
        if isinstance(example.get('prompt'), list):
            example['prompt'] = _align_image_placeholders_to_images(example['prompt'], len(kept))
        return example

    capped = ds.map(_cap, desc=f'Capping {split_name} GRPO rows to {max_images_per_example} images for GPU memory')
    print(
        f'[train_vlm_grpo] training memory guard capped {stats["truncated_rows"]}/{len(ds)} '
        f'{split_name} rows to at most {max_images_per_example} images '
        f'({stats["dropped_image_refs"]} image refs omitted from training projection only).',
        flush=True,
    )
    return capped, stats

def _cast_images_column(ds, image_column: str):
    if image_column == 'images':
        features = ds.features.copy()
        features['images'] = Sequence(HFImage())
        return ds.cast(features)
    return ds.cast_column(image_column, HFImage())


def _ensure_trl_images_column(ds, source_column: str | None = None):
    """Ensure the VLM dataset exposes the exact top-level `images` column TRL reads."""
    if 'images' in ds.column_names:
        return ds
    if source_column == 'image' and 'image' in ds.column_names:
        return ds.map(lambda example: {'images': _normalize_image_list(example.get('image'), Path('.'))})
    return ds.add_column('images', [[] for _ in range(len(ds))])


def maybe_prepare_dataset(ds, image_column: str, requested_mode: str):
    candidate_columns = []
    if image_column:
        candidate_columns.append(image_column)
    candidate_columns.extend(['images', 'image'])
    detected_column = next((col for col in candidate_columns if col in ds.column_names), None)

    if requested_mode == 'text':
        image_columns = [col for col in ['images', 'image'] if col in ds.column_names]
        return ds.remove_columns(image_columns) if image_columns else ds, 'text'

    if detected_column is None:
        if requested_mode == 'vlm':
            ds = _ensure_trl_images_column(ds)
            return _cast_images_column(ds, 'images'), 'vlm'
        return ds, 'text'

    sample = ds[: min(len(ds), 256)]
    non_null = sum(1 for x in sample.get(detected_column, []) if _value_has_image(x))

    if requested_mode == 'vlm':
        ds = _ensure_trl_images_column(ds, detected_column)
        image_columns_to_remove = [col for col in ['image'] if col in ds.column_names]
        if image_columns_to_remove:
            ds = ds.remove_columns(image_columns_to_remove)
        return _cast_images_column(ds, 'images'), 'vlm'

    image_columns = [col for col in ['images', 'image'] if col in ds.column_names]
    if non_null == 0:
        return ds.remove_columns(image_columns), 'text'

    if detected_column != 'images':
        ds = _ensure_trl_images_column(ds, detected_column)
        detected_column = 'images'
    image_columns_to_remove = [col for col in ['image'] if col in ds.column_names]
    if image_columns_to_remove:
        ds = ds.remove_columns(image_columns_to_remove)

    # TRL expects VLM datasets to have the top-level `images` column in this version.
    # Mixed text-only + image rows are valid; keep empty lists for text-only rows.
    return _cast_images_column(ds, detected_column), 'vlm'


# 4. CLI

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-id', required=True)
    ap.add_argument('--train-file', type=Path, required=True)
    ap.add_argument('--eval-file', type=Path, default=None)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--learning-rate', type=float, default=1e-5)
    ap.add_argument('--num-train-epochs', type=float, default=1.0)
    ap.add_argument('--max-steps', type=int, default=-1)
    ap.add_argument('--per-device-train-batch-size', type=int, default=1)
    ap.add_argument('--per-device-eval-batch-size', type=int, default=1)
    ap.add_argument('--gradient-accumulation-steps', type=int, default=8)
    ap.add_argument('--num-generations', type=int, default=2)
    ap.add_argument('--num-generations-eval', type=int, default=2)
    ap.add_argument('--max-completion-length', type=int, default=512)
    ap.add_argument('--num-iterations', type=int, default=1, help='GRPO inner optimization iterations per generation batch when supported by installed TRL.')
    ap.add_argument('--logging-steps', type=int, default=5)
    ap.add_argument('--save-steps', type=int, default=40)
    ap.add_argument('--eval-steps', type=int, default=40)
    ap.add_argument('--save-total-limit', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--lora-r', type=int, default=32)
    ap.add_argument('--lora-alpha', type=int, default=64)
    ap.add_argument('--lora-dropout', type=float, default=0.05)
    ap.add_argument('--lora-target-modules', nargs='+', default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    ap.add_argument('--use-lora', action='store_true')
    ap.add_argument('--qlora', action='store_true')
    ap.add_argument('--sft-adapter-path', type=Path, default=None)
    ap.add_argument('--bf16', action='store_true')
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--tf32', action='store_true')
    ap.add_argument('--gradient-checkpointing', action='store_true')
    ap.add_argument('--attn-implementation', default='auto', choices=['auto', 'sdpa', 'flash_attention_2', 'eager'])
    ap.add_argument('--trust-remote-code', action='store_true')
    ap.add_argument('--train-mode', choices=['auto', 'text', 'vlm'], default='auto')
    ap.add_argument('--image-column', default='images')
    ap.add_argument('--min-pixels', type=int, default=None)
    ap.add_argument('--max-pixels', type=int, default=None)
    ap.add_argument(
        '--max-images-per-example',
        type=int,
        default=0,
        help='Training-time VLM memory guard: keep at most this many images per prompt. 0 disables the projection.',
    )
    ap.add_argument('--warmup-ratio', type=float, default=0.08)
    ap.add_argument('--beta', type=float, default=0.02, help='KL coefficient for GRPO. Set 0 only for memory-constrained smoke runs.')
    ap.add_argument('--optim', default='adamw_torch_fused')
    ap.add_argument('--lr-scheduler-type', default='cosine')
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--max-grad-norm', type=float, default=0.3)
    ap.add_argument('--dataloader-num-workers', type=int, default=2)
    ap.add_argument(
        '--ddp-find-unused-parameters',
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            'Pass find_unused_parameters to DistributedDataParallel. '
            'Default: False. Enable only when a DDP run fails with unused-parameter '
            'errors; this flag adds autograd graph traversal overhead.'
        ),
    )
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--top-p', type=float, default=0.95)
    ap.add_argument('--top-k', type=int, default=0)
    ap.add_argument('--epsilon', type=float, default=0.2, help='Lower PPO/GRPO clipping epsilon when supported by installed TRL.')
    ap.add_argument('--epsilon-high', type=float, default=0.28, help='Upper clipping epsilon for asymmetric clipping when supported by installed TRL.')
    ap.add_argument('--top-entropy-quantile', type=float, default=0.2, help='Only update on high-entropy tokens/positions when supported by installed TRL.')
    ap.add_argument('--mask-truncated-completions', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--importance-sampling-level', default='sequence', choices=['token', 'sequence'])
    ap.add_argument('--multi-objective-aggregation', default='normalize_then_sum', choices=['sum_then_normalize', 'normalize_then_sum'])
    ap.add_argument('--reward-normalization-clip', type=float, default=2.5, help='Clip value for task-aware robust reward normalization.')
    ap.add_argument('--reward-normalization-temperature', type=float, default=1.5, help='Tanh temperature for task-aware robust reward normalization.')
    ap.add_argument('--reward-weights', type=float, nargs='+', default=[0.0, 1.0, 0.8, 1.2, 0.5, 1.5], help="Weights for: label, schema, temporal, graph, evidence, verdict")
    ap.add_argument('--log-completions', action='store_true')
    ap.add_argument('--min-reward-std', type=float, default=0.02, help='Warn/fail if post-run reward std is below this threshold for active components.')
    ap.add_argument('--max-zero-std-frac', type=float, default=0.7, help='Warn/fail if grouped reward zero-std fraction exceeds this threshold.')
    ap.add_argument('--fail-on-weak-reward', action=argparse.BooleanOptionalAction, default=False, help='Exit non-zero after training if reward trace audit indicates degenerate GRPO signal.')
    ap.add_argument('--dry-run', action='store_true')
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    enforce_minimum_grpo_generations(args)
    if args.tf32:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
    if len(args.reward_weights) != 6:
        raise ValueError('--reward-weights must contain exactly 6 values: label, schema, temporal, graph, evidence, verdict')
    
    REWARD_LOGGER.path = args.output_dir / "grpo_reward_trace.jsonl"

    data_files = {'train': str(args.train_file)}
    if args.eval_file: 
        data_files['eval'] = str(args.eval_file)
        
    ds = load_dataset('json', data_files=data_files)
    train_ds = ds['train']
    eval_ds = ds.get('eval')

    format_grpo = make_grpo_formatter(args.train_file.parent)
    train_ds = train_ds.map(format_grpo)
    train_image_cap_stats = {"enabled": False, "max_images_per_example": int(args.max_images_per_example or 0), "truncated_rows": 0, "dropped_image_refs": 0, "selection_policy": "evidence_aware_top_k_then_original_order"}
    eval_image_cap_stats = {"enabled": False, "max_images_per_example": int(args.max_images_per_example or 0), "truncated_rows": 0, "dropped_image_refs": 0, "selection_policy": "evidence_aware_top_k_then_original_order"}
    if args.max_images_per_example and int(args.max_images_per_example) > 0:
        train_ds, train_image_cap_stats = cap_grpo_images_for_memory(train_ds, args.max_images_per_example, 'train')
    if eval_ds is not None:
        eval_base_dir = args.eval_file.parent if args.eval_file else args.train_file.parent
        eval_ds = eval_ds.map(make_grpo_formatter(eval_base_dir))
        if args.max_images_per_example and int(args.max_images_per_example) > 0:
            eval_ds, eval_image_cap_stats = cap_grpo_images_for_memory(eval_ds, args.max_images_per_example, 'eval')

    if args.train_mode == 'vlm' or (args.train_mode == 'auto' and _dataset_has_any_images(train_ds)):
        train_ds = _filter_text_only_vlm_grpo_rows(train_ds, 'train', required=args.train_mode == 'vlm')
        if eval_ds is not None:
            eval_before = len(eval_ds)
            eval_ds = _filter_text_only_vlm_grpo_rows(eval_ds, 'eval', required=False)
            if eval_before and len(eval_ds) == 0:
                print(
                    '[train_vlm_grpo] disabled eval split because no eval rows with images remain for VLM GRPO.',
                    flush=True,
                )
                eval_ds = None

    train_ds, actual_mode = maybe_prepare_dataset(train_ds, args.image_column, args.train_mode)
    if eval_ds is not None: 
        eval_ds, _ = maybe_prepare_dataset(eval_ds, args.image_column, actual_mode)

    processor = load_processor(args.model_id, args.min_pixels, args.max_pixels, args.trust_remote_code)
    tokenizer = getattr(processor, 'tokenizer', None) or AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code, **offline_pretrained_kwargs())
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(processor, "tokenizer") and getattr(processor.tokenizer, "pad_token", None) is None:
        processor.tokenizer.pad_token = tokenizer.pad_token

    model = load_qwen_model(args.model_id, args.qlora, args.bf16, args.fp16, args.attn_implementation, args.trust_remote_code)
    if args.gradient_checkpointing and hasattr(model, 'config'):
        try:
            model.config.use_cache = False
        except Exception:
            pass
    if args.qlora: 
        model = prepare_model_for_kbit_training(model)
        
    if args.sft_adapter_path:
        if PeftModel is None:
            raise RuntimeError('peft.PeftModel is required for --sft-adapter-path; install a full peft package.')
        model = PeftModel.from_pretrained(model, str(args.sft_adapter_path), is_trainable=True, **offline_pretrained_kwargs())
        if args.gradient_checkpointing and hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
    elif args.use_lora or args.qlora:
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r, 
                lora_alpha=args.lora_alpha, 
                lora_dropout=args.lora_dropout, 
                bias='none',
                target_modules=args.lora_target_modules, 
                task_type='CAUSAL_LM',
            ),
        )
        if args.gradient_checkpointing and hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()

    grpo_kwargs = dict(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        beta=args.beta,
        optim=args.optim,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        num_generations=args.num_generations,
        num_generations_eval=args.num_generations_eval,
        max_completion_length=args.max_completion_length,
        num_iterations=args.num_iterations,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        epsilon=args.epsilon,
        epsilon_high=args.epsilon_high,
        top_entropy_quantile=args.top_entropy_quantile,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        tf32=args.tf32,
        remove_unused_columns=False,
        reward_weights=args.reward_weights,
        mask_truncated_completions=args.mask_truncated_completions,
        importance_sampling_level=args.importance_sampling_level,
        multi_objective_aggregation=args.multi_objective_aggregation,
        log_completions=args.log_completions,
        eval_strategy="steps" if eval_ds is not None else "no",
        save_strategy="steps",
        logging_strategy="steps",
        report_to=[],
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=resolve_ddp_find_unused_parameters(args, actual_mode),
    )
    grpo_args = GRPOConfig(**_supports_kwargs(GRPOConfig, grpo_kwargs))

    run_config = vars(args).copy()
    run_config['resolved_mode'] = actual_mode
    run_config['train_examples'] = len(train_ds)
    run_config['eval_examples'] = len(eval_ds) if eval_ds is not None else 0
    run_config['ddp_find_unused_parameters_resolved'] = resolve_ddp_find_unused_parameters(args, actual_mode)
    run_config['kl_beta'] = args.beta
    run_config['train_image_cap_stats'] = train_image_cap_stats
    run_config['eval_image_cap_stats'] = eval_image_cap_stats
    
    (args.output_dir / 'planned_run_config.json').write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2, default=str), 
        encoding='utf-8'
    )

    if args.dry_run:
        print(json.dumps(run_config, ensure_ascii=False, indent=2, default=str))
        return

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            reward_label_exact_match,
            reward_schema_validity, 
            reward_temporal_consistency, 
            reward_graph_consistency, 
            reward_evidence_presence, 
            reward_expert_override_match
        ],
        args=grpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor if actual_mode == 'vlm' else tokenizer,
    )
    disable_trl_model_card_creation(
        trainer,
        'DataSphere may expose an ASCII locale; TRL/huggingface_hub model-card templates are UTF-8.',
    )

    train_result = trainer.train()
    trainer.save_state()
    trainer.save_metrics('train', {**train_result.metrics, 'train_examples': len(train_ds)})
    
    if eval_ds is not None:
        eval_metrics = trainer.evaluate()
        eval_metrics['eval_examples'] = len(eval_ds)
        trainer.save_metrics('eval', eval_metrics)
    
    trainer.save_model(args.output_dir)

    if is_main_process():
        reward_audit = audit_reward_trace_file(
            args.output_dir / 'grpo_reward_trace.jsonl',
            min_reward_std=args.min_reward_std,
            max_zero_std_frac=args.max_zero_std_frac,
        )
        (args.output_dir / 'post_run_reward_audit.json').write_text(
            json.dumps(reward_audit, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        if reward_audit.get('weak_reward'):
            print('[train_vlm_grpo] weak reward signal detected; see post_run_reward_audit.json', flush=True)
            if args.fail_on_weak_reward:
                raise SystemExit(2)
        processor.save_pretrained(args.output_dir)

if __name__ == '__main__':
    main()