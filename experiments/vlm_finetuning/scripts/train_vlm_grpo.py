#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Image as HFImage, Sequence
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    set_seed,
)
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


# 2. Reward functions for RL

def reward_schema_validity(prompts, completions, sample_id=None, domain=None, task_family=None, trainer_state=None, **kwargs):
    task_family = task_family or [""] * len(completions)
    sample_id = sample_id or [""] * len(completions)
    domain = domain or [""] * len(completions)

    rewards = []
    for completion, tf in zip(completions, task_family):
        obj = parse_prediction_object(completion)
        reward = -1.0
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
    REWARD_LOGGER.append("schema_validity", trainer_state, sample_id, domain, task_family, rewards)
    return rewards

def reward_graph_consistency(prompts, completions, reference_assertions_json=None, sample_id=None, domain=None, task_family=None, trainer_state=None, **kwargs):
    task_family = task_family or [""] * len(completions)
    reference_assertions_json = reference_assertions_json or [None] * len(completions)
    sample_id = sample_id or [""] * len(completions)
    domain = domain or [""] * len(completions)

    rewards = []
    for completion, ref_json, tf in zip(completions, reference_assertions_json, task_family):
        if tf not in {"trajectory_reasoning_rl", "assertion_review_rl"}:
            rewards.append(0.0)
            continue
        
        pred_obj = parse_prediction_object(completion)
        pred_keys = {triple_key(x) for x in flatten_assertions(pred_obj)}
        
        try:
            ref_assertions = json.loads(ref_json) if isinstance(ref_json, str) else (ref_json or [])
        except Exception:
            ref_assertions = []
        ref_keys = {triple_key(x) for x in ref_assertions if isinstance(x, dict)}
        
        if not pred_keys and not ref_keys:
            rewards.append(0.0)
            continue
            
        tp = len(pred_keys & ref_keys)
        fp, fn = len(pred_keys - ref_keys), len(ref_keys - pred_keys)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rewards.append(2.0 * f1 - 1.0)
        
    REWARD_LOGGER.append("graph_consistency", trainer_state, sample_id, domain, task_family, rewards)
    return rewards

def reward_temporal_consistency(prompts, completions, reference_temporal_json=None, sample_id=None, domain=None, task_family=None, trainer_state=None, **kwargs):
    task_family = task_family or [""] * len(completions)
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
        
    REWARD_LOGGER.append("temporal_consistency", trainer_state, sample_id, domain, task_family, rewards)
    return rewards

def reward_evidence_presence(prompts, completions, evidence_text=None, sample_id=None, domain=None, task_family=None, trainer_state=None, **kwargs):
    task_family = task_family or [""] * len(completions)
    evidence_text = evidence_text or [None] * len(completions)
    sample_id = sample_id or [""] * len(completions)
    domain = domain or [""] * len(completions)

    rewards = []
    for completion, ev_text, tf in zip(completions, evidence_text, task_family):
        pred_obj = parse_prediction_object(completion)
        if not isinstance(pred_obj, dict):
            rewards.append(-0.5)
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
            
    REWARD_LOGGER.append("evidence_presence", trainer_state, sample_id, domain, task_family, rewards)
    return rewards

def reward_expert_override_match(prompts, completions, expected_verdict=None, sample_id=None, domain=None, task_family=None, trainer_state=None, **kwargs):
    task_family = task_family or [""] * len(completions)
    expected_verdict = expected_verdict or [None] * len(completions)
    sample_id = sample_id or [""] * len(completions)
    domain = domain or [""] * len(completions)

    rewards = []
    for completion, verdict, tf in zip(completions, expected_verdict, task_family):
        if tf not in {"assertion_review_rl", "mm_review_rl", "temporal_fix_rl"}:
            rewards.append(0.0)
            continue
            
        pred_obj = parse_prediction_object(completion)
        if not isinstance(pred_obj, dict):
            rewards.append(-1.0)
            continue
            
        if tf == "mm_review_rl":
            pred_verdict = pred_obj.get("mm_verdict") or pred_obj.get("verdict")
        else:
            pred_verdict = pred_obj.get("verdict")
            
        if not verdict:
            rewards.append(0.0)
        else:
            rewards.append(1.0 if norm_text(pred_verdict) == norm_text(verdict) else -1.0)
            
    REWARD_LOGGER.append("expert_override_match", trainer_state, sample_id, domain, task_family, rewards)
    return rewards


# 3. Dataset and model preparation

def get_local_rank() -> int:
    return int(os.environ.get('LOCAL_RANK', '0'))

def load_qwen_model(model_id: str, qlora: bool, bf16: bool, fp16: bool, attn_impl: str):
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

    kwargs = {'attn_implementation': attn_impl}
    if torch_dtype is not None: kwargs['torch_dtype'] = torch_dtype
    if quant_config is not None: kwargs['quantization_config'] = quant_config
    if device_map is not None: kwargs['device_map'] = device_map

    if 'Qwen3-VL' in model_id: 
        return Qwen3VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    elif 'Qwen2.5-VL' in model_id or 'Qwen2-VL' in model_id: 
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    else:
        raise ValueError(f'Unsupported model: {model_id}')

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
        path = (base_dir / path).resolve()
    return str(path.as_posix())


def _canonicalize_messages(messages, base_dir: Path):
    canonical = []
    images = []
    if not isinstance(messages, list):
        return canonical, images
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get('role') or 'user')
        content = msg.get('content')
        if isinstance(content, str):
            canonical.append({'role': role, 'content': [{'type': 'text', 'text': content}]})
            continue
        blocks = []
        if isinstance(content, list):
            for block in content:
                if isinstance(block, str):
                    if block.strip():
                        blocks.append({'type': 'text', 'text': block})
                    continue
                if not isinstance(block, dict):
                    continue
                block_type = str(block.get('type') or 'text').lower()
                if block_type == 'image':
                    image_ref = _resolve_image_ref(block.get('image') or block.get('path') or block.get('url'), base_dir)
                    if image_ref and image_ref not in images:
                        images.append(image_ref)
                    blocks.append({'type': 'image'})
                elif block_type == 'text':
                    text = block.get('text')
                    if text is not None and str(text):
                        blocks.append({'type': 'text', 'text': str(text)})
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
    if prompt:
        example['prompt'] = _flatten_single_text_messages(prompt)
    images = []
    images.extend(_normalize_image_list(example.get('images'), base_dir))
    images.extend(_normalize_image_list(example.get('image'), base_dir))
    for image_ref in embedded_images:
        if image_ref not in images:
            images.append(image_ref)
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
        if prompt:
            example['prompt'] = prompt
        images = []
        images.extend(_normalize_image_list(example.get('images'), base_dir))
        images.extend(_normalize_image_list(example.get('image'), base_dir))
        for image_ref in embedded_images:
            if image_ref not in images:
                images.append(image_ref)
        # Keep a stable list column: VLM rows contain resolved image paths,
        # text-only rows contain [] and can coexist in recent TRL/Transformers.
        example['images'] = images
        return example
    return format_grpo


def _value_has_image(value) -> bool:
    if value in (None, ''):
        return False
    if isinstance(value, list):
        return any(item not in (None, '', []) for item in value)
    return True


def _cast_images_column(ds, image_column: str):
    if image_column == 'images':
        features = ds.features.copy()
        features['images'] = Sequence(HFImage())
        return ds.cast(features)
    return ds.cast_column(image_column, HFImage())


def maybe_prepare_dataset(ds, image_column: str, requested_mode: str):
    candidate_columns = []
    if image_column:
        candidate_columns.append(image_column)
    candidate_columns.extend(['images', 'image'])
    detected_column = next((col for col in candidate_columns if col in ds.column_names), None)
    if detected_column is None:
        return ds, 'text'

    sample = ds[: min(len(ds), 256)]
    non_null = sum(1 for x in sample.get(detected_column, []) if _value_has_image(x))
    image_columns_to_remove = [col for col in ['images', 'image'] if col in ds.column_names]

    if requested_mode == 'text' or non_null == 0:
        return ds.remove_columns(image_columns_to_remove), 'text'

    # TRL expects VLM datasets to have a top-level `image` or `images` column.
    # Mixed text-only + image rows are valid on recent transformers/TRL; keep
    # empty lists for text-only rows instead of silently falling back to text.
    return _cast_images_column(ds, detected_column), 'vlm'


# 4. CLI

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-id', required=True)
    ap.add_argument('--train-file', type=Path, required=True)
    ap.add_argument('--eval-file', type=Path, default=None)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--learning-rate', type=float, default=2e-5)
    ap.add_argument('--num-train-epochs', type=float, default=1.0)
    ap.add_argument('--max-steps', type=int, default=-1)
    ap.add_argument('--per-device-train-batch-size', type=int, default=1)
    ap.add_argument('--per-device-eval-batch-size', type=int, default=1)
    ap.add_argument('--gradient-accumulation-steps', type=int, default=8)
    ap.add_argument('--num-generations', type=int, default=4)
    ap.add_argument('--max-completion-length', type=int, default=512)
    ap.add_argument('--logging-steps', type=int, default=10)
    ap.add_argument('--save-steps', type=int, default=100)
    ap.add_argument('--eval-steps', type=int, default=100)
    ap.add_argument('--save-total-limit', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--lora-r', type=int, default=16)
    ap.add_argument('--lora-alpha', type=int, default=32)
    ap.add_argument('--lora-dropout', type=float, default=0.05)
    ap.add_argument('--lora-target-modules', nargs='+', default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    ap.add_argument('--use-lora', action='store_true')
    ap.add_argument('--qlora', action='store_true')
    ap.add_argument('--bf16', action='store_true')
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--gradient-checkpointing', action='store_true')
    ap.add_argument('--attn-implementation', default='sdpa')
    ap.add_argument('--train-mode', choices=['auto', 'text', 'vlm'], default='auto')
    ap.add_argument('--image-column', default='images')
    ap.add_argument('--reward-weights', type=float, nargs=5, default=[1.0, 1.0, 1.0, 0.75, 1.0], help="Weights for: schema, temporal, graph, evidence, verdict")
    ap.add_argument('--dry-run', action='store_true')
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    REWARD_LOGGER.path = args.output_dir / "grpo_reward_trace.jsonl"

    data_files = {'train': str(args.train_file)}
    if args.eval_file: 
        data_files['eval'] = str(args.eval_file)
        
    ds = load_dataset('json', data_files=data_files)
    train_ds = ds['train']
    eval_ds = ds.get('eval')

    format_grpo = make_grpo_formatter(args.train_file.parent)
    train_ds = train_ds.map(format_grpo)
    if eval_ds is not None:
        eval_base_dir = args.eval_file.parent if args.eval_file else args.train_file.parent
        eval_ds = eval_ds.map(make_grpo_formatter(eval_base_dir))

    train_ds, actual_mode = maybe_prepare_dataset(train_ds, args.image_column, args.train_mode)
    if eval_ds is not None: 
        eval_ds, _ = maybe_prepare_dataset(eval_ds, args.image_column, actual_mode)

    processor = AutoProcessor.from_pretrained(args.model_id)
    tokenizer = getattr(processor, 'tokenizer', None) or AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(processor, "tokenizer") and getattr(processor.tokenizer, "pad_token", None) is None:
        processor.tokenizer.pad_token = tokenizer.pad_token

    model = load_qwen_model(args.model_id, args.qlora, args.bf16, args.fp16, args.attn_implementation)
    if args.qlora: 
        model = prepare_model_for_kbit_training(model)
        
    if args.use_lora or args.qlora:
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

    grpo_args = GRPOConfig(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        # max_prompt_length=None if actual_mode == "vlm" else 4096,
        max_completion_length=args.max_completion_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        # reward_weights=args.reward_weights,
        log_completions=True,
        eval_strategy="steps" if eval_ds is not None else "no",
        save_strategy="steps",
        logging_strategy="steps",
        report_to=[],
    )

    run_config = vars(args).copy()
    run_config['resolved_mode'] = actual_mode
    run_config['train_examples'] = len(train_ds)
    run_config['eval_examples'] = len(eval_ds) if eval_ds is not None else 0
    
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

    train_result = trainer.train()
    trainer.save_state()
    trainer.save_metrics('train', {**train_result.metrics, 'train_examples': len(train_ds)})
    
    if eval_ds is not None:
        eval_metrics = trainer.evaluate()
        eval_metrics['eval_examples'] = len(eval_ds)
        trainer.save_metrics('eval', eval_metrics)
    
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

if __name__ == '__main__':
    main()