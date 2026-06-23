#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from datasets import Dataset, Image as HFImage, Sequence
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    set_seed,
)


def install_torch_fsdp_module_import_compat() -> bool:
    """Make newer TRL DPO imports work with torch builds lacking FSDPModule.

    Recent TRL versions import ``FSDPModule`` from ``torch.distributed.fsdp``
    even when the current job uses DDP rather than FSDP. Some DataSphere
    torch builds expose only the legacy ``FullyShardedDataParallel`` symbol,
    which makes ``from trl import DPOTrainer`` fail before training starts.
    Installing this narrow alias is safe for our DDP/LoRA path: it is only an
    import-time compatibility shim for TRL's optional FSDP helpers.
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
from trl import DPOConfig, DPOTrainer


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-id', required=True)
    ap.add_argument('--sft-adapter-path', type=Path, default=None)
    ap.add_argument('--train-file', type=Path, required=True)
    ap.add_argument('--eval-file', type=Path, default=None)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--report-to', default='none')
    ap.add_argument('--learning-rate', type=float, default=5e-6)
    ap.add_argument('--num-train-epochs', type=float, default=1.0)
    ap.add_argument('--max-steps', type=int, default=-1)
    ap.add_argument('--per-device-train-batch-size', type=int, default=1)
    ap.add_argument('--per-device-eval-batch-size', type=int, default=1)
    ap.add_argument('--gradient-accumulation-steps', type=int, default=8)
    ap.add_argument('--warmup-ratio', type=float, default=0.03)
    ap.add_argument('--logging-steps', type=int, default=10)
    ap.add_argument('--save-steps', type=int, default=100)
    ap.add_argument('--eval-steps', type=int, default=100)
    ap.add_argument('--save-total-limit', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--beta', type=float, default=0.06)
    ap.add_argument('--loss-type', nargs='+', default=['robust'], help='One or more DPO loss types, e.g. robust sft for mixed preference+SFT training.')
    ap.add_argument('--loss-weights', type=float, nargs='+', default=None, help='Weights for multi-loss DPO; length must match --loss-type when provided.')
    ap.add_argument('--label-smoothing', type=float, default=0.05, help='DPO noisy-preference smoothing for robust/cDPO-style losses when supported by installed TRL.')
    ap.add_argument('--use-weighting', action=argparse.BooleanOptionalAction, default=True, help='Enable WPO-style pair weighting when supported by installed TRL.')
    ap.add_argument('--precompute-ref-log-probs', action='store_true', help='Precompute reference log-probs when supported; incompatible with some IterableDataset/Liger setups.')
    ap.add_argument('--precompute-ref-batch-size', type=int, default=None)
    ap.add_argument('--padding-free', action='store_true', help='Use TRL padding-free mode when supported.')
    ap.add_argument('--activation-offloading', action='store_true', help='Use TRL activation offloading when supported.')
    ap.add_argument('--use-lora', action='store_true')
    ap.add_argument('--qlora', action='store_true')
    ap.add_argument('--lora-r', type=int, default=16)
    ap.add_argument('--lora-alpha', type=int, default=32)
    ap.add_argument('--lora-dropout', type=float, default=0.05)
    ap.add_argument('--lora-target-modules', default='all-linear')
    ap.add_argument('--bf16', action='store_true')
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--gradient-checkpointing', action='store_true')
    ap.add_argument('--attn-implementation', default='sdpa')
    ap.add_argument('--trust-remote-code', action='store_true')
    ap.add_argument('--train-mode', choices=['auto', 'text', 'vlm'], default='auto')
    ap.add_argument('--image-column', default='images')
    ap.add_argument('--min-pixels', type=int, default=None)
    ap.add_argument('--max-pixels', type=int, default=None)
    ap.add_argument('--max-length', type=int, default=None)
    ap.add_argument(
        '--max-images-per-example',
        type=int,
        default=0,
        help='Training-time VLM memory guard: keep at most this many images per preference row. 0 disables the projection.',
    )
    ap.add_argument(
        '--ddp-find-unused-parameters',
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            'Pass find_unused_parameters to DistributedDataParallel. '
            'Default: enabled automatically for multi-process LoRA/adapter DPO runs, '
            'because some text/VLM branches may be unused on a rank for a step.'
        ),
    )
    ap.add_argument('--resume-from-checkpoint', default=None)
    ap.add_argument('--load-best-model-at-end', action=argparse.BooleanOptionalAction, default=True, help='Select the best eval checkpoint before final save when eval is enabled.')
    ap.add_argument('--native-load-best-model-at-end', action=argparse.BooleanOptionalAction, default=False, help='Use Transformers native best-checkpoint reload. Default is false for PEFT adapter runs because some PEFT/Transformers combinations fail while reloading LoRA adapters in DDP.')
    ap.add_argument('--metric-for-best-model', default='eval_loss')
    ap.add_argument('--greater-is-better', action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument('--dry-run', action='store_true')
    return ap.parse_args()




def _supports_kwargs(callable_obj, kwargs):
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def is_main_process() -> bool:
    return int(os.environ.get('RANK', '0')) == 0


def get_world_size() -> int:
    try:
        return int(os.environ.get('WORLD_SIZE', '1'))
    except ValueError:
        return 1


def resolve_ddp_find_unused_parameters(args: argparse.Namespace, actual_mode: str) -> bool:
    """Return the DDP unused-parameter setting for DPOConfig.

    DPO continues from SFT/VLM adapters and may keep LoRA targets that are not
    used by every rank on every mini-batch. For multi-process runs, enable DDP
    unused-parameter detection unless the caller explicitly overrides it.
    """
    requested = getattr(args, 'ddp_find_unused_parameters', None)
    if requested is not None:
        return bool(requested)
    return get_world_size() > 1


def resolve_precompute_ref_log_probs(args: argparse.Namespace, actual_mode: str) -> bool:
    """Return whether TRL may precompute DPO reference log-probabilities.

    TRL rejects ``precompute_ref_log_probs=True`` for vision datasets because
    VLM image processing is performed on the fly. The full DataSphere DPO stage
    uses mixed multimodal rows, so keep the user-facing flag for text-only DPO
    but force it off for VLM mode to avoid a late DPOTrainer construction crash.
    """
    requested = bool(getattr(args, 'precompute_ref_log_probs', False))
    if requested and actual_mode == 'vlm':
        if is_main_process():
            print(
                '[train_vlm_dpo] disabling precompute_ref_log_probs for VLM mode: '
                'TRL vision datasets are processed on the fly.',
                flush=True,
            )
        return False
    return requested


def _truthy_env(name: str, default: str = '0') -> bool:
    return os.environ.get(name, default).lower() in {'1', 'true', 'yes', 'on'}




def offline_pretrained_kwargs() -> dict[str, Any]:
    """Force local cache usage after DataSphere/Kaggle switches HF into offline mode."""
    if os.environ.get('HF_HUB_OFFLINE') == '1' or os.environ.get('TRANSFORMERS_OFFLINE') == '1':
        return {'local_files_only': True}
    return {}


def _flash_attn_available() -> bool:
    return importlib.util.find_spec('flash_attn') is not None


def resolve_attn_implementation(attn_impl: str) -> str:
    if attn_impl == 'auto':
        return 'flash_attention_2' if _flash_attn_available() else 'sdpa'
    if attn_impl == 'flash_attention_2' and not _flash_attn_available():
        print('[train_vlm_dpo] flash_attn is not installed; falling back to sdpa.', flush=True)
        return 'sdpa'
    return attn_impl


def is_peft_adapter_model(model: Any) -> bool:
    return hasattr(model, 'peft_config') or model.__class__.__name__.lower().startswith('peft')


def should_native_load_best_model(args: argparse.Namespace, eval_ds: Any, model: Any) -> bool:
    if not (bool(args.load_best_model_at_end) and eval_ds is not None):
        return False
    if bool(getattr(args, 'native_load_best_model_at_end', False)):
        return True
    if _truthy_env('USE_NATIVE_PEFT_BEST_MODEL_RELOAD', '0'):
        return True
    return not is_peft_adapter_model(model)


def copy_checkpoint_artifacts(checkpoint_dir: str | Path, output_dir: str | Path, *, prefix: str) -> dict[str, Any]:
    src = Path(checkpoint_dir)
    dst = Path(output_dir)
    manifest: dict[str, Any] = {
        'source_checkpoint': str(src),
        'output_dir': str(dst),
        'copied_files': [],
        'skipped_files': [],
        'status': 'not_started',
        'mode': 'safe_checkpoint_file_copy',
    }
    if not src.exists() or not src.is_dir():
        manifest['status'] = 'missing_checkpoint'
        return manifest
    dst.mkdir(parents=True, exist_ok=True)
    skip_exact = {'optimizer.pt', 'scheduler.pt', 'scaler.pt', 'trainer_state.json', 'training_args.bin'}
    skip_prefixes = ('rng_state',)
    for item in sorted(src.iterdir()):
        name = item.name
        if name in skip_exact or any(name.startswith(prefix) for prefix in skip_prefixes):
            manifest['skipped_files'].append(name)
            continue
        if item.is_file():
            shutil.copy2(item, dst / name)
            manifest['copied_files'].append(name)
        elif item.is_dir() and name in {'tokenizer', 'processor'}:
            target = dst / name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
            manifest['copied_files'].append(name + '/')
        else:
            manifest['skipped_files'].append(name + ('/' if item.is_dir() else ''))
    manifest['status'] = 'copied' if manifest['copied_files'] else 'no_reload_artifacts_found'
    (dst / f'{prefix}_best_checkpoint_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding='utf-8',
    )
    return manifest


def save_best_or_current_model(
    trainer: Any,
    model: Any,
    processor: Any,
    output_dir: Path,
    *,
    native_best_reload_enabled: bool,
    prefix: str,
) -> dict[str, Any]:
    best_checkpoint = getattr(getattr(trainer, 'state', None), 'best_model_checkpoint', None)
    manifest: dict[str, Any] = {
        'best_model_checkpoint': best_checkpoint,
        'native_best_reload_enabled': native_best_reload_enabled,
        'status': 'not_started',
    }
    if best_checkpoint and not native_best_reload_enabled:
        manifest.update(copy_checkpoint_artifacts(best_checkpoint, output_dir, prefix=prefix))
        try:
            processor.save_pretrained(output_dir)
        except Exception as exc:
            manifest['processor_save_error'] = str(exc)
        if manifest.get('status') == 'copied':
            return manifest
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    manifest['status'] = 'trainer_save_model'
    (output_dir / f'{prefix}_best_checkpoint_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding='utf-8',
    )
    return manifest


def disable_trl_model_card_creation(trainer, reason: str) -> None:
    if os.environ.get('DISABLE_TRL_MODEL_CARD', '1').lower() in {'0', 'false', 'no', 'off'}:
        return
    def _skip_model_card(*args, **kwargs):
        if is_main_process():
            print(f'[train_vlm_dpo] skipping TRL model card creation: {reason}', flush=True)
        return None
    try:
        trainer.create_model_card = _skip_model_card
    except Exception as exc:
        if is_main_process():
            print(f'[train_vlm_dpo] warning: could not disable TRL model card creation: {exc}', flush=True)

def get_local_rank() -> int:
    return int(os.environ.get('LOCAL_RANK', '0'))


def load_qwen_model(model_id: str, qlora: bool, bf16: bool, fp16: bool, trust_remote_code: bool, attn_impl: str):
    import torch

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
    kwargs = {'trust_remote_code': trust_remote_code, 'attn_implementation': resolve_attn_implementation(attn_impl), **offline_pretrained_kwargs()}
    if torch_dtype is not None:
        kwargs['torch_dtype'] = torch_dtype
    if quant_config is not None:
        kwargs['quantization_config'] = quant_config
    if device_map is not None:
        kwargs['device_map'] = device_map
    if 'Qwen3-VL' in model_id:
        model_cls = Qwen3VLForConditionalGeneration
    elif 'Qwen2.5-VL' in model_id or 'Qwen2-VL' in model_id:
        model_cls = Qwen2_5_VLForConditionalGeneration
    else:
        raise ValueError(f'Unsupported model for this entrypoint: {model_id}')
    try:
        return model_cls.from_pretrained(model_id, **kwargs)
    except Exception as exc:
        if kwargs.get('attn_implementation') == 'flash_attention_2':
            print(f'[train_vlm_dpo] flash_attention_2 load failed ({exc!r}); retrying with sdpa.', flush=True)
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
        path = (base_dir / path).resolve()
    return str(path.as_posix())


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


def _text_block(value: Any) -> dict[str, str]:
    return {'type': 'text', 'text': '' if value is None else str(value)}


def _messages_have_image_block(messages: list[dict[str, Any]]) -> bool:
    for msg in messages:
        content = msg.get('content')
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'image':
                    return True
    return False


def _align_image_placeholders_to_images(messages: list[dict[str, Any]], image_count: int) -> list[dict[str, Any]]:
    """Keep TRL multimodal placeholders in lockstep with the images list.

    TRL validates that every conversational DPO example contains exactly as
    many ``{'type': 'image'}`` blocks in ``prompt`` as there are elements in
    ``images``. DPO applies a training-time image cap for memory, so the prompt
    must be realigned after formatting and after any later image projection.
    """
    image_count = max(0, int(image_count or 0))
    kept_images = 0
    aligned: list[dict[str, Any]] = []
    for msg in messages:
        role = str(msg.get('role') or 'user')
        content = msg.get('content')
        if not isinstance(content, list):
            aligned.append({'role': role, 'content': content})
            continue
        new_content: list[dict[str, Any]] = []
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


def _ensure_image_placeholder(messages: list[dict[str, Any]], has_images: bool) -> list[dict[str, Any]]:
    if not has_images or _messages_have_image_block(messages):
        return messages
    for msg in messages:
        if msg.get('role') == 'user':
            content = msg.setdefault('content', [])
            if isinstance(content, str):
                msg['content'] = [{'type': 'image'}, {'type': 'text', 'text': content}]
            elif isinstance(content, list):
                content.insert(0, {'type': 'image'})
            else:
                msg['content'] = [{'type': 'image'}]
            return messages
    messages.insert(0, {'role': 'user', 'content': [{'type': 'image'}]})
    return messages


def _assistant_completion_message(value: Any) -> list[dict[str, str]]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and value.get('role'):
        return [value]
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False)
    return [{'role': 'assistant', 'content': value}]


def make_dpo_formatter(base_dir: Path):
    def format_dpo(example):
        images = []
        for image in _normalize_image_list(example.get('images'), base_dir):
            if image not in images:
                images.append(image)
        for image in _normalize_image_list(example.get('image'), base_dir):
            if image not in images:
                images.append(image)
        example['images'] = images
        # The v2 builder stores prompt as chat messages and chosen/rejected as
        # compact strings to keep JSONL readable.  TRL's conversational DPO
        # format requires all three fields to be message lists, so normalize
        # completions here rather than duplicating message wrappers in the data.
        # For VLM rows, also make prompt image placeholders match the normalized
        # image list before TRL's multimodal DPO collator validates the sample.
        if isinstance(example.get('prompt'), list):
            prompt = _ensure_image_placeholder(example.get('prompt') or [], bool(images))
            example['prompt'] = _align_image_placeholders_to_images(prompt, len(images))
            example['chosen'] = _assistant_completion_message(example.get('chosen', ''))
            example['rejected'] = _assistant_completion_message(example.get('rejected', ''))
        return example
    return format_dpo



def _read_json_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    text = path.read_text(encoding='utf-8')
    if not text.strip():
        return []
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    obj = json.loads(text)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ('data', 'rows', 'examples'):
            if isinstance(obj.get(key), list):
                return obj[key]
        return [obj]
    raise ValueError(f'Unsupported JSON top-level value in {path}: {type(obj).__name__}')


def _load_dpo_json_dataset_loose(path: str | Path, base_dir: Path):
    """Load DPO JSONL through a stable preference-column projection.

    Raw v2 rows may contain heterogeneous metadata dictionaries that break
    Arrow schema inference in ``load_dataset('json')``.  DPOTrainer only needs
    prompt/chosen/rejected plus optional image columns, so normalize those before
    constructing the Dataset.
    """
    formatter = make_dpo_formatter(base_dir)
    rows = []
    for raw in _read_json_records(path):
        if not isinstance(raw, dict):
            continue
        ex = formatter(dict(raw))
        images = _normalize_image_list(ex.get('images'), Path('.'))
        image = ex.get('image')
        if image in (None, '', []):
            image = images[0] if images else None
        rows.append({
            'prompt': ex.get('prompt') or [],
            'chosen': ex.get('chosen') or '',
            'rejected': ex.get('rejected') or '',
            'images': images,
            'image': image,
        })
    if not rows:
        raise ValueError(f'No DPO rows found in {path}')
    return Dataset.from_list(rows)

def _value_has_image(value) -> bool:
    if value in (None, ''):
        return False
    if isinstance(value, list):
        return any(item not in (None, '', []) for item in value)
    return True



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

def cap_dpo_images_for_memory(ds, max_images_per_example: int, split_name: str):
    if max_images_per_example is None or int(max_images_per_example) <= 0:
        return ds, {"enabled": False, "max_images_per_example": int(max_images_per_example or 0), "truncated_rows": 0, "dropped_image_refs": 0, "selection_policy": "evidence_aware_top_k_then_original_order"}
    max_images_per_example = int(max_images_per_example)
    stats = {"enabled": True, "max_images_per_example": max_images_per_example, "truncated_rows": 0, "dropped_image_refs": 0, "selection_policy": "evidence_aware_top_k_then_original_order"}
    for row in ds:
        images = []
        images.extend(_normalize_image_list(row.get('images'), Path('.')))
        images.extend(_normalize_image_list(row.get('image'), Path('.')))
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

    def _cap(example):
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
            prompt = _ensure_image_placeholder(example.get('prompt') or [], bool(kept))
            example['prompt'] = _align_image_placeholders_to_images(prompt, len(kept))
        return example

    capped = ds.map(_cap, desc=f'Capping {split_name} DPO rows to {max_images_per_example} images for GPU memory')
    if is_main_process():
        print(
            f'[train_vlm_dpo] training memory guard capped {stats["truncated_rows"]}/{len(ds)} '
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

    # Keep mixed text-only + image rows instead of filtering them out. This
    # matches the SFT/GRPO entrypoints and prevents silent loss of examples.
    return _cast_images_column(ds, detected_column), 'vlm'


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.loss_weights is not None and len(args.loss_weights) != len(args.loss_type):
        raise ValueError('--loss-weights length must match --loss-type length')

    train_ds = _load_dpo_json_dataset_loose(args.train_file, args.train_file.parent)
    eval_ds = _load_dpo_json_dataset_loose(args.eval_file, args.eval_file.parent) if args.eval_file else None

    train_image_cap_stats = {"enabled": False, "max_images_per_example": int(args.max_images_per_example or 0), "truncated_rows": 0, "dropped_image_refs": 0, "selection_policy": "evidence_aware_top_k_then_original_order"}
    eval_image_cap_stats = {"enabled": False, "max_images_per_example": int(args.max_images_per_example or 0), "truncated_rows": 0, "dropped_image_refs": 0, "selection_policy": "evidence_aware_top_k_then_original_order"}
    if args.max_images_per_example and int(args.max_images_per_example) > 0:
        train_ds, train_image_cap_stats = cap_dpo_images_for_memory(train_ds, args.max_images_per_example, 'train')
    if eval_ds is not None:
        if args.max_images_per_example and int(args.max_images_per_example) > 0:
            eval_ds, eval_image_cap_stats = cap_dpo_images_for_memory(eval_ds, args.max_images_per_example, 'eval')

    train_ds, actual_mode = maybe_prepare_dataset(train_ds, args.image_column, args.train_mode)
    if eval_ds is not None:
        eval_ds, _ = maybe_prepare_dataset(eval_ds, args.image_column, actual_mode)

    processor_kwargs = {'trust_remote_code': args.trust_remote_code}
    if args.min_pixels is not None:
        processor_kwargs['min_pixels'] = args.min_pixels
    if args.max_pixels is not None:
        processor_kwargs['max_pixels'] = args.max_pixels
    try:
        processor = AutoProcessor.from_pretrained(args.model_id, **processor_kwargs, **offline_pretrained_kwargs())
    except TypeError:
        processor_kwargs.pop('min_pixels', None)
        processor_kwargs.pop('max_pixels', None)
        processor = AutoProcessor.from_pretrained(args.model_id, **processor_kwargs, **offline_pretrained_kwargs())
    tokenizer = getattr(processor, 'tokenizer', None) or AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code, **offline_pretrained_kwargs())
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(processor, 'tokenizer') and getattr(processor.tokenizer, 'pad_token', None) is None:
        processor.tokenizer.pad_token = tokenizer.pad_token

    model = load_qwen_model(args.model_id, args.qlora, args.bf16, args.fp16, args.trust_remote_code, args.attn_implementation)
    if args.qlora:
        model = prepare_model_for_kbit_training(model)

    if args.sft_adapter_path:
        model = PeftModel.from_pretrained(model, str(args.sft_adapter_path), is_trainable=True, **offline_pretrained_kwargs())
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

    effective_loss_type = args.loss_type if len(args.loss_type) > 1 else args.loss_type[0]
    effective_loss_weights = args.loss_weights

    effective_precompute_ref_log_probs = resolve_precompute_ref_log_probs(args, actual_mode)

    dpo_kwargs = dict(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        report_to=[] if args.report_to == 'none' else [args.report_to],
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        bf16=args.bf16,
        fp16=args.fp16,
        max_length=args.max_length,
        save_strategy='steps',
        eval_strategy='steps' if eval_ds is not None else 'no',
        logging_strategy='steps',
        ddp_find_unused_parameters=resolve_ddp_find_unused_parameters(args, actual_mode),
        beta=args.beta,
        loss_type=effective_loss_type,
        loss_weights=effective_loss_weights,
        label_smoothing=args.label_smoothing,
        use_weighting=args.use_weighting,
        precompute_ref_log_probs=effective_precompute_ref_log_probs,
        precompute_ref_batch_size=args.precompute_ref_batch_size if effective_precompute_ref_log_probs else None,
        padding_free=args.padding_free,
        activation_offloading=args.activation_offloading,
        load_best_model_at_end=should_native_load_best_model(args, eval_ds, model),
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
    )
    native_best_reload_enabled = bool(dpo_kwargs['load_best_model_at_end'])
    dpo_args = DPOConfig(**_supports_kwargs(DPOConfig, dpo_kwargs))
    if actual_mode == 'vlm' and dpo_args.max_length is not None:
        raise ValueError('For VLM DPO use max_length=None to avoid truncating image tokens.')

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor if actual_mode == 'vlm' else tokenizer,
    )

    disable_trl_model_card_creation(
        trainer,
        'DataSphere may expose an ASCII locale; TRL/huggingface_hub model-card templates are UTF-8.',
    )

    run_config = vars(args).copy()
    run_config['resolved_mode'] = actual_mode
    run_config['ddp_find_unused_parameters_resolved'] = resolve_ddp_find_unused_parameters(args, actual_mode)
    run_config['precompute_ref_log_probs_resolved'] = effective_precompute_ref_log_probs
    run_config['train_examples'] = len(train_ds)
    run_config['eval_examples'] = len(eval_ds) if eval_ds is not None else 0
    run_config['train_image_cap_stats'] = train_image_cap_stats
    run_config['eval_image_cap_stats'] = eval_image_cap_stats
    run_config['best_checkpoint_selection_enabled'] = bool(args.load_best_model_at_end and eval_ds is not None)
    run_config['native_best_checkpoint_reload_enabled'] = native_best_reload_enabled
    run_config['best_checkpoint_selection_mode'] = 'native_load' if native_best_reload_enabled else ('safe_checkpoint_file_copy' if args.load_best_model_at_end and eval_ds is not None else 'disabled')
    run_config['effective_loss_type'] = effective_loss_type
    run_config['effective_loss_weights'] = effective_loss_weights
    (args.output_dir / 'run_config.json').write_text(json.dumps(run_config, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

    if args.dry_run:
        print(json.dumps(run_config, ensure_ascii=False, indent=2, default=str))
        return

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_metrics('train', {**train_result.metrics, 'train_examples': len(train_ds)})
    if eval_ds is not None:
        eval_metrics = trainer.evaluate()
        eval_metrics['eval_examples'] = len(eval_ds)
        trainer.save_metrics('eval', eval_metrics)
    save_manifest = save_best_or_current_model(
        trainer,
        model,
        processor,
        args.output_dir,
        native_best_reload_enabled=native_best_reload_enabled,
        prefix='dpo',
    )
    if is_main_process():
        print(f'[train_vlm_dpo] final save status: {save_manifest.get("status")}; best checkpoint: {save_manifest.get("best_model_checkpoint")}', flush=True)




def cleanup_distributed_process_group() -> None:
    """Best-effort DDP/NCCL shutdown to avoid noisy or hanging process teardown."""
    try:
        import torch
        dist = getattr(torch, 'distributed', None)
        if dist is not None and getattr(dist, 'is_available', lambda: False)() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception as exc:
        if is_main_process():
            print(f'[train_vlm_dpo] warning: failed to destroy distributed process group: {exc}', flush=True)


def main_with_distributed_cleanup() -> None:
    try:
        main()
    finally:
        cleanup_distributed_process_group()


if __name__ == '__main__':
    main_with_distributed_cleanup()
