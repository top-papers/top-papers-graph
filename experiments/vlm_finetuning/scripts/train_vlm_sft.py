#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Dict

from datasets import Image as HFImage, Sequence
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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
from trl import SFTConfig, SFTTrainer


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-id', required=True)
    ap.add_argument('--train-file', type=str, required=True)
    ap.add_argument('--eval-file', type=str, default=None)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--report-to', default='none')
    ap.add_argument('--learning-rate', type=float, default=7e-5)
    ap.add_argument('--num-train-epochs', type=float, default=3.0)
    ap.add_argument('--max-steps', type=int, default=-1)
    ap.add_argument('--per-device-train-batch-size', type=int, default=1)
    ap.add_argument('--per-device-eval-batch-size', type=int, default=1)
    ap.add_argument('--gradient-accumulation-steps', type=int, default=8)
    ap.add_argument('--warmup-ratio', type=float, default=0.05)
    ap.add_argument('--logging-steps', type=int, default=10)
    ap.add_argument('--save-steps', type=int, default=100)
    ap.add_argument('--eval-steps', type=int, default=100)
    ap.add_argument('--save-total-limit', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--lora-r', type=int, default=32)
    ap.add_argument('--lora-alpha', type=int, default=64)
    ap.add_argument('--lora-dropout', type=float, default=0.05)
    ap.add_argument('--lora-target-modules', default='all-linear')
    ap.add_argument('--use-lora', action='store_true')
    ap.add_argument('--qlora', action='store_true')
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
    ap.add_argument('--max-length', type=int, default=None)
    ap.add_argument('--assistant-only-loss', action='store_true')
    ap.add_argument('--optim', default='adamw_torch_fused')
    ap.add_argument('--lr-scheduler-type', default='cosine')
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--max-grad-norm', type=float, default=0.3)
    ap.add_argument('--dataloader-num-workers', type=int, default=4)
    ap.add_argument('--resume-from-checkpoint', default=None)
    ap.add_argument('--save-adapter-only', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    return ap.parse_args()


def get_local_rank() -> int:
    return int(os.environ.get('LOCAL_RANK', '0'))


def is_main_process() -> bool:
    return int(os.environ.get('RANK', '0')) == 0


def _supports_kwargs(callable_obj: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter config kwargs for compatibility across TRL versions and tests."""
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _flash_attn_available() -> bool:
    return importlib.util.find_spec('flash_attn') is not None


def resolve_attn_implementation(attn_impl: str) -> str:
    if attn_impl == 'auto':
        return 'flash_attention_2' if _flash_attn_available() else 'sdpa'
    if attn_impl == 'flash_attention_2' and not _flash_attn_available():
        print('[train_vlm_sft] flash_attn is not installed; falling back to sdpa.', flush=True)
        return 'sdpa'
    return attn_impl


def load_processor(model_id: str, trust_remote_code: bool, min_pixels: int | None, max_pixels: int | None):
    kwargs: Dict[str, Any] = {'trust_remote_code': trust_remote_code}
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
    kwargs: Dict[str, Any] = {'trust_remote_code': trust_remote_code, 'attn_implementation': resolve_attn_implementation(attn_impl)}
    if torch_dtype is not None:
        # `torch_dtype` remains accepted by released Transformers versions; model cards may show `dtype`.
        kwargs['torch_dtype'] = torch_dtype
    if quant_config is not None:
        kwargs['quantization_config'] = quant_config
    if device_map is not None:
        kwargs['device_map'] = device_map

    model_cls: Any
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
            print(f'[train_vlm_sft] flash_attention_2 load failed ({exc!r}); retrying with sdpa.', flush=True)
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
        if path.exists():
            path = path.resolve()
        else:
            path = (base_dir / path).resolve()
    return str(path.as_posix())


def _safe_text(value: Any) -> str:
    """Convert arbitrary JSON-like content into text that chat templates can render safely."""
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
    resolved = image_ref
    if resolved and resolved not in images:
        images.append(resolved)
    # TRL VLM datasets carry actual images in the top-level image/images column.
    # Keep the message block minimal and template-safe.
    blocks.append({'type': 'image'})


def _canonicalize_content_blocks(content: Any, base_dir: Path, images: list[Any]) -> list[Dict[str, Any]]:
    """Return only Qwen/TRL-safe content blocks.

    Qwen3-VL's Jinja chat template performs membership tests such as
    ``'image' in content`` on each content item. If a raw export contains an
    integer, list, or arbitrary JSON object inside message.content, the template
    can raise ``TypeError: argument of type 'int' is not iterable``.  This
    normalizer makes every item either ``{'type': 'text', 'text': ...}`` or
    ``{'type': 'image'}`` before TRL tokenization.
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
            # This training pipeline is image-only. Preserve the information as
            # text rather than sending unsupported video blocks into Qwen3-VL SFT.
            _append_text_block(blocks, item)
            continue
        if 'text' in item:
            _append_text_block(blocks, item.get('text'))
            continue
        # Arbitrary structured export content is retained as JSON text so that
        # the sample remains useful and the chat template never sees raw ints.
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


def _flatten_single_text_messages(messages):
    """Keep pure text turns in classic TRL chat format for legacy callers only."""
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



def _strict_qwen_safe_messages(messages: Any, row_idx: int | None = None) -> list[Dict[str, Any]]:
    """Validate and re-sanitize messages immediately before SFTTrainer.

    TRL passes conversational rows to ``processor.apply_chat_template``.  The
    Qwen3-VL template tests each user/tool content item with expressions like
    ``content.type == 'image' or 'image' in content``; a raw integer in any
    content list therefore raises ``TypeError: argument of type 'int' is not
    iterable``.  This final guard makes the dataset schema deliberately narrow:
    messages contain only role + either string content or list[{'type': ...}].
    """
    safe, _ = _canonicalize_messages(messages, Path('.'))
    out: list[Dict[str, Any]] = []
    for msg in safe:
        role = str(msg.get('role') or 'user')
        if role not in {'system', 'user', 'assistant', 'tool'}:
            role = 'user'
        content = msg.get('content')
        if isinstance(content, str):
            out.append({'role': role, 'content': content})
            continue
        if not isinstance(content, list):
            content = [_text_block(content)]
        blocks: list[Dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                block = _text_block(block)
            btype = str(block.get('type') or '').lower()
            if btype == 'image':
                blocks.append({'type': 'image'})
            elif btype == 'video':
                # The current pipeline is image-only; keep video metadata as text.
                blocks.append(_text_block(block))
            else:
                text_value = block.get('text', block)
                text = _safe_text(text_value)
                if text.strip():
                    blocks.append({'type': 'text', 'text': text})
        if blocks:
            out.append({'role': role, 'content': blocks})
    if not out:
        raise ValueError(f'Row {row_idx}: messages became empty after sanitization')
    return out


def _sanitize_sft_row_for_trl(example: Dict[str, Any], idx: int | None = None) -> Dict[str, Any]:
    example = dict(example)
    example['messages'] = _strict_qwen_safe_messages(example.get('messages'), idx)
    return example


def _keep_only_trl_sft_columns(ds, image_column: str):
    keep = {'messages'}
    for col in {image_column, 'images', 'image'}:
        if col and col in ds.column_names:
            keep.add(col)
    remove = [col for col in ds.column_names if col not in keep]
    return ds.remove_columns(remove) if remove else ds


def _validate_sft_dataset_for_qwen(ds, split_name: str, limit: int = 256) -> None:
    n = min(len(ds), limit)
    for i in range(n):
        messages = ds[i].get('messages')
        _strict_qwen_safe_messages(messages, i)
    print(f'[train_vlm_sft] validated {n}/{len(ds)} {split_name} rows for Qwen3-VL chat template safety.', flush=True)

def _make_label_messages(label_value: Any) -> list[Dict[str, Any]]:
    return [
        {
            'role': 'user',
            'content': [
                {'type': 'image'},
                {'type': 'text', 'text': 'What is the correct label for this image? Return only the class label.'},
            ],
        },
        {
            'role': 'assistant',
            'content': [{'type': 'text', 'text': str(label_value)}],
        },
    ]



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


def _normalise_sft_example(example, base_dir: Path | None = None):
    """Normalize SFT rows from HF export, imagefolder, or task chat artifacts.

    The returned ``messages`` value intentionally keeps every content field as a
    list of explicit blocks.  This avoids heterogeneous Arrow schemas and keeps
    Qwen3-VL's Jinja chat template from seeing raw integers/lists inside
    ``message.content``.
    """
    base_dir = base_dir or Path('.')
    if example.get('image') not in (None, '', []):
        example['image'] = _resolve_image_ref(example.get('image'), base_dir)
    if example.get('images') not in (None, '', []):
        example['images'] = _normalize_image_list(example.get('images'), base_dir)

    source_messages = example.get('messages')
    if not source_messages and isinstance(example.get('chat'), dict):
        source_messages = example['chat'].get('messages')

    images = []
    images.extend(_normalize_image_list(example.get('images'), base_dir))
    if example.get('image') not in (None, '', []):
        image_ref = _resolve_image_ref(example.get('image'), base_dir)
        if image_ref and image_ref not in images:
            images.append(image_ref)

    if source_messages:
        messages, embedded_images = _canonicalize_messages(source_messages, base_dir)
        for image_ref in embedded_images:
            if image_ref not in images:
                images.append(image_ref)
        if not messages and 'label' in example:
            messages = _make_label_messages(example.get('label_text', example['label']))
        example['messages'] = _ensure_image_placeholder(messages, bool(images))
        example['images'] = images
        if images:
            example['image'] = images[0]
        return example

    if 'label' in example:
        label_value = example.get('label_text', example['label'])
        example['messages'] = _make_label_messages(label_value)
    if images:
        example['messages'] = _ensure_image_placeholder(example.get('messages', []), True)
        example['images'] = images
        example['image'] = images[0]
    return example

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
    image_columns = [col for col in ['images', 'image'] if col in ds.column_names]
    image_columns_to_remove = [col for col in image_columns if col != detected_column]

    if requested_mode == 'text' or non_null == 0:
        return ds.remove_columns(image_columns), 'text'

    if image_columns_to_remove:
        ds = ds.remove_columns(image_columns_to_remove)
    return _cast_images_column(ds, detected_column), 'vlm'


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.tf32:
        try:
            import torch

            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    train_file_str = args.train_file
    if train_file_str.endswith('.json') or train_file_str.endswith('.jsonl'):
        data_files = {'train': train_file_str}
        if args.eval_file:
            data_files['eval'] = args.eval_file
        ds = load_dataset('json', data_files=data_files)
        train_ds = ds['train']
        eval_ds = ds.get('eval')
    else:
        ds = load_dataset(train_file_str)
        train_ds = ds.get('train')
        if train_ds is None and 'validation' in ds:
            train_ds = ds['validation']
        eval_ds = ds.get('eval')

    train_base_dir = Path(train_file_str).parent if train_file_str.endswith(('.json', '.jsonl')) else Path('.')
    train_ds = train_ds.map(lambda example: _normalise_sft_example(example, train_base_dir))
    train_ds = _keep_only_trl_sft_columns(train_ds, args.image_column)
    train_ds = train_ds.map(_sanitize_sft_row_for_trl, with_indices=True, desc='Sanitizing train messages for Qwen3-VL')
    _validate_sft_dataset_for_qwen(train_ds, 'train')
    if eval_ds is not None:
        eval_base_dir = Path(args.eval_file).parent if args.eval_file else train_base_dir
        eval_ds = eval_ds.map(lambda example: _normalise_sft_example(example, eval_base_dir))
        eval_ds = _keep_only_trl_sft_columns(eval_ds, args.image_column)
        eval_ds = eval_ds.map(_sanitize_sft_row_for_trl, with_indices=True, desc='Sanitizing eval messages for Qwen3-VL')
        _validate_sft_dataset_for_qwen(eval_ds, 'eval')

    train_ds, actual_mode = maybe_prepare_dataset(train_ds, args.image_column, args.train_mode)
    if eval_ds is not None:
        eval_ds, _ = maybe_prepare_dataset(eval_ds, args.image_column, actual_mode)

    processor = load_processor(args.model_id, args.trust_remote_code, args.min_pixels, args.max_pixels)
    tokenizer = getattr(processor, 'tokenizer', None) or AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(processor, 'tokenizer') and getattr(processor.tokenizer, 'pad_token', None) is None:
        processor.tokenizer.pad_token = tokenizer.pad_token

    model = load_qwen_model(args.model_id, args.qlora, args.bf16, args.fp16, args.trust_remote_code, args.attn_implementation)
    if args.gradient_checkpointing and hasattr(model, 'config'):
        try:
            model.config.use_cache = False
        except Exception:
            pass
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
        if args.gradient_checkpointing and hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
        if is_main_process():
            model.print_trainable_parameters()

    sft_kwargs = {
        'output_dir': str(args.output_dir),
        'learning_rate': args.learning_rate,
        'num_train_epochs': args.num_train_epochs,
        'max_steps': args.max_steps,
        'per_device_train_batch_size': args.per_device_train_batch_size,
        'per_device_eval_batch_size': args.per_device_eval_batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'warmup_ratio': args.warmup_ratio,
        'logging_steps': args.logging_steps,
        'save_steps': args.save_steps,
        'eval_steps': args.eval_steps,
        'save_total_limit': args.save_total_limit,
        'report_to': [] if args.report_to == 'none' else [args.report_to],
        'remove_unused_columns': False,
        'gradient_checkpointing': args.gradient_checkpointing,
        'gradient_checkpointing_kwargs': {'use_reentrant': False},
        'bf16': args.bf16,
        'fp16': args.fp16,
        'tf32': args.tf32,
        'max_length': args.max_length,
        'save_strategy': 'steps',
        'eval_strategy': 'steps' if eval_ds is not None else 'no',
        'logging_strategy': 'steps',
        'dataset_num_proc': 1,
        'assistant_only_loss': args.assistant_only_loss,
        'optim': args.optim,
        'lr_scheduler_type': args.lr_scheduler_type,
        'weight_decay': args.weight_decay,
        'max_grad_norm': args.max_grad_norm,
        'dataloader_num_workers': args.dataloader_num_workers,
        'dataloader_pin_memory': True,
        'ddp_find_unused_parameters': False,
    }
    sft_args = SFTConfig(**_supports_kwargs(SFTConfig, sft_kwargs))
    if actual_mode == 'vlm' and sft_args.max_length is not None:
        raise ValueError('For VLM training use max_length=None to avoid truncating image tokens.')

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor if actual_mode == 'vlm' else tokenizer,
    )

    run_config = vars(args).copy()
    run_config['resolved_mode'] = actual_mode
    run_config['train_examples'] = len(train_ds)
    run_config['eval_examples'] = len(eval_ds) if eval_ds is not None else 0
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
    if args.save_adapter_only and hasattr(model, 'save_pretrained'):
        model.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)
    else:
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)


if __name__ == '__main__':
    main()
