#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import Image as HFImage
from datasets import Sequence as HFSequence
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
from trl import SFTConfig, SFTTrainer


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-id', required=True)
    ap.add_argument('--train-file', type=Path, required=True)
    ap.add_argument('--eval-file', type=Path, default=None)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--report-to', default='none')
    ap.add_argument('--learning-rate', type=float, default=2e-4)
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
    ap.add_argument('--lora-r', type=int, default=16)
    ap.add_argument('--lora-alpha', type=int, default=32)
    ap.add_argument('--lora-dropout', type=float, default=0.05)
    ap.add_argument('--lora-target-modules', default='all-linear')
    ap.add_argument('--use-lora', action='store_true')
    ap.add_argument('--qlora', action='store_true')
    ap.add_argument('--bf16', action='store_true')
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--gradient-checkpointing', action='store_true')
    ap.add_argument('--attn-implementation', default='sdpa')
    ap.add_argument('--trust-remote-code', action='store_true')
    ap.add_argument('--train-mode', choices=['auto', 'text', 'vlm'], default='auto')
    ap.add_argument('--image-column', default='image')
    ap.add_argument('--images-column', default='images')
    ap.add_argument('--max-length', type=int, default=None)
    ap.add_argument('--resume-from-checkpoint', default=None)
    ap.add_argument('--save-adapter-only', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    return ap.parse_args()


def get_local_rank() -> int:
    return int(os.environ.get('LOCAL_RANK', '0'))


def is_main_process() -> bool:
    return int(os.environ.get('RANK', '0')) == 0


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
    kwargs = {'trust_remote_code': trust_remote_code, 'attn_implementation': attn_impl}
    if torch_dtype is not None:
        kwargs['torch_dtype'] = torch_dtype
    if quant_config is not None:
        kwargs['quantization_config'] = quant_config
    if device_map is not None:
        kwargs['device_map'] = device_map

    if 'Qwen3-VL' in model_id:
        return Qwen3VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    if 'Qwen2.5-VL' in model_id or 'Qwen2-VL' in model_id:
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    raise ValueError(f'Unsupported model for this entrypoint: {model_id}')


def _as_messages(value):
    if isinstance(value, dict) and isinstance(value.get('messages'), list):
        return value['messages']
    if isinstance(value, list):
        return value
    return None


def _normalise_message_content(content):
    image_paths = []
    if isinstance(content, str):
        return content, image_paths
    if not isinstance(content, list):
        return str(content or ''), image_paths

    blocks = []
    has_image = False
    text_parts = []
    for block in content:
        if isinstance(block, str):
            if block.strip():
                blocks.append({'type': 'text', 'text': block})
                text_parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        block_type = str(block.get('type') or '').strip().lower()
        if block_type == 'image':
            has_image = True
            image_path = block.get('image') or block.get('path') or block.get('url')
            if image_path not in (None, '', []):
                image_paths.append(str(image_path))
            blocks.append({'type': 'image'})
        elif block_type == 'text' or 'text' in block:
            text = str(block.get('text') or '')
            if text:
                blocks.append({'type': 'text', 'text': text})
                text_parts.append(text)

    if has_image:
        return blocks, image_paths
    return '\n'.join(part for part in text_parts if part).strip(), image_paths


def _normalise_messages(messages):
    normalised = []
    image_paths = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content, paths = _normalise_message_content(message.get('content', ''))
        image_paths.extend(paths)
        normalised.append({'role': message.get('role', 'user'), 'content': content})
    return normalised, image_paths


def _normalise_sft_example(example):
    messages = _as_messages(example.get('messages')) or _as_messages(example.get('chat'))
    if messages is None:
        return example
    normalised, image_paths = _normalise_messages(messages)
    example['messages'] = normalised
    example['images'] = image_paths
    return example


def _has_image_value(value) -> bool:
    if value in (None, '', []):
        return False
    if isinstance(value, list):
        return any(_has_image_value(item) for item in value)
    return True


def _image_columns(ds, image_column: str, images_column: str) -> list[str]:
    candidates = [image_column, images_column, 'image', 'images']
    out = []
    for column in candidates:
        if column and column in ds.column_names and column not in out:
            out.append(column)
    return out


def _column_has_sequence_values(ds, column: str) -> bool:
    sample = ds[: min(len(ds), 128)]
    return any(isinstance(value, list) for value in sample.get(column, []))


def _cast_image_columns(ds, columns: list[str]):
    for column in columns:
        feature = HFSequence(HFImage()) if _column_has_sequence_values(ds, column) else HFImage()
        ds = ds.cast_column(column, feature)
    return ds


def maybe_prepare_dataset(ds, image_column: str, images_column: str, requested_mode: str):
    columns = _image_columns(ds, image_column, images_column)
    sample_size = min(len(ds), 128)
    sample = ds[:sample_size]
    image_rows = 0
    for i in range(sample_size):
        if any(_has_image_value(sample.get(column, [None] * sample_size)[i]) for column in columns):
            image_rows += 1

    if requested_mode == 'text':
        return ds.remove_columns(columns) if columns else ds, 'text'
    if requested_mode == 'vlm':
        return _cast_image_columns(ds, columns) if columns else ds, 'vlm'
    if image_rows > 0:
        if image_rows < len(ds):
            print(f'[warn] mixed text/VLM dataset detected ({image_rows}/{len(ds)} image rows). Keeping VLM mode; transformers>=4.57 is required for mixed batches.')
        return _cast_image_columns(ds, columns) if columns else ds, 'vlm'
    return ds.remove_columns(columns) if columns else ds, 'text'


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_files = {'train': str(args.train_file)}
    if args.eval_file:
        data_files['eval'] = str(args.eval_file)
    ds = load_dataset('json', data_files=data_files)
    train_ds = ds['train']
    eval_ds = ds.get('eval')

    train_ds = train_ds.map(_normalise_sft_example)
    if eval_ds is not None:
        eval_ds = eval_ds.map(_normalise_sft_example)

    train_ds, actual_mode = maybe_prepare_dataset(train_ds, args.image_column, args.images_column, args.train_mode)
    if eval_ds is not None:
        eval_ds, _ = maybe_prepare_dataset(eval_ds, args.image_column, args.images_column, actual_mode)

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    tokenizer = getattr(processor, 'tokenizer', None) or AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(processor, 'tokenizer') and getattr(processor.tokenizer, 'pad_token', None) is None:
        processor.tokenizer.pad_token = tokenizer.pad_token

    model = load_qwen_model(args.model_id, args.qlora, args.bf16, args.fp16, args.trust_remote_code, args.attn_implementation)
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
        if is_main_process():
            model.print_trainable_parameters()

    sft_args = SFTConfig(
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
        bf16=args.bf16,
        fp16=args.fp16,
        max_length=args.max_length,
        save_strategy='steps',
        eval_strategy='steps' if eval_ds is not None else 'no',
        logging_strategy='steps',
        dataset_num_proc=1,
    )
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
