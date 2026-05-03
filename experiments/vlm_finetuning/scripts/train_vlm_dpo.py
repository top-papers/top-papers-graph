#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import Image as HFImage, Sequence
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
    ap.add_argument('--beta', type=float, default=0.1)
    ap.add_argument('--loss-type', default='sigmoid')
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
    ap.add_argument('--max-length', type=int, default=None)
    ap.add_argument('--resume-from-checkpoint', default=None)
    ap.add_argument('--dry-run', action='store_true')
    return ap.parse_args()


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


def make_dpo_formatter(base_dir: Path):
    def format_dpo(example):
        images = []
        images.extend(_normalize_image_list(example.get('images'), base_dir))
        images.extend(_normalize_image_list(example.get('image'), base_dir))
        example['images'] = images
        return example
    return format_dpo


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

    # Keep mixed text-only + image rows instead of filtering them out. This
    # matches the SFT/GRPO entrypoints and prevents silent loss of examples.
    return _cast_images_column(ds, detected_column), 'vlm'


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

    train_ds = train_ds.map(make_dpo_formatter(args.train_file.parent))
    if eval_ds is not None:
        eval_base_dir = args.eval_file.parent if args.eval_file else args.train_file.parent
        eval_ds = eval_ds.map(make_dpo_formatter(eval_base_dir))

    train_ds, actual_mode = maybe_prepare_dataset(train_ds, args.image_column, args.train_mode)
    if eval_ds is not None:
        eval_ds, _ = maybe_prepare_dataset(eval_ds, args.image_column, actual_mode)

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    tokenizer = getattr(processor, 'tokenizer', None) or AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(processor, 'tokenizer') and getattr(processor.tokenizer, 'pad_token', None) is None:
        processor.tokenizer.pad_token = tokenizer.pad_token

    model = load_qwen_model(args.model_id, args.qlora, args.bf16, args.fp16, args.trust_remote_code, args.attn_implementation)
    if args.qlora:
        model = prepare_model_for_kbit_training(model)

    if args.sft_adapter_path:
        model = PeftModel.from_pretrained(model, str(args.sft_adapter_path), is_trainable=True)
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

    dpo_args = DPOConfig(
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
        beta=args.beta,
        loss_type=args.loss_type,
    )
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

    run_config = vars(args).copy()
    run_config['resolved_mode'] = actual_mode
    run_config['train_examples'] = len(train_ds)
    run_config['eval_examples'] = len(eval_ds) if eval_ds is not None else 0
    (args.output_dir / 'run_config.json').write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding='utf-8')

    if args.dry_run:
        print(json.dumps(run_config, ensure_ascii=False, indent=2))
        return

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
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
