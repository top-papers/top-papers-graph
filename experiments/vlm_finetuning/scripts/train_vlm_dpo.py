#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
from pathlib import Path
from typing import Any

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


def _truthy_env(name: str, default: str = '0') -> bool:
    return os.environ.get(name, default).lower() in {'1', 'true', 'yes', 'on'}


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
        bf16=args.bf16,
        fp16=args.fp16,
        max_length=args.max_length,
        save_strategy='steps',
        eval_strategy='steps' if eval_ds is not None else 'no',
        logging_strategy='steps',
        beta=args.beta,
        loss_type=args.loss_type,
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
    run_config['train_examples'] = len(train_ds)
    run_config['eval_examples'] = len(eval_ds) if eval_ds is not None else 0
    run_config['best_checkpoint_selection_enabled'] = bool(args.load_best_model_at_end and eval_ds is not None)
    run_config['native_best_checkpoint_reload_enabled'] = native_best_reload_enabled
    run_config['best_checkpoint_selection_mode'] = 'native_load' if native_best_reload_enabled else ('safe_checkpoint_file_copy' if args.load_best_model_at_end and eval_ds is not None else 'disabled')
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


if __name__ == '__main__':
    main()
