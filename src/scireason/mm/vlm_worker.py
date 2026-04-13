from __future__ import annotations

import gc
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_PRELOADED_MODELS: set[str] = set()


def _emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _get_settings():
    try:
        from scireason.config import settings  # type: ignore
        return settings
    except Exception:
        class _Fallback:
            vlm_min_pixels = 256 * 28 * 28
            vlm_max_pixels = 1280 * 28 * 28
        return _Fallback()


def _disable_hf_progress_bars() -> None:
    try:
        from huggingface_hub.utils import disable_progress_bars  # type: ignore
        disable_progress_bars()
    except Exception:
        pass


def _should_retry_on_cpu(exc: Exception) -> bool:
    msg = str(exc).lower()
    hints = (
        'libnvrtc-builtins',
        'nvrtc',
        'cuda error',
        'no kernel image is available',
        'cublas',
        'cutlass',
    )
    return any(h in msg for h in hints)


def _best_attn_implementation() -> str | None:
    try:
        import torch  # type: ignore
        if not torch.cuda.is_available():
            return None
    except Exception:
        return None
    try:
        import flash_attn  # type: ignore  # noqa: F401
        return 'flash_attention_2'
    except Exception:
        return 'sdpa'


def _processor_kwargs() -> dict[str, Any]:
    settings = _get_settings()
    kwargs: dict[str, Any] = {}
    min_pixels = int(getattr(settings, 'vlm_min_pixels', 0) or 0)
    max_pixels = int(getattr(settings, 'vlm_max_pixels', 0) or 0)
    if min_pixels > 0:
        kwargs['min_pixels'] = min_pixels
    if max_pixels > 0:
        kwargs['max_pixels'] = max_pixels
    return kwargs


def _model_kwargs(device_mode: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        'torch_dtype': 'auto',
        'low_cpu_mem_usage': True,
    }
    if device_mode != 'cpu':
        kwargs['device_map'] = 'auto'
        attn_impl = _best_attn_implementation()
        if attn_impl:
            kwargs['attn_implementation'] = attn_impl
    return kwargs


@lru_cache(maxsize=4)
def _load_transformers_vlm(model_id: str, device_mode: str = 'auto'):
    _disable_hf_progress_bars()
    try:
        from transformers import AutoProcessor  # type: ignore
    except Exception as exc:
        raise RuntimeError("Для VLM backend нужна зависимость 'transformers/torch'.") from exc

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, **_processor_kwargs())
    kwargs = _model_kwargs(device_mode)

    if 'Qwen/Qwen3-VL' in model_id or 'Qwen3-VL' in model_id:
        from transformers import Qwen3VLForConditionalGeneration  # type: ignore
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, trust_remote_code=True, **kwargs)
        model.eval()
        return processor, model, 'qwen3_vl'

    if 'Qwen/Qwen2.5-VL' in model_id or 'Qwen2.5-VL' in model_id:
        from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore
        import qwen_vl_utils  # noqa: F401
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, trust_remote_code=True, **kwargs)
        model.eval()
        return processor, model, 'qwen2_5_vl'

    try:
        from transformers import AutoModelForImageTextToText as _GenericImageTextModel  # type: ignore
    except Exception:
        from transformers import AutoModelForVision2Seq as _GenericImageTextModel  # type: ignore
    model = _GenericImageTextModel.from_pretrained(model_id, trust_remote_code=True, **kwargs)
    model.eval()
    return processor, model, 'generic'


def _preload_model(model_id: str) -> dict[str, Any]:
    processor, model, family = _load_transformers_vlm(model_id, 'auto')
    device = _model_device(model)
    device_text = str(device) if device is not None else 'unknown'
    _PRELOADED_MODELS.add(model_id)
    return {
        'model_id': model_id,
        'family': family,
        'device': device_text,
        'processor': processor.__class__.__name__,
    }


def _model_device(model):
    try:
        return next(model.parameters()).device
    except Exception:
        return getattr(model, 'device', None)


def _prepare_qwen_inputs(processor, image_path: Path, prompt: str):
    from qwen_vl_utils import process_vision_info  # type: ignore

    image_uri = image_path.resolve().as_uri()
    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image_uri},
                {'type': 'text', 'text': prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors='pt',
    )


def _describe(image_path: Path, prompt: str, model_id: str, max_new_tokens: int, device_mode: str = 'auto') -> dict:
    try:
        import torch  # type: ignore
        from PIL import Image  # type: ignore
    except Exception as exc:
        raise RuntimeError("Для VLM backend нужна зависимость 'transformers/torch'.") from exc

    processor, model, family = _load_transformers_vlm(model_id, device_mode)
    full_prompt = (
        'Ты — научный ассистент. '
        '1) Дай краткую подпись к изображению (1-3 предложения). '
        '2) Если на изображении есть таблицы — извлеки их в Markdown. '
        '3) Если есть формулы — перепиши их в LaTeX. '
        f"\n\nЗадача/контекст: {prompt}"
    )

    try:
        if family in {'qwen2_5_vl', 'qwen3_vl'}:
            inputs = _prepare_qwen_inputs(processor, image_path, full_prompt)
        else:
            with Image.open(image_path) as opened_img:
                img = opened_img.convert('RGB')
                inputs = processor(images=img, text=full_prompt, return_tensors='pt')

        device = _model_device(model)
        if device is not None:
            try:
                inputs = {k: v.to(device) for k, v in inputs.items()}
            except Exception:
                pass

        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

        if family in {'qwen2_5_vl', 'qwen3_vl'} and 'input_ids' in inputs:
            prompt_len = int(inputs['input_ids'].shape[1])
            trimmed = [out_ids[prompt_len:] for out_ids in generated_ids]
            raw_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        else:
            raw_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    except Exception as exc:
        if device_mode != 'cpu' and _should_retry_on_cpu(exc):
            _load_transformers_vlm.cache_clear()
            return _describe(image_path=image_path, prompt=prompt, model_id=model_id, max_new_tokens=max_new_tokens, device_mode='cpu')
        raise
    finally:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    caption = raw_text
    tables_md = None
    equations_md = None
    parts = caption.split('TABLES:', 1)
    if len(parts) == 2:
        caption, rest = parts[0].strip(), parts[1].strip()
        parts2 = rest.split('EQUATIONS:', 1)
        if len(parts2) == 2:
            tables_md, equations_md = parts2[0].strip(), parts2[1].strip()
        else:
            tables_md = rest
    return {
        'caption': caption,
        'extracted_tables_md': tables_md,
        'extracted_equations_md': equations_md,
    }


def main() -> int:
    default_model = (sys.argv[1] if len(sys.argv) > 1 else '').strip()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception as exc:
            _emit({'ok': False, 'error': f'bad_request_json: {type(exc).__name__}: {exc}'})
            continue
        cmd = str(payload.get('cmd') or 'describe')
        if cmd == 'shutdown':
            _emit({'ok': True, 'status': 'bye'})
            return 0
        if cmd == 'ping':
            _emit({'ok': True, 'status': 'pong'})
            continue
        model_id = str(payload.get('model_id') or default_model or 'Qwen/Qwen2.5-VL-3B-Instruct')
        try:
            if cmd == 'preload':
                meta = _preload_model(model_id)
                _emit({'ok': True, 'status': 'ready', **meta})
                continue
            result = _describe(
                image_path=Path(str(payload.get('image_path') or '')),
                prompt=str(payload.get('prompt') or ''),
                model_id=model_id,
                max_new_tokens=int(payload.get('max_new_tokens') or 256),
            )
            _emit({'ok': True, **result})
        except Exception as exc:
            _emit({'ok': False, 'error': f'{type(exc).__name__}: {exc}'})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
