from __future__ import annotations

import gc
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional


_FORCE_CPU_MODEL_IDS: set[str] = set()


def _emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _error_text(exc: Exception) -> str:
    msg = str(exc) or repr(exc)
    return f"{type(exc).__name__}: {msg}"


def _env_device_override() -> str:
    raw = str(os.environ.get("SCIREASON_LOCAL_VLM_DEVICE", "auto") or "auto").strip().lower()
    if raw in {"cpu", "cuda", "auto"}:
        return raw
    return "auto"


def _preferred_device_mode(model_id: str) -> str:
    override = _env_device_override()
    if override in {"cpu", "cuda"}:
        return override
    if model_id in _FORCE_CPU_MODEL_IDS:
        return "cpu"
    try:
        import torch  # type: ignore
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _mark_model_cpu_only(model_id: str) -> None:
    if model_id:
        _FORCE_CPU_MODEL_IDS.add(model_id)


def _clear_model_cache() -> None:
    try:
        _load_transformers_vlm.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def _looks_like_broken_cuda_runtime(exc: Exception) -> bool:
    msg = _error_text(exc).lower()
    needles = [
        "libnvrtc-builtins",
        "nvrtc:",
        "nvrtc-builtins",
        "failed to open nvrtc-builtins",
        "failed to open libnvrtc-builtins",
        "cuda driver",
        "cuda error",
        "cuda runtime",
        "no kernel image is available",
    ]
    return any(token in msg for token in needles)


@lru_cache(maxsize=4)
def _load_transformers_vlm(model_id: str, device_mode: str = "auto"):
    try:
        import torch  # type: ignore
        from transformers import AutoProcessor  # type: ignore
    except Exception as exc:
        raise RuntimeError("Для VLM backend нужна зависимость 'transformers/torch'.") from exc

    effective_device = device_mode if device_mode in {"cpu", "cuda"} else _preferred_device_mode(model_id)
    use_cuda = effective_device == "cuda" and bool(torch.cuda.is_available())
    dtype = torch.float16 if use_cuda else torch.float32
    device_map = "auto" if use_cuda else None

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    common_kwargs = dict(
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    # HF docs for Qwen2.5-VL show sdpa as a supported attention implementation.
    # It is a safer default than relying on whatever the environment wires in.
    common_kwargs["attn_implementation"] = "sdpa"

    if "Qwen/Qwen3-VL" in model_id or "Qwen3-VL" in model_id:
        from transformers import Qwen3VLForConditionalGeneration  # type: ignore
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **common_kwargs)
        return processor, model, "qwen3_vl", effective_device

    if "Qwen/Qwen2.5-VL" in model_id or "Qwen2.5-VL" in model_id:
        from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore
        import qwen_vl_utils  # noqa: F401
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **common_kwargs)
        return processor, model, "qwen2_5_vl", effective_device

    from transformers import AutoModelForVision2Seq  # type: ignore
    model = AutoModelForVision2Seq.from_pretrained(model_id, **common_kwargs)
    return processor, model, "generic", effective_device


def _move_inputs_if_needed(inputs, model, use_cuda: bool):
    if not use_cuda:
        return inputs
    try:
        target_device = getattr(model, "device", None)
        if target_device is None:
            target_device = next(model.parameters()).device
        return {k: v.to(target_device) for k, v in inputs.items()}
    except Exception:
        return inputs


def _run_once(image_path: Path, prompt: str, model_id: str, max_new_tokens: int, device_mode: str) -> dict:
    try:
        import torch  # type: ignore
        from PIL import Image  # type: ignore
    except Exception as exc:
        raise RuntimeError("Для VLM backend нужна зависимость 'transformers/torch'.") from exc

    processor, model, family, actual_device = _load_transformers_vlm(model_id, device_mode)
    use_cuda = actual_device == "cuda" and bool(torch.cuda.is_available())
    full_prompt = (
        "Ты — научный ассистент. "
        "1) Дай краткую подпись к изображению (1-3 предложения). "
        "2) Если на изображении есть таблицы — извлеки их в Markdown. "
        "3) Если есть формулы — перепиши их в LaTeX. "
        f"\n\nЗадача/контекст: {prompt}"
    )

    raw_text = ""
    with Image.open(image_path) as opened_img:
        img = opened_img.convert("RGB")
        try:
            if family in {"qwen2_5_vl", "qwen3_vl"}:
                messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": full_prompt}]}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt")
                inputs = _move_inputs_if_needed(inputs, model, use_cuda)
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
                prompt_len = inputs["input_ids"].shape[1]
                trimmed = [out_ids[prompt_len:] for out_ids in generated_ids]
                raw_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
            else:
                inputs = processor(images=img, text=full_prompt, return_tensors="pt")
                inputs = _move_inputs_if_needed(inputs, model, use_cuda)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
                raw_text = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        finally:
            try:
                img.close()
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    caption = raw_text
    tables_md = None
    equations_md = None
    parts = caption.split("TABLES:", 1)
    if len(parts) == 2:
        caption, rest = parts[0].strip(), parts[1].strip()
        parts2 = rest.split("EQUATIONS:", 1)
        if len(parts2) == 2:
            tables_md, equations_md = parts2[0].strip(), parts2[1].strip()
        else:
            tables_md = rest
    return {
        "caption": caption,
        "extracted_tables_md": tables_md,
        "extracted_equations_md": equations_md,
    }


def _describe(image_path: Path, prompt: str, model_id: str, max_new_tokens: int) -> dict:
    first_mode = _preferred_device_mode(model_id)
    try:
        return _run_once(image_path=image_path, prompt=prompt, model_id=model_id, max_new_tokens=max_new_tokens, device_mode=first_mode)
    except Exception as exc:
        if first_mode == "cuda" and _looks_like_broken_cuda_runtime(exc):
            _mark_model_cpu_only(model_id)
            _clear_model_cache()
            return _run_once(image_path=image_path, prompt=prompt, model_id=model_id, max_new_tokens=max_new_tokens, device_mode="cpu")
        raise


def main() -> int:
    default_model = (sys.argv[1] if len(sys.argv) > 1 else "").strip()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception as exc:
            _emit({"ok": False, "error": f"bad_request_json: {type(exc).__name__}: {exc}"})
            continue
        cmd = str(payload.get("cmd") or "describe")
        if cmd == "shutdown":
            _emit({"ok": True, "status": "bye"})
            return 0
        model_id = str(payload.get("model_id") or default_model or "Qwen/Qwen2.5-VL-3B-Instruct")
        try:
            result = _describe(
                image_path=Path(str(payload.get("image_path") or "")),
                prompt=str(payload.get("prompt") or ""),
                model_id=model_id,
                max_new_tokens=int(payload.get("max_new_tokens") or 512),
            )
            _emit({"ok": True, **result})
        except Exception as exc:
            _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
