from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import importlib.util
from pathlib import Path
from typing import Optional, Literal

from rich.console import Console

from ..config import settings


console = Console()


Backend = Literal["auto", "none", "qwen2_vl", "qwen3_vl", "llava", "phi3_vision", "g4f"]


@dataclass
class VLMResult:
    """Результат работы VL-модели по изображению страницы/фигуры."""
    caption: str
    extracted_tables_md: Optional[str] = None
    extracted_equations_md: Optional[str] = None


def _require(pkg: str) -> None:
    raise RuntimeError(
        f"Для VLM backend нужна зависимость '{pkg}'.\n"
        "Установите extras: pip install -e '.[mm]'\n"
        "Или отключите VLM: export VLM_BACKEND=none"
    )


@lru_cache(maxsize=1)
def _load_transformers_vlm(model_id: str):
    """Load and cache a HF vision-language model.

    Qwen2.5-VL имеет собственный класс в Transformers, поэтому для него
    используем официальный путь через `Qwen2_5_VLForConditionalGeneration`.
    Для остальных backends оставляем AutoModelForVision2Seq fallback.
    """
    try:
        import torch  # type: ignore
        from transformers import AutoProcessor  # type: ignore
    except Exception:
        _require("transformers/torch")

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    if "Qwen/Qwen3-VL" in model_id or "Qwen3-VL" in model_id:
        try:
            from transformers import Qwen3VLForConditionalGeneration  # type: ignore
        except Exception:
            _require("transformers>=4.57.0 with Qwen3-VL support")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        return processor, model, "qwen3_vl"

    if "Qwen/Qwen2.5-VL" in model_id or "Qwen2.5-VL" in model_id:
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore
        except Exception:
            _require("transformers>=Qwen2.5-VL support")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        return processor, model, "qwen2_5_vl"

    try:
        from transformers import AutoModelForVision2Seq  # type: ignore
    except Exception:
        _require("transformers/torch")

    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    return processor, model, "generic"


def _has_g4f() -> bool:
    return importlib.util.find_spec("g4f") is not None


def _has_local_vlm_stack() -> bool:
    return all(importlib.util.find_spec(pkg) is not None for pkg in ("torch", "transformers", "PIL"))


def _default_model_id_for_backend(backend: Backend) -> str:
    if backend == "g4f":
        return str(getattr(settings, "task2_default_g4f_model", "auto") or "auto")
    return str(getattr(settings, "task2_default_local_vlm_model", "Qwen/Qwen2.5-VL-3B-Instruct") or "Qwen/Qwen2.5-VL-3B-Instruct")


def _resolve_model_id_for_backend(backend: Backend, model_id: Optional[str]) -> str:
    requested = str(model_id or getattr(settings, "vlm_model_id", "") or "").strip()
    local_default = str(getattr(settings, "task2_default_local_vlm_model", "Qwen/Qwen2.5-VL-3B-Instruct") or "Qwen/Qwen2.5-VL-3B-Instruct")
    g4f_default = str(getattr(settings, "task2_default_g4f_model", "auto") or "auto")

    if backend == "g4f":
        if not requested or requested.lower() == "auto":
            return g4f_default
        if requested.startswith("Qwen/") or "Qwen2.5-VL" in requested or "Qwen3-VL" in requested:
            return g4f_default
        return requested

    if not requested or requested.lower() == "auto":
        return local_default
    return requested


def _resolve_backend(backend: Optional[Backend], model_id: Optional[str]) -> Backend:
    requested = str(backend or getattr(settings, "vlm_backend", "none") or "none").strip().lower()

    if requested == "auto":
        if model_id and ("Qwen/Qwen3-VL" in model_id or "Qwen3-VL" in model_id):
            return "qwen3_vl" if _has_local_vlm_stack() else ("g4f" if _has_g4f() else "none")
        if _has_local_vlm_stack():
            return "qwen2_vl"
        if _has_g4f():
            return "g4f"
        return "none"

    if requested == "g4f" and not _has_g4f():
        if _has_local_vlm_stack():
            console.print("[yellow]g4f не установлен; переключаю VLM на локальный Transformers backend.[/yellow]")
            return "qwen2_vl"
        console.print("[yellow]g4f не установлен; продолжу без VLM-captioning.[/yellow]")
        return "none"

    if requested in {"qwen2_vl", "qwen3_vl", "llava", "phi3_vision"} and not _has_local_vlm_stack():
        if requested != "none":
            console.print("[yellow]Локальный VLM-стек недоступен; продолжу без VLM-captioning.[/yellow]")
        return "none"

    return requested  # type: ignore[return-value]


def _describe_image_g4f(image_path: Path, prompt: str, model_id: str) -> VLMResult:
    try:
        import base64
        from g4f.client import Client as G4FClient  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Для VLM backend='g4f' нужен пакет g4f (pip install -U g4f[all] или pip install -e '.[g4f]').") from e

    client = G4FClient(api_key=getattr(settings, "g4f_api_key", None) or None)
    providers = None
    raw_providers = getattr(settings, "g4f_providers", None)
    if raw_providers:
        providers = [p.strip() for p in str(raw_providers).split(",") if p.strip()]
        if len(providers) == 1:
            providers = providers[0]

    mime = "image/png"
    suffix = image_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"

    data_url = f"data:{mime};base64," + base64.b64encode(image_path.read_bytes()).decode("ascii")

    text = None
    errors = []

    try:
        resp = client.chat.completions.create(
            model=model_id,
            provider=providers,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        )
        text = resp.choices[0].message.content if getattr(resp, "choices", None) else None
    except Exception as e:
        errors.append(f"openai_style={type(e).__name__}: {e}")

    if not text:
        try:
            resp = client.chat.completions.create(
                model=model_id,
                provider=providers,
                messages=[{"role": "user", "content": prompt}],
                images=[[data_url, image_path.name]],
            )
            text = resp.choices[0].message.content if getattr(resp, "choices", None) else None
        except Exception as e:
            errors.append(f"images_arg={type(e).__name__}: {e}")

    if not text:
        raise RuntimeError("; ".join(errors) or "g4f multimodal call returned empty response")

    return VLMResult(caption=str(text).strip())


def _describe_image_qwen(image_path: Path, prompt: str, model_id: str, max_new_tokens: int) -> VLMResult:
    try:
        import torch  # type: ignore
        from PIL import Image  # type: ignore
    except Exception:
        _require("torch/pillow")

    processor, model, family = _load_transformers_vlm(model_id)
    img = Image.open(image_path).convert("RGB")
    full_prompt = (
        "Ты — научный ассистент. "
        "1) Дай краткую подпись к изображению (1-3 предложения). "
        "2) Если на изображении есть таблицы — извлеки их в Markdown. "
        "3) Если есть формулы — перепиши их в LaTeX. "
        f"\n\nЗадача/контекст: {prompt}"
    )

    if family in {"qwen2_5_vl", "qwen3_vl"}:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": full_prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        prompt_len = inputs["input_ids"].shape[1]
        trimmed = [out_ids[prompt_len:] for out_ids in generated_ids]
        raw_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    else:
        inputs = processor(images=img, text=full_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens)
        raw_text = processor.batch_decode(out, skip_special_tokens=True)[0].strip()

    caption = raw_text
    tables_md = None
    equations_md = None
    m = caption.split("TABLES:", 1)
    if len(m) == 2:
        caption, rest = m[0].strip(), m[1].strip()
        m2 = rest.split("EQUATIONS:", 1)
        if len(m2) == 2:
            tables_md, equations_md = m2[0].strip(), m2[1].strip()
        else:
            tables_md = rest

    return VLMResult(caption=caption, extracted_tables_md=tables_md, extracted_equations_md=equations_md)


@contextmanager
def temporary_vlm_selection(*, vlm_backend: Optional[str] = None, vlm_model_id: Optional[str] = None):
    """Temporarily override VLM routing for notebook/CLI execution."""

    prev_backend = getattr(settings, "vlm_backend", "none")
    prev_model = getattr(settings, "vlm_model_id", "")
    try:
        if vlm_backend and str(vlm_backend).strip():
            settings.vlm_backend = str(vlm_backend).strip()
        if vlm_model_id and str(vlm_model_id).strip():
            settings.vlm_model_id = str(vlm_model_id).strip()
        yield
    finally:
        settings.vlm_backend = prev_backend
        settings.vlm_model_id = prev_model


def describe_image(
    image_path: Path,
    prompt: str,
    backend: Optional[Backend] = None,
    model_id: Optional[str] = None,
    max_new_tokens: int = 512,
) -> VLMResult:
    """Описывает изображение (страница PDF / figure / table) через VL-модель.

    Никогда не роняет общий ingest из-за отсутствия опционального VLM backend.
    """
    requested_model_id = model_id or settings.vlm_model_id  # type: ignore[attr-defined]
    effective_backend = _resolve_backend(backend or settings.vlm_backend, requested_model_id)  # type: ignore[attr-defined]
    effective_model_id = _resolve_model_id_for_backend(effective_backend, requested_model_id)

    if effective_backend == "none":
        return VLMResult(caption="")

    try:
        if effective_backend == "g4f":
            return _describe_image_g4f(image_path=image_path, prompt=prompt, model_id=effective_model_id)

        return _describe_image_qwen(
            image_path=image_path,
            prompt=prompt,
            model_id=effective_model_id,
            max_new_tokens=max_new_tokens,
        )
    except Exception as e:
        console.print(
            f"[yellow]VLM warning for {image_path.name} ({effective_backend}): {type(e).__name__}: {e}. "
            "Продолжаю без caption/tables/equations для этой страницы.[/yellow]"
        )
        return VLMResult(caption="")
