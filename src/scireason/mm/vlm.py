from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from ..config import settings


Backend = Literal["none", "qwen2_vl", "llava", "phi3_vision", "g4f"]


@dataclass
class VLMResult:
    """Result of a vision-language model pass over a scientific page / figure / table."""

    caption: str
    extracted_tables_md: Optional[str] = None
    extracted_equations_md: Optional[str] = None


PROMPT_TEMPLATE = (
    "Ты — научный ассистент по мультимодальному анализу PDF. "
    "Опиши изображение как доказательство из статьи. "
    "Выдели графики, оси, тенденции, подписи, таблицы и формулы. "
    "Ответ верни СТРОГО в трёх блоках:\n"
    "CAPTION: <краткое описание и смысл>\n"
    "TABLES: <markdown таблица или пусто>\n"
    "EQUATIONS: <LaTeX или пусто>\n\n"
    "Контекст задачи: {prompt}"
)


def _require(pkg: str) -> None:
    raise RuntimeError(
        f"Для VLM backend нужна зависимость '{pkg}'.\n"
        "Установите extras: pip install -e '.[mm,g4f]'\n"
        "Или отключите VLM: export VLM_BACKEND=none"
    )


@lru_cache(maxsize=1)
def _load_qwen2_vl(model_id: str):
    try:
        import torch  # type: ignore
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration  # type: ignore
    except Exception:
        _require("transformers/torch")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return processor, model, torch


@lru_cache(maxsize=1)
def _load_generic_transformers_vlm(model_id: str):
    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq  # type: ignore
        import torch  # type: ignore
    except Exception:
        _require("transformers/torch")

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    return processor, model, torch


def _parse_vlm_text(text: str) -> VLMResult:
    raw = (text or "").strip()
    if not raw:
        return VLMResult(caption="")

    caption = raw
    tables_md = None
    equations_md = None

    # Preferred format: CAPTION / TABLES / EQUATIONS.
    upper = raw.upper()
    if "CAPTION:" in upper:
        try:
            work = raw
            if "TABLES:" in work:
                before_tables, after_tables = work.split("TABLES:", 1)
            else:
                before_tables, after_tables = work, ""
            if "CAPTION:" in before_tables:
                caption = before_tables.split("CAPTION:", 1)[1].strip()
            else:
                caption = before_tables.strip()
            if after_tables:
                if "EQUATIONS:" in after_tables:
                    tables_part, eq_part = after_tables.split("EQUATIONS:", 1)
                    tables_md = tables_part.strip() or None
                    equations_md = eq_part.strip() or None
                else:
                    tables_md = after_tables.strip() or None
        except Exception:
            caption = raw
    elif "TABLES:" in raw:
        parts = raw.split("TABLES:", 1)
        caption = parts[0].strip()
        rest = parts[1].strip()
        if "EQUATIONS:" in rest:
            tpart, epart = rest.split("EQUATIONS:", 1)
            tables_md = tpart.strip() or None
            equations_md = epart.strip() or None
        else:
            tables_md = rest or None

    if tables_md:
        low = tables_md.lower().strip()
        if low in {"none", "n/a", "нет", "пусто"}:
            tables_md = None
    if equations_md:
        low = equations_md.lower().strip()
        if low in {"none", "n/a", "нет", "пусто"}:
            equations_md = None

    return VLMResult(caption=caption.strip(), extracted_tables_md=tables_md, extracted_equations_md=equations_md)


def _describe_image_qwen2_vl(image_path: Path, full_prompt: str, model_id: str, max_new_tokens: int) -> VLMResult:
    try:
        from PIL import Image  # type: ignore
    except Exception:
        _require("pillow")

    processor, model, torch = _load_qwen2_vl(model_id)
    img = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": full_prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt")
    if torch.cuda.is_available():
        moved = {}
        for k, v in inputs.items():
            try:
                moved[k] = v.to(model.device)
            except Exception:
                moved[k] = v
        inputs = moved

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    input_ids = inputs.get("input_ids")
    if input_ids is not None:
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)]
        text_out = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    else:
        text_out = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return _parse_vlm_text(text_out)


def _describe_image_generic(image_path: Path, full_prompt: str, model_id: str, max_new_tokens: int) -> VLMResult:
    try:
        from PIL import Image  # type: ignore
    except Exception:
        _require("pillow")

    processor, model, torch = _load_generic_transformers_vlm(model_id)
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, text=full_prompt, return_tensors="pt")
    if torch.cuda.is_available():
        moved = {}
        for k, v in inputs.items():
            try:
                moved[k] = v.to(model.device)
            except Exception:
                moved[k] = v
        inputs = moved

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    text_out = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
    return _parse_vlm_text(text_out)


def _describe_image_g4f(image_path: Path, full_prompt: str, model_id: str) -> VLMResult:
    from ..llm import _g4f_client, _g4f_model_candidates  # type: ignore

    client = _g4f_client()
    model_req = (model_id or settings.llm_model or "auto").strip() or "auto"
    candidates = _g4f_model_candidates() if model_req.lower() in {"", "auto"} else [model_req]
    if not candidates:
        candidates = ["gpt-4o-mini", "gpt-4o"]

    last_err: Exception | None = None
    for name in candidates[:8]:
        try:
            with image_path.open("rb") as fp:
                resp = client.chat.completions.create(
                    model=name,
                    messages=[{"role": "user", "content": full_prompt}],
                    images=[[fp.read(), image_path.name]],
                )
            text_out = (resp.choices[0].message.content or "").strip()
            if text_out:
                return _parse_vlm_text(text_out)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"g4f vision failed (last_err={last_err})")


def describe_image(
    image_path: Path,
    prompt: str,
    backend: Optional[Backend] = None,
    model_id: Optional[str] = None,
    max_new_tokens: int = 512,
) -> VLMResult:
    """Describe a scientific image / page via a local or remote VLM.

    Supported backends:
      - qwen2_vl: local Hugging Face Qwen2-VL via transformers
      - g4f: remote/community provider routed through g4f vision API
      - llava / phi3_vision: generic transformers Vision2Seq fallback
    """
    backend = backend or settings.vlm_backend  # type: ignore[attr-defined]
    model_id = model_id or settings.vlm_model_id  # type: ignore[attr-defined]

    if backend == "none":
        return VLMResult(caption="")

    full_prompt = PROMPT_TEMPLATE.format(prompt=prompt)

    if backend == "g4f":
        return _describe_image_g4f(image_path=image_path, full_prompt=full_prompt, model_id=model_id)

    if backend == "qwen2_vl":
        try:
            return _describe_image_qwen2_vl(
                image_path=image_path,
                full_prompt=full_prompt,
                model_id=model_id,
                max_new_tokens=max_new_tokens,
            )
        except Exception:
            # If a specific Qwen2-VL loader fails, still try the generic transformers path.
            return _describe_image_generic(
                image_path=image_path,
                full_prompt=full_prompt,
                model_id=model_id,
                max_new_tokens=max_new_tokens,
            )

    return _describe_image_generic(
        image_path=image_path,
        full_prompt=full_prompt,
        model_id=model_id,
        max_new_tokens=max_new_tokens,
    )
