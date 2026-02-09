from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Literal

from ..config import settings


Backend = Literal["none", "qwen2_vl", "llava", "phi3_vision"]


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

    Most multimodal pipelines call `describe_image()` once per page; caching avoids reloading
    multi‑GB weights for every page.
    """
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
    return processor, model


def describe_image(
    image_path: Path,
    prompt: str,
    backend: Optional[Backend] = None,
    model_id: Optional[str] = None,
    max_new_tokens: int = 512,
) -> VLMResult:
    """Описывает изображение (страница PDF / figure / table) через VL-модель.

    По умолчанию использует настройки из .env:
      - VLM_BACKEND
      - VLM_MODEL_ID

    Замечание: здесь намеренно сделан тонкий слой-адаптер.
    В курсовом проекте участники могут заменить/усилить реализацию (vLLM, llama.cpp, etc.).
    """
    backend = backend or settings.vlm_backend  # type: ignore[attr-defined]
    model_id = model_id or settings.vlm_model_id  # type: ignore[attr-defined]

    if backend == "none":
        return VLMResult(caption="")

    # transformers-based backends
    try:
        import torch  # type: ignore
        from PIL import Image  # type: ignore
    except Exception:
        _require("torch/pillow")

    # NOTE: Это «базовая» реализация под CPU/GPU.
    # На практике для 7B моделей лучше использовать GPU, quantization (bitsandbytes) или vLLM.
    processor, model = _load_transformers_vlm(model_id)

    img = Image.open(image_path).convert("RGB")

    # Унифицированный промпт: просим дать краткую подпись + извлечь таблицы/формулы при наличии.
    full_prompt = (
        "Ты — научный ассистент. "
        "1) Дай краткую подпись к изображению (1-3 предложения). "
        "2) Если на изображении есть таблицы — извлеки их в Markdown. "
        "3) Если есть формулы — перепиши их в LaTeX. "
        f"\n\nЗадача/контекст: {prompt}"
    )

    inputs = processor(images=img, text=full_prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)

    text = processor.batch_decode(out, skip_special_tokens=True)[0].strip()

    # Очень простой парсер (для MVP): ожидаем блоки с маркерами.
    caption = text
    tables_md = None
    equations_md = None

    # Если участники будут улучшать — лучше вернуться к structured outputs через JSON schema.
    m = caption.split("TABLES:", 1)
    if len(m) == 2:
        caption, rest = m[0].strip(), m[1].strip()
        # EQUATIONS может идти после таблиц
        m2 = rest.split("EQUATIONS:", 1)
        if len(m2) == 2:
            tables_md, equations_md = m2[0].strip(), m2[1].strip()
        else:
            tables_md = rest

    return VLMResult(caption=caption, extracted_tables_md=tables_md, extracted_equations_md=equations_md)
