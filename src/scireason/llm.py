from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict, List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from rich.console import Console

from .config import settings

console = Console()

try:
    import litellm
except Exception:  # pragma: no cover
    litellm = None


@lru_cache(maxsize=1)
def _st_model(model_name: str) -> Any:
    """Cache sentence-transformers model across calls.

    This keeps the embedding path usable in interactive sessions and when indexing many chunks.
    """
    from sentence_transformers import SentenceTransformer  # type: ignore

    return SentenceTransformer(model_name)


def _litellm_kwargs(provider: str | None = None) -> dict:
    """Provider-specific kwargs for LiteLLM.

    LiteLLM reads API keys from environment variables. For local Ollama we only need `api_base`.
    """
    prov = (provider or settings.llm_provider).lower()
    if prov == "ollama":
        return {"api_base": settings.ollama_base_url}
    return {}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def chat_json(system: str, user: str, schema_hint: str, temperature: float = 0.2) -> dict:
    """Запрос к LLM с требованием вернуть JSON.
    schema_hint — строка-подсказка (например, pydantic model JSON schema или краткое описание).
    """
    if litellm is None:
        raise RuntimeError("LiteLLM не установлен. Установите зависимости: pip install -e '.[dev]'")

    messages = [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": f"""{user.strip()}

Верни ТОЛЬКО валидный JSON (без markdown). Схема/ожидания:
{schema_hint}
"""},
    ]

    resp = litellm.completion(
        model=f"{settings.llm_provider}/{settings.llm_model}" if "/" not in settings.llm_model else settings.llm_model,
        messages=messages,
        temperature=temperature,
        **_litellm_kwargs(settings.llm_provider),
    )

    text = resp["choices"][0]["message"]["content"]
    try:
        return json.loads(text)
    except Exception:
        # иногда модель добавляет лишний текст — попробуем вытащить первый JSON-блок
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def embed(texts: List[str]) -> List[List[float]]:
    """Эмбеддинги.

    Поддерживаем два сценария:
    1) `embed_provider=sentence-transformers` (по умолчанию, работает без API ключей)
    2) `embed_provider=<litellm provider>` — через `litellm.embedding()` (OpenAI/Anthropic/Ollama/vLLM и т.п.)
    """

    # ---- 1) Local sentence-transformers (default) ----
    if getattr(settings, "embed_provider", "sentence-transformers") == "sentence-transformers":
        try:
            model = _st_model(getattr(settings, "embed_model", "sentence-transformers/all-MiniLM-L6-v2"))
            return model.encode(list(texts), normalize_embeddings=True).tolist()
        except Exception as e:
            # fall through to LiteLLM attempt
            console.print(
                f"[yellow]sentence-transformers embedding не сработал: {e}. Пытаюсь LiteLLM...[/yellow]"
            )

    # ---- 2) LiteLLM embeddings ----
    if litellm is None:
        raise RuntimeError(
            "LiteLLM не установлен, а embed_provider != 'sentence-transformers'. "
            "Установите зависимости: pip install -e '.[dev]' или верните embed_provider=sentence-transformers"
        )

    provider = getattr(settings, "embed_provider", settings.llm_provider)
    model_name = getattr(settings, "embed_model", settings.llm_model)
    model = f"{provider}/{model_name}" if "/" not in model_name else model_name

    resp = litellm.embedding(
        model=model,
        input=texts,
        **_litellm_kwargs(provider),
    )
    return [d["embedding"] for d in resp["data"]]
