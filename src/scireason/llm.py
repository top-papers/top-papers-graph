from __future__ import annotations

import hashlib
import json
import math
import re
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

try:
    from g4f.client import Client as G4FClient  # type: ignore
except Exception:  # pragma: no cover
    G4FClient = None


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


@lru_cache(maxsize=1)
def _g4f_client() -> Any:
    if G4FClient is None:  # pragma: no cover
        raise RuntimeError("g4f не установлен. Установите: pip install g4f")
    return G4FClient()


@lru_cache(maxsize=16)
def _warn_once(key: str) -> None:
    # Keyed warnings – cached so we don't spam the console inside loops.
    console.print(f"[yellow]{key}[/yellow]")


_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def _hash_embed_one(text: str, dim: int) -> List[float]:
    """Lightweight deterministic embedding via feature hashing.

    This is a pragmatic fallback when neither sentence-transformers nor a remote embedding
    provider is available. It keeps the pipeline runnable without extra services.
    """

    vec = [0.0] * dim
    tokens = _TOKEN_RE.findall((text or "").lower())
    for tok in tokens:
        h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        n = int.from_bytes(h, "big", signed=False)
        idx = n % dim
        sign = 1.0 if (n & 1) == 0 else -1.0
        vec[idx] += sign

    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def chat_json(system: str, user: str, schema_hint: str, temperature: float = 0.2) -> dict:
    """Запрос к LLM с требованием вернуть JSON.
    schema_hint — строка-подсказка (например, pydantic model JSON schema или краткое описание).
    """
    provider = (settings.llm_provider or "").lower()

    messages = [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": f"""{user.strip()}

Верни ТОЛЬКО валидный JSON (без markdown). Схема/ожидания:
{schema_hint}
"""},
    ]

    # ---- 1) g4f (default) ----
    if provider == "g4f":
        client = _g4f_client()
        resp = client.chat.completions.create(
            model=settings.llm_model or "auto",
            messages=messages,
        )
        text = resp.choices[0].message.content
    else:
        # ---- 2) LiteLLM ----
        if litellm is None:
            raise RuntimeError("LiteLLM не установлен. Установите зависимости: pip install -e '.[dev]'")

        resp = litellm.completion(
            model=f"{settings.llm_provider}/{settings.llm_model}"
            if "/" not in settings.llm_model
            else settings.llm_model,
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

    provider = getattr(settings, "embed_provider", "hash")

    # ---- 0) Hash embeddings (default; always available) ----
    if provider == "hash":
        dim = int(getattr(settings, "hash_embed_dim", 384) or 384)
        return [_hash_embed_one(t, dim) for t in texts]

    # ---- 1) Local sentence-transformers (optional) ----
    if provider == "sentence-transformers":
        try:
            model = _st_model(getattr(settings, "embed_model", "sentence-transformers/all-MiniLM-L6-v2"))
            return model.encode(list(texts), normalize_embeddings=True).tolist()
        except Exception as e:
            _warn_once(
                "sentence-transformers недоступен "
                f"({type(e).__name__}). Использую hash-эмбеддинги. "
                "Чтобы включить ST: pip install -e '.[embeddings]'"
            )
            dim = int(getattr(settings, "hash_embed_dim", 384) or 384)
            return [_hash_embed_one(t, dim) for t in texts]

    # ---- 2) LiteLLM embeddings ----
    if litellm is None:
        raise RuntimeError(
            "LiteLLM не установлен, а embed_provider != 'sentence-transformers'. "
            "Установите зависимости: pip install -e '.[dev]' или верните embed_provider=sentence-transformers"
        )

    # ---- 2) LiteLLM embeddings (remote / local servers) ----
    provider = getattr(settings, "embed_provider", settings.llm_provider)
    model_name = getattr(settings, "embed_model", settings.llm_model)
    model = f"{provider}/{model_name}" if "/" not in model_name else model_name

    if litellm is None:
        _warn_once("LiteLLM не установлен – использую hash-эмбеддинги.")
        dim = int(getattr(settings, "hash_embed_dim", 384) or 384)
        return [_hash_embed_one(t, dim) for t in texts]

    try:
        resp = litellm.embedding(
            model=model,
            input=texts,
            **_litellm_kwargs(provider),
        )
        return [d["embedding"] for d in resp["data"]]
    except Exception as e:
        _warn_once(
            f"LiteLLM embeddings недоступны ({type(e).__name__}). Использую hash-эмбеддинги."
        )
        dim = int(getattr(settings, "hash_embed_dim", 384) or 384)
        return [_hash_embed_one(t, dim) for t in texts]
