from __future__ import annotations

import hashlib
import json
import math
import os
import re
from functools import lru_cache
from typing import Any, List, Optional
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


def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        # Drop opening fence line (``` or ```json)
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        # Drop trailing fence
        t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _extract_first_json_block(text: str) -> str | None:
    """Return the first well-balanced JSON object/array substring, if any.

    Works even when the model returns extra prose around JSON.
    """

    s = _strip_code_fences(text)
    # Find the first opening brace/bracket.
    start = None
    for i, ch in enumerate(s):
        if ch in "[{":
            start = i
            break
    if start is None:
        return None

    stack: list[str] = []
    in_str = False
    esc = False
    for j in range(start, len(s)):
        ch = s[j]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch in "[{":
            stack.append(ch)
            continue

        if ch in "]}":
            if not stack:
                return None
            open_ch = stack.pop()
            if (open_ch == "[" and ch != "]") or (open_ch == "{" and ch != "}"):
                return None
            if not stack:
                return s[start : j + 1]

    return None


def _json_loads_best_effort(text: str) -> Any:
    """Parse JSON from an LLM response, tolerating common wrappers."""

    raw = _strip_code_fences(text)

    # 1) direct
    try:
        return json.loads(raw)
    except Exception:
        pass

    # 2) first balanced object/array
    block = _extract_first_json_block(raw)
    if block is not None:
        return json.loads(block)

    raise json.JSONDecodeError("No JSON object/array found", raw, 0)


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
    """Create a g4f client.

    By default we let g4f route per-model to its `best_provider` mapping.
    Users can still force a provider shortlist via `G4F_PROVIDERS` / settings.
    """

    if G4FClient is None:  # pragma: no cover
        raise RuntimeError("g4f не установлен. Установите: pip install g4f")

    api_key = (settings.g4f_api_key or os.getenv("G4F_API_KEY"))

    raw = (settings.g4f_providers or os.getenv("G4F_PROVIDERS") or "").strip()
    if raw:
        try:
            from g4f import Provider as P  # type: ignore
            from g4f.Provider import RetryProvider  # type: ignore

            names = [x.strip() for x in raw.split(",") if x.strip()]
            providers = [getattr(P, n) for n in names if hasattr(P, n)]
            if providers:
                return G4FClient(provider=RetryProvider(providers, shuffle=True), api_key=api_key)
        except Exception:
            # If RetryProvider path fails, fall back to plain client.
            pass

    # Default: no provider override; g4f will use the model registry routing.
    return G4FClient(api_key=api_key)


def _dedup_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


@lru_cache(maxsize=1)
def _g4f_model_candidates() -> list[str]:
    """Return a prioritized list of *text-capable* g4f model names.

    The source of truth is g4f's own model registry (g4f/models.py).
    We prefer the working model name list exposed there (Model.__all__ / _all_models).
    """

    try:
        from g4f import models as gm  # type: ignore
    except Exception:
        return []

    names: list[str] = []

    # 1) Preferred: Model.__all__() returns the working model name list (g4f/models.py).
    try:
        Model = getattr(gm, "Model", None)
        if Model is not None and hasattr(Model, "__all__"):
            cand = Model.__all__()  # type: ignore
            if isinstance(cand, (list, tuple)):
                names = list(cand)
    except Exception:
        pass

    # 2) Fallbacks used by g4f itself
    if not names:
        try:
            names = list(getattr(gm, "_all_models", []) or [])
        except Exception:
            names = []

    if not names:
        try:
            names = list(getattr(gm, "__models__", {}).keys())
        except Exception:
            names = []

    names = _dedup_keep_order([str(x) for x in names if str(x).strip()])

    # Filter out non-chat models when registry provides mixed capabilities.
    try:
        from g4f.models import ModelRegistry  # type: ignore
        ImageModel = getattr(gm, "ImageModel", None)
        AudioModel = getattr(gm, "AudioModel", None)
        VideoModel = getattr(gm, "VideoModel", None)

        filtered: list[str] = []
        for n in names:
            try:
                model_obj = ModelRegistry.get(n)
            except Exception:
                model_obj = None

            if model_obj is None:
                continue
            if ImageModel is not None and isinstance(model_obj, ImageModel):
                continue
            if AudioModel is not None and isinstance(model_obj, AudioModel):
                continue
            if VideoModel is not None and isinstance(model_obj, VideoModel):
                continue
            filtered.append(n)
        names = filtered
    except Exception:
        # If filtering isn't possible (older g4f), keep the raw list.
        pass

    # User preference: a comma-separated list that will be tried first.
    prefer_raw = (os.getenv("G4F_MODEL_PREFER") or os.getenv("G4F_MODEL_PREFERENCES") or "").strip()
    prefer = [x.strip() for x in prefer_raw.split(",") if x.strip()]
    preferred = [m for m in prefer if m in names]

    # Reasonable defaults (only if present in the registry list).
    if not preferred:
        default_pref = [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4",
            "deepseek-v3",
            "deepseek-r1",
            "qwen-2.5-72b",
            "llama-3.3-70b",
        ]
        preferred = [m for m in default_pref if m in names]

    rest = [m for m in names if m not in preferred]

    # Cap the candidate list to avoid spending too long on dead providers.
    try:
        cap = int(os.getenv("G4F_AUTO_MAX_MODELS") or 25)
    except Exception:
        cap = 25

    return (preferred + rest)[: max(1, cap)]


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
def chat_json(system: str, user: str, schema_hint: str, temperature: float = 0.2) -> Any:
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

        model_req = (settings.llm_model or "auto").strip() or "auto"

        if model_req.lower() in {"auto", ""}:
            # Important: we source candidate models from g4f's own registry (g4f/models.py).
            model_candidates = _g4f_model_candidates()
            if not model_candidates:
                # Fallback when g4f registry isn't accessible
                model_candidates = ["gpt-4o-mini", "gpt-4o", "deepseek-r1"]
        else:
            model_candidates = [model_req]

        last_err: Exception | None = None
        last_text = ""

        for mname in model_candidates:
            try:
                resp = client.chat.completions.create(
                    model=mname,
                    messages=messages,
                    temperature=temperature,
                )
                text_out = (resp.choices[0].message.content or "").strip()
                if not text_out:
                    last_err = RuntimeError(f"g4f returned empty response for model={mname}")
                    continue

                # Validate JSON early: if model returns non-JSON, try the next one.
                try:
                    return _json_loads_best_effort(text_out)
                except Exception as je:
                    last_err = je
                    last_text = text_out
                    continue

            except Exception as e:
                last_err = e
                continue

        tail = (last_text[:200] + "…") if last_text and len(last_text) > 200 else last_text
        raise RuntimeError(
            f"g4f failed to produce valid JSON (last_err={last_err}; last_text={tail!r})"
        )

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
    return _json_loads_best_effort(text)


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
