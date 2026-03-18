from __future__ import annotations

"""Model adapters for the optional `smolagents` backend.

We support 3 backends:

1) scireason: wrap this project's LLM router (LLM_PROVIDER=auto|g4f|ollama|...) as a smolagents model
2) transformers: local HF inference via smolagents.TransformersModel
3) g4f: direct g4f client calls as a smolagents model

The goal is to keep the pipeline usable in offline/classroom settings (via the `mock` provider),
while still allowing students to switch to real models when available.
"""

import importlib.util
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..config import settings


def _has_smolagents() -> bool:
    return importlib.util.find_spec("smolagents") is not None


def _require_smolagents() -> None:
    if not _has_smolagents():
        raise RuntimeError(
            "smolagents не установлен. Установите зависимости: pip install -e '.[agents]' "
            "(для локальных моделей HF: pip install -e '.[agents_hf]')"
        )


def _flatten_content(content: Any) -> str:
    """Convert smolagents-style content (string or list of segments) to plain text."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (bytes, bytearray)):
        try:
            return content.decode("utf-8", errors="ignore")
        except Exception:
            return str(content)
    if isinstance(content, list):
        parts: List[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                # smolagents uses {"type": "text", "text": "..."} for text segments
                if p.get("type") == "text":
                    parts.append(str(p.get("text") or ""))
                else:
                    # Best-effort fallback
                    parts.append(str(p.get("text") or ""))
            else:
                parts.append(str(p))
        return "\n".join([x for x in parts if x])
    return str(content)


def _to_openai_messages(messages: Any) -> List[Dict[str, str]]:
    """Convert ChatMessage objects or dict messages into OpenAI-like [{role, content}]."""

    out: List[Dict[str, str]] = []
    if not isinstance(messages, list):
        return out
    for m in messages:
        role = None
        content = None
        if isinstance(m, dict):
            role = m.get("role")
            content = m.get("content")
        else:
            # smolagents ChatMessage has `.role` and `.content`
            role = getattr(m, "role", None)
            content = getattr(m, "content", None)
        role_s = str(role) if role is not None else "user"
        # Enum values can look like MessageRole.USER; keep last part
        if "." in role_s:
            role_s = role_s.split(".")[-1]
        role_s = role_s.lower()
        if role_s not in {"system", "user", "assistant", "tool"}:
            role_s = "user"
        out.append({"role": role_s, "content": _flatten_content(content)})
    return out


def _truncate_on_stop(text: str, stop_sequences: Optional[List[str]] = None) -> str:
    if not stop_sequences:
        return text
    out = text
    for s in stop_sequences:
        if not s:
            continue
        idx = out.find(s)
        if idx >= 0:
            out = out[:idx]
    return out


@dataclass
class _SimpleChatMessage:
    """Fallback ChatMessage-like object with `.content`.

    Some older/newer smolagents versions accept any object with `.content`.
    We still prefer returning the real `smolagents.models.ChatMessage` when available.
    """

    content: str


def _make_chat_message(content: str):
    """Return a real smolagents ChatMessage if available, otherwise a small shim."""

    if _has_smolagents():
        try:
            from smolagents.models import ChatMessage, MessageRole  # type: ignore

            return ChatMessage(role=MessageRole.ASSISTANT, content=content)
        except Exception:
            pass
    return _SimpleChatMessage(content=content)


class ScireasonRouterModel:
    """smolagents-compatible model that delegates to `scireason.llm.chat_messages`."""

    def __init__(self, **kwargs: Any):
        self.kwargs = dict(kwargs or {})

    def __call__(
        self,
        messages: List[Dict[str, Any]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ):
        # We intentionally ignore `grammar` here: our router doesn't provide constrained generation.
        from ..llm import chat_messages

        temperature = float(kwargs.get("temperature", self.kwargs.get("temperature", 0.2)) or 0.2)
        norm_messages = _to_openai_messages(messages)
        text = chat_messages(norm_messages, temperature=temperature)
        text = _truncate_on_stop(text, stop_sequences)
        return _make_chat_message(text)


class G4FSmolModel:
    """smolagents-compatible model that calls g4f directly.

    This is useful when you want to use smolagents agents, but keep g4f as the backend.
    """

    def __init__(self, model_id: str = "auto", **kwargs: Any):
        self.model_id = (model_id or "auto").strip()
        self.kwargs = dict(kwargs or {})

    def __call__(
        self,
        messages: List[Dict[str, Any]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ):
        # g4f does not support grammar-constrained generation in a portable way.
        _ = grammar

        from ..llm import _g4f_client, _g4f_model_candidates  # type: ignore

        temperature = float(kwargs.get("temperature", self.kwargs.get("temperature", 0.2)) or 0.2)
        norm_messages = _to_openai_messages(messages)

        client = _g4f_client()
        model_req = self.model_id
        model_candidates = (
            _g4f_model_candidates() if model_req.lower() in {"auto", ""} else [model_req]
        )
        if not model_candidates:
            model_candidates = ["gpt-4o-mini", "gpt-4o"]

        last_err: Exception | None = None
        for mname in model_candidates:
            try:
                resp = client.chat.completions.create(
                    model=mname,
                    messages=norm_messages,
                    temperature=temperature,
                )
                text = (resp.choices[0].message.content or "").strip()
                if text:
                    text = _truncate_on_stop(text, stop_sequences)
                    return _make_chat_message(text)
            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(f"g4f failed for smolagents model (last_err={last_err})")


def create_smol_model() -> Any:
    """Create a model callable suitable for `smolagents.CodeAgent`.

    Returns a callable that takes `messages` and returns an object with `.content`.
    """

    _require_smolagents()

    backend = (getattr(settings, "smol_model_backend", "scireason") or "scireason").strip().lower()

    if backend == "transformers":
        # Local HF model (requires smolagents[transformers])
        try:
            from smolagents import TransformersModel  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "TransformersModel недоступна. Установите: pip install -e '.[agents_hf]' "
                "(или pip install 'smolagents[transformers]')"
            ) from e

        model_id = getattr(settings, "smol_model_id", None) or "HuggingFaceTB/SmolLM-135M-Instruct"

        # Map our settings to TransformersModel signature.
        kwargs: Dict[str, Any] = {}
        if getattr(settings, "smol_device_map", None):
            kwargs["device_map"] = settings.smol_device_map
        if getattr(settings, "smol_torch_dtype", None):
            kwargs["torch_dtype"] = settings.smol_torch_dtype
        # Generation kwargs are passed via **kwargs.
        if getattr(settings, "smol_max_new_tokens", None):
            kwargs["max_new_tokens"] = int(settings.smol_max_new_tokens)

        try:
            return TransformersModel(model_id=model_id, **kwargs)
        except TypeError:
            # Some older versions used positional `model_id`.
            return TransformersModel(model_id, **kwargs)

    if backend == "g4f":
        # Direct g4f integration
        return G4FSmolModel(model_id=getattr(settings, "smol_g4f_model", "auto") or "auto")

    # Default: reuse this project's LLM router
    return ScireasonRouterModel()
