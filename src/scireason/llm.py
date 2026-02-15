from __future__ import annotations

import hashlib
import json
import math
import os
import random
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


def _resolve_auto_provider() -> str:
    """Resolve settings.llm_provider='auto' into an actual provider.

    Strategy (course-friendly):
    1) If local Ollama is reachable -> 'ollama'
    2) Else if g4f is installed -> 'g4f'
    3) Else -> 'mock' (offline deterministic)
    """

    # 1) Ollama reachable?
    try:
        import httpx

        base = getattr(settings, "ollama_base_url", "http://localhost:11434")
        url = base.rstrip("/") + "/api/version"
        r = httpx.get(url, timeout=1.5)
        if r.status_code == 200:
            return "ollama"
    except Exception:
        pass

    # 2) g4f installed?
    if G4FClient is not None:
        return "g4f"

    return "mock"


def _resolve_model_for_ollama(model_req: str) -> str:
    """If model is 'auto', pick the first available local Ollama model (best effort)."""
    model_req = (model_req or "").strip()
    if model_req and model_req.lower() != "auto":
        return model_req

    try:
        import httpx

        base = getattr(settings, "ollama_base_url", "http://localhost:11434")
        url = base.rstrip("/") + "/api/tags"
        r = httpx.get(url, timeout=2.0)
        if r.status_code == 200:
            data = r.json()
            models = data.get("models") or []
            if models:
                name = models[0].get("name")
                if name:
                    return str(name)
    except Exception:
        pass

    # Conservative default that many installations use in demos.
    return "llama3.2"


def _mock_json(schema_hint: str, user: str) -> Any:
    """Offline deterministic JSON responses.

    This is NOT intended for quality, only for running the whole pipeline (tests / classroom).
    """
    sh = (schema_hint or "").lower()

    # TemporalTriplet[]
    if "temporaltriplet" in sh or "subject" in sh and "predicate" in sh and "polarity" in sh:
        # Make 3 toy triplets from the user text tokens.
        toks = re.findall(r"[a-zA-Z0-9_\-]+", user.lower())
        toks = [t for t in toks if len(t) >= 4][:12]
        a = toks[0] if len(toks) > 0 else "a"
        b = toks[1] if len(toks) > 1 else "b"
        c = toks[2] if len(toks) > 2 else "c"
        return [
            {
                "subject": a,
                "predicate": "correlates_with",
                "object": b,
                "confidence": 0.55,
                "polarity": "unknown",
                "evidence_quote": "(mock) extracted from text",
                "time": {"start": "2020", "end": "2020", "granularity": "year"},
            },
            {
                "subject": b,
                "predicate": "influences",
                "object": c,
                "confidence": 0.55,
                "polarity": "unknown",
                "evidence_quote": "(mock) extracted from text",
                "time": {"start": "2021", "end": "2021", "granularity": "year"},
            },
            {
                "subject": a,
                "predicate": "may_relate_to",
                "object": c,
                "confidence": 0.5,
                "polarity": "unknown",
                "evidence_quote": "(mock) extracted from text",
                "time": {"start": "2022", "end": "2022", "granularity": "year"},
            },
        ]

    # HypothesisDraft
    if "hypothesisdraft" in sh and "proposed_experiment" in sh:
        return {
            "title": "(mock) Graph-derived hypothesis",
            "premise": "(mock) Premise based on graph signals.",
            "mechanism": "(mock) Candidate mechanism.",
            "time_scope": "(mock)",
            "proposed_experiment": "(mock) Controlled experiment with metrics and ablations.",
            "supporting_evidence": [],
            "confidence_score": 5,
        }

    # Entities extraction
    if "entities" in sh:
        toks = re.findall(r"[a-zA-Z0-9_\-]+", user)
        toks = [t.strip() for t in toks if len(t.strip()) >= 4]
        uniq: list[str] = []
        for t in toks:
            if t.lower() not in {u.lower() for u in uniq}:
                uniq.append(t)
            if len(uniq) >= 8:
                break
        return {"entities": uniq}

    # HypothesisTestResult
    if "hypothesistestresult" in sh and "verdict" in sh:
        return {
            "verdict": "insufficient_evidence",
            "summary": "(mock) Not enough evidence in the provided context.",
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "temporal_notes": None,
            "recommended_experiments": "(mock) Add a targeted experiment.",
            "confidence_score": 4,
        }

    # Generic fallback: empty object
    return {}


def chat_text(system: str, user: str, *, temperature: float = 0.2) -> str:
    """Plain-text LLM call (used by code-writing agents)."""

    provider = (settings.llm_provider or "").lower().strip() or "auto"
    if provider == "auto":
        provider = _resolve_auto_provider()

    messages = [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": user.strip()},
    ]
    if provider == "mock":
        # For code agents we return deterministic, tool-using python.
        # If the prompt requests hypothesis *candidates* (a list of edges), return a list-like answer.
        full_prompt = (system + "\n" + user).lower()

        wants_candidates = (
            "candidate hypotheses" in full_prompt
            or "return a list of dicts" in full_prompt
            or "final_answer(<the_list>)" in full_prompt
            or "kind is one of" in full_prompt
        )

        if wants_candidates:
            return (
                "# mock provider: deterministic candidate generation\n"
                "G = build_graph()\n"
                "comms = []\n"
                "try:\n"
                "    comms = communities(G, method='greedy', max_communities=8)\n"
                "except Exception:\n"
                "    try:\n"
                "        comms = communities(G)\n"
                "    except Exception:\n"
                "        comms = []\n"
                "bridges = []\n"
                "try:\n"
                "    bridges = cross_bridges(G, comms, top_k=12)\n"
                "except Exception:\n"
                "    bridges = []\n"
                "cands = []\n"
                "for item in (bridges or [])[:10]:\n"
                "    try:\n"
                "        u, v, s = item\n"
                "    except Exception:\n"
                "        continue\n"
                "    cands.append({\n"
                "        'kind': 'cross_bridge',\n"
                "        'source': str(u),\n"
                "        'target': str(v),\n"
                "        'predicate': 'may_relate_to',\n"
                "        'score': float(s),\n"
                "        'graph_signals': {'bridge_score': float(s)}\n"
                "    })\n"
                "if not cands:\n"
                "    lp = []\n"
                "    try:\n"
                "        lp = link_prediction(G, method='adamic_adar', k=12)\n"
                "    except Exception:\n"
                "        lp = []\n"
                "    for item in (lp or [])[:10]:\n"
                "        try:\n"
                "            u, v, s = item\n"
                "        except Exception:\n"
                "            continue\n"
                "        cands.append({\n"
                "            'kind': 'link_prediction',\n"
                "            'source': str(u),\n"
                "            'target': str(v),\n"
                "            'predicate': 'may_relate_to',\n"
                "            'score': float(s),\n"
                "            'graph_signals': {'adamic_adar': float(s)}\n"
                "        })\n"
                "final_answer(cands)\n"
            )

        seed = int(hashlib.blake2b((system + user).encode('utf-8'), digest_size=4).hexdigest(), 16)
        random.seed(seed)
        return (
            "# mock provider: deterministic tool-using code\n"
            "G = None\n"
            "try:\n"
            "    G = build_graph()\n"
            "except Exception:\n"
            "    pass\n"
            "summary = graph_summary(G) if G is not None else {}\n"
            "comms = []\n"
            "try:\n"
            "    comms = communities(G) if G is not None else []\n"
            "except Exception:\n"
            "    pass\n"
            "cent = {}\n"
            "try:\n"
            "    cent = centrality(G, k=8) if G is not None else {}\n"
            "except Exception:\n"
            "    pass\n"
            "bridges = []\n"
            "try:\n"
            "    bridges = cross_bridges(G, comms, top_k=8) if G is not None else []\n"
            "except Exception:\n"
            "    pass\n"
            "_answer = {\"summary\": summary, \"communities\": comms[:3], \"centrality\": cent, \"bridges\": bridges}\n"
            "try:\n"
            "    final_answer(_answer)\n"
            "except Exception:\n"
            "    final_answer = _answer\n"
        )

    # ---- g4f ----
    if provider == "g4f":
        client = _g4f_client()
        model_req = (settings.llm_model or "auto").strip() or "auto"
        model_candidates = _g4f_model_candidates() if model_req.lower() in {"auto", ""} else [model_req]
        if not model_candidates:
            model_candidates = ["gpt-4o-mini", "gpt-4o"]

        last_err: Exception | None = None
        for mname in model_candidates:
            try:
                resp = client.chat.completions.create(
                    model=mname,
                    messages=messages,
                    temperature=temperature,
                )
                text_out = (resp.choices[0].message.content or "").strip()
                if text_out:
                    return text_out
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"g4f failed (last_err={last_err})")

    # ---- LiteLLM (including ollama/openai/anthropic/etc.) ----
    if litellm is None:
        raise RuntimeError("LiteLLM не установлен. Установите зависимости: pip install -e '.[dev]'")

    model = settings.llm_model
    if provider == "ollama":
        model = _resolve_model_for_ollama(model)
        model = f"ollama/{model}" if "/" not in model else model
    else:
        model = f"{provider}/{model}" if "/" not in model else model

    resp = litellm.completion(
        model=model,
        messages=messages,
        temperature=temperature,
        **_litellm_kwargs(provider),
    )
    return str(resp["choices"][0]["message"]["content"] or "")


def _normalize_message_content(content: Any) -> str:
    """Normalize message content to plain text.

    Some agent frameworks (including `smolagents`) may represent message content as a
    list of segments, e.g. {"type": "text", "text": "..."}. For maximum provider
    compatibility we collapse everything into a single string.
    """

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
                if p.get("type") == "text":
                    parts.append(str(p.get("text") or ""))
                else:
                    # Best-effort fallback
                    parts.append(str(p.get("text") or ""))
            else:
                parts.append(str(p))
        return "\n".join([x for x in parts if x])
    return str(content)


def chat_messages(messages: List[Dict[str, Any]], *, temperature: float = 0.2) -> str:
    """Chat completion over an explicit list of messages.

    This is primarily used by the optional `smolagents` backend, where the agent builds
    a multi-turn message history.
    """

    provider = (settings.llm_provider or "").lower().strip() or "auto"
    if provider == "auto":
        provider = _resolve_auto_provider()

    norm_messages: List[Dict[str, str]] = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "user")
        content = _normalize_message_content(m.get("content"))
        norm_messages.append({"role": role, "content": content})
    if provider == "mock":
        # Deterministic, tool-using code for agent runs in offline mode.
        blob = json.dumps(norm_messages, ensure_ascii=False, sort_keys=True)
        seed = int(hashlib.blake2b(blob.encode('utf-8'), digest_size=4).hexdigest(), 16)
        random.seed(seed)

        full_prompt = "\n".join([m.get('content', '') for m in norm_messages]).lower()
        wants_candidates = (
            "candidate hypotheses" in full_prompt
            or "return a list of dicts" in full_prompt
            or "final_answer(<the_list>)" in full_prompt
            or "kind is one of" in full_prompt
        )

        if wants_candidates:
            return (
                "# mock provider: deterministic candidate generation (chat_messages)\n"
                f"# seed={seed}\n"
                "G = build_graph()\n"
                "comms = []\n"
                "try:\n"
                "    comms = communities(G, method='greedy', max_communities=8)\n"
                "except Exception:\n"
                "    try:\n"
                "        comms = communities(G)\n"
                "    except Exception:\n"
                "        comms = []\n"
                "bridges = []\n"
                "try:\n"
                "    bridges = cross_bridges(G, comms, top_k=12)\n"
                "except Exception:\n"
                "    bridges = []\n"
                "cands = []\n"
                "for item in (bridges or [])[:10]:\n"
                "    try:\n"
                "        u, v, s = item\n"
                "    except Exception:\n"
                "        continue\n"
                "    cands.append({\n"
                "        'kind': 'cross_bridge',\n"
                "        'source': str(u),\n"
                "        'target': str(v),\n"
                "        'predicate': 'may_relate_to',\n"
                "        'score': float(s),\n"
                "        'graph_signals': {'bridge_score': float(s)}\n"
                "    })\n"
                "if not cands:\n"
                "    lp = []\n"
                "    try:\n"
                "        lp = link_prediction(G, method='adamic_adar', k=12)\n"
                "    except Exception:\n"
                "        lp = []\n"
                "    for item in (lp or [])[:10]:\n"
                "        try:\n"
                "            u, v, s = item\n"
                "        except Exception:\n"
                "            continue\n"
                "        cands.append({\n"
                "            'kind': 'link_prediction',\n"
                "            'source': str(u),\n"
                "            'target': str(v),\n"
                "            'predicate': 'may_relate_to',\n"
                "            'score': float(s),\n"
                "            'graph_signals': {'adamic_adar': float(s)}\n"
                "        })\n"
                "final_answer(cands)\n"
            )

        return (
            "# mock provider: deterministic tool-using code (chat_messages)\n"
            f"# seed={seed}\n"
            "G = None\n"
            "try:\n"
            "    G = build_graph()\n"
            "except Exception:\n"
            "    pass\n"
            "summary = graph_summary(G) if G is not None else {}\n"
            "comms = []\n"
            "try:\n"
            "    comms = communities(G) if G is not None else []\n"
            "except Exception:\n"
            "    pass\n"
            "cent = {}\n"
            "try:\n"
            "    cent = centrality(G, k=8) if G is not None else {}\n"
            "except Exception:\n"
            "    pass\n"
            "bridges = []\n"
            "try:\n"
            "    bridges = cross_bridges(G, comms, top_k=8) if G is not None else []\n"
            "except Exception:\n"
            "    pass\n"
            "_answer = {\"summary\": summary, \"communities\": comms[:3], \"centrality\": cent, \"bridges\": bridges}\n"
            "try:\n"
            "    final_answer(_answer)\n"
            "except Exception:\n"
            "    final_answer = _answer\n"
        )

    # ---- g4f ----
    if provider == "g4f":
        client = _g4f_client()
        model_req = (settings.llm_model or "auto").strip() or "auto"
        model_candidates = _g4f_model_candidates() if model_req.lower() in {"auto", ""} else [model_req]
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
                text_out = (resp.choices[0].message.content or "").strip()
                if text_out:
                    return text_out
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"g4f failed (last_err={last_err})")

    # ---- LiteLLM (including ollama/openai/anthropic/etc.) ----
    if litellm is None:
        raise RuntimeError("LiteLLM не установлен. Установите зависимости: pip install -e '.[dev]'")

    model = settings.llm_model
    if provider == "ollama":
        model = _resolve_model_for_ollama(model)
        model = f"ollama/{model}" if "/" not in model else model
    else:
        model = f"{provider}/{model}" if "/" not in model else model

    resp = litellm.completion(
        model=model,
        messages=norm_messages,
        temperature=temperature,
        **_litellm_kwargs(provider),
    )
    return str(resp["choices"][0]["message"]["content"] or "")


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
    provider = (settings.llm_provider or "").lower().strip() or "auto"
    if provider == "auto":
        provider = _resolve_auto_provider()

    messages = [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": f"""{user.strip()}

Верни ТОЛЬКО валидный JSON (без markdown). Схема/ожидания:
{schema_hint}
"""},
    ]


    # ---- 0) offline mock ----
    if provider == "mock":
        return _mock_json(schema_hint=schema_hint, user=user)

    # ---- 1) g4f ----
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

    # ---- 2) LiteLLM (incl. ollama/openai/anthropic/...) ----
    if litellm is None:
        raise RuntimeError("LiteLLM не установлен. Установите зависимости: pip install -e '.[dev]'")

    model = settings.llm_model
    if provider == "ollama":
        model = _resolve_model_for_ollama(model)
        model = f"ollama/{model}" if "/" not in model else model
    else:
        model = f"{provider}/{model}" if "/" not in model else model

    resp = litellm.completion(
        model=model,
        messages=messages,
        temperature=temperature,
        **_litellm_kwargs(provider),
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
