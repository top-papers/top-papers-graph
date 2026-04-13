from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import gc
import importlib.util
import json
import os
import subprocess
import sys
import threading
import tempfile
from pathlib import Path
from typing import Optional, Literal

from rich.console import Console

from ..config import settings


console = Console()

_G4F_AUTH_DISABLED = False
_G4F_TIMEOUT_DISABLED = False
_LOCAL_VLM_DISABLED = False
_LOCAL_VLM_DISABLE_REASON = ""
_LOCAL_VLM_WORKER = None
_LOCAL_VLM_WORKER_MODEL_ID = ""
_LOCAL_VLM_WORKER_LOCK = threading.Lock()


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
    Для остальных backends сначала пробуем актуальный
    `AutoModelForImageTextToText`, а для старых версий Transformers —
    `AutoModelForVision2Seq` как backward-compatible fallback.
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
            low_cpu_mem_usage=True,
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
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        return processor, model, "qwen2_5_vl"

    generic_model_cls = None
    try:
        from transformers import AutoModelForImageTextToText as generic_model_cls  # type: ignore
    except Exception:
        try:
            from transformers import AutoModelForVision2Seq as generic_model_cls  # type: ignore
        except Exception:
            _require("transformers image-text runtime (AutoModelForImageTextToText/AutoModelForVision2Seq)")

    model = generic_model_cls.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    return processor, model, "generic"


def _has_g4f() -> bool:
    return importlib.util.find_spec("g4f") is not None


def _run_isolated_python(code: str, *args: str, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    return subprocess.run(
        [sys.executable, "-c", code, *[str(a) for a in args]],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )




def _tail_worker_stderr(worker, limit: int = 8000) -> str:
    stderr_file = getattr(worker, "_scireason_stderr_file", None)
    if stderr_file is None:
        return ""
    try:
        stderr_file.flush()
        stderr_file.seek(0)
        data = stderr_file.read()
    except Exception:
        return ""
    data = str(data or "")
    if len(data) <= limit:
        return data.strip()
    return data[-limit:].strip()


def _close_local_vlm_worker() -> None:
    global _LOCAL_VLM_WORKER, _LOCAL_VLM_WORKER_MODEL_ID
    worker = _LOCAL_VLM_WORKER
    _LOCAL_VLM_WORKER = None
    _LOCAL_VLM_WORKER_MODEL_ID = ""
    if worker is None:
        return
    try:
        if getattr(worker, "stdin", None):
            worker.stdin.write(json.dumps({"cmd": "shutdown"}, ensure_ascii=False) + "\n")
            worker.stdin.flush()
    except Exception:
        pass
    try:
        worker.terminate()
    except Exception:
        pass
    try:
        worker.wait(timeout=5)
    except Exception:
        try:
            worker.kill()
        except Exception:
            pass
    stderr_file = getattr(worker, "_scireason_stderr_file", None)
    if stderr_file is not None:
        try:
            stderr_file.close()
        except Exception:
            pass


def _augment_pythonpath_for_repo(env: dict[str, str]) -> dict[str, str]:
    env = dict(env)
    repo_root = Path(__file__).resolve().parents[3]
    src_dir = repo_root / "src"
    parts: list[str] = []
    existing = str(env.get("PYTHONPATH") or "").strip()
    if src_dir.exists():
        parts.append(str(src_dir))
    if repo_root.exists():
        parts.append(str(repo_root))
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(p for p in parts if p)
    return env


def _ensure_local_vlm_worker(model_id: str):
    global _LOCAL_VLM_WORKER, _LOCAL_VLM_WORKER_MODEL_ID
    model_id = str(model_id or "").strip()
    with _LOCAL_VLM_WORKER_LOCK:
        worker = _LOCAL_VLM_WORKER
        if worker is not None and worker.poll() is None and _LOCAL_VLM_WORKER_MODEL_ID == model_id:
            return worker
        _close_local_vlm_worker()
        cmd = [sys.executable, "-m", "scireason.mm.vlm_worker", model_id]
        env = _augment_pythonpath_for_repo(os.environ.copy())
        env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        stderr_file = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
        worker = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            text=True,
            bufsize=1,
            env=env,
            cwd=str(Path(__file__).resolve().parents[3]),
        )
        setattr(worker, "_scireason_stderr_file", stderr_file)
        _LOCAL_VLM_WORKER = worker
        _LOCAL_VLM_WORKER_MODEL_ID = model_id
        return worker


def _describe_image_qwen_worker(image_path: Path, prompt: str, model_id: str, max_new_tokens: int) -> VLMResult:
    worker = _ensure_local_vlm_worker(model_id)
    if worker.stdin is None or worker.stdout is None:
        raise RuntimeError("local_vlm_worker_pipes_unavailable")

    payload = {
        "cmd": "describe",
        "image_path": str(image_path),
        "prompt": prompt,
        "model_id": model_id,
        "max_new_tokens": int(max_new_tokens),
    }
    try:
        worker.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        worker.stdin.flush()
        line = worker.stdout.readline()
    except Exception as exc:
        _close_local_vlm_worker()
        raise RuntimeError(f"local_vlm_worker_io_error: {type(exc).__name__}: {exc}") from exc

    if not line:
        stderr = _tail_worker_stderr(worker)
        _close_local_vlm_worker()
        detail = stderr or f"worker_exited={worker.poll()}"
        raise RuntimeError(f"local_vlm_worker_no_response: {detail}")

    try:
        reply = json.loads(line)
    except Exception as exc:
        _close_local_vlm_worker()
        raise RuntimeError(f"local_vlm_worker_bad_json: {line[:500]}") from exc

    if not bool(reply.get("ok")):
        raise RuntimeError(str(reply.get("error") or "local_vlm_worker_failed"))

    return VLMResult(
        caption=str(reply.get("caption") or ""),
        extracted_tables_md=reply.get("extracted_tables_md") or None,
        extracted_equations_md=reply.get("extracted_equations_md") or None,
    )


def _prefer_isolated_local_vlm() -> bool:
    raw = str(os.environ.get("SCIREASON_LOCAL_VLM_MODE", "worker") or "worker").strip().lower()
    return raw not in {"0", "false", "off", "inprocess", "direct"}


def _allow_inprocess_local_vlm_fallback() -> bool:
    raw = os.environ.get("SCIREASON_LOCAL_VLM_ALLOW_INPROCESS_FALLBACK")
    if raw is None:
        try:
            return bool(getattr(settings, "local_vlm_allow_inprocess_fallback", False))
        except Exception:
            return False
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=16)
def _local_vlm_runtime_check(model_id: Optional[str] = None) -> tuple[bool, str]:
    """Return whether the local VLM runtime is really importable.

    Проверка идёт в отдельном subprocess, чтобы не ломать notebook-kernel повторным
    импортом torch/transformers после pip-install и не ловить ложные отрицания из-за
    частично отравленного sys.modules в текущем процессе.
    """
    requested = str(model_id or "").strip()
    probe = r"""import json
import sys

requested = (sys.argv[1] if len(sys.argv) > 1 else '').strip()
result = {"ok": False, "reason": ""}

try:
    import torch  # noqa: F401
    from PIL import Image  # noqa: F401
    from transformers import AutoProcessor  # noqa: F401
except Exception as e:
    result["reason"] = f"{type(e).__name__}: {e}"
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0)

try:
    if 'Qwen3-VL' in requested:
        from transformers import Qwen3VLForConditionalGeneration  # noqa: F401
    elif 'Qwen/Qwen2.5-VL' in requested or 'Qwen2.5-VL' in requested:
        from transformers import Qwen2_5_VLForConditionalGeneration  # noqa: F401
        import qwen_vl_utils  # noqa: F401
    else:
        try:
            from transformers import AutoModelForImageTextToText  # noqa: F401
        except Exception:
            from transformers import AutoModelForVision2Seq  # noqa: F401
except Exception as e:
    result["reason"] = f"{type(e).__name__}: {e}"
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0)

result["ok"] = True
print(json.dumps(result, ensure_ascii=False))
"""
    try:
        proc = _run_isolated_python(probe, requested, timeout=120)
    except subprocess.TimeoutExpired:
        return False, "TimeoutExpired: isolated_import_probe"

    lines = (proc.stdout or "").strip().splitlines()
    payload = lines[-1] if lines else ""
    if payload:
        try:
            parsed = json.loads(payload)
            return bool(parsed.get("ok")), str(parsed.get("reason") or "")
        except Exception:
            pass

    stderr_lines = (proc.stderr or "").strip().splitlines()
    reason = stderr_lines[-1] if stderr_lines else f"isolated_probe_exit_{proc.returncode}"
    return False, reason


def _has_local_vlm_stack(model_id: Optional[str] = None) -> bool:
    ok, _ = _local_vlm_runtime_check(model_id=model_id)
    return ok


def reset_vlm_runtime_state() -> None:
    global _G4F_AUTH_DISABLED, _G4F_TIMEOUT_DISABLED, _LOCAL_VLM_DISABLED, _LOCAL_VLM_DISABLE_REASON
    _G4F_AUTH_DISABLED = False
    _G4F_TIMEOUT_DISABLED = False
    _LOCAL_VLM_DISABLED = False
    _LOCAL_VLM_DISABLE_REASON = ""
    _close_local_vlm_worker()
    try:
        _local_vlm_runtime_check.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        pass


def _local_stack_available(model_id: Optional[str] = None) -> bool:
    try:
        return bool(_has_local_vlm_stack(model_id=model_id))
    except TypeError:
        return bool(_has_local_vlm_stack())

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


def _missing_auth_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    indicators = (
        "missingautherror",
        "api key is required",
        "add a \"api_key\"",
        "not authenticated",
        "unauthorized",
        "no yupp accounts configured",
        "set yupp_api_key",
        "set openai_api_key",
        "set gemini_api_key",
        "set groq_api_key",
        "set openrouter_api_key",
        "requires a .har file",
        "requires authentication",
    )
    return any(token in msg for token in indicators)


def _provider_unavailable_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    indicators = (
        "all providers failed",
        "no provider",
        "no providers",
        "provider not found",
        "provider is not available",
        "provider init",
        "retryprovider",
        "best_provider",
        "model is not supported by provider",
        "not supported by provider",
    )
    return any(token in msg for token in indicators)


def _run_with_timeout(timeout_seconds: float, fn, /, *args, **kwargs):
    """Run a blocking function in a daemon thread and fail fast on timeout.

    This protects notebook/CLI execution from hanging forever on flaky remote VLM
    providers (most notably g4f-backed multimodal calls).
    """

    timeout_seconds = float(timeout_seconds or 0)
    if timeout_seconds <= 0:
        return fn(*args, **kwargs)

    state: dict[str, object] = {}

    def _target() -> None:
        try:
            state["result"] = fn(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - exercised via caller tests
            state["error"] = exc

    thread = threading.Thread(target=_target, name=f"vlm-timeout:{getattr(fn, '__name__', 'call')}", daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        raise TimeoutError(f"VLM call exceeded {timeout_seconds:g}s")
    if "error" in state:
        raise state["error"]  # type: ignore[misc]
    return state.get("result")


def _resolve_backend(backend: Optional[Backend], model_id: Optional[str]) -> Backend:
    global _G4F_AUTH_DISABLED, _G4F_TIMEOUT_DISABLED, _LOCAL_VLM_DISABLED
    requested = str(backend or getattr(settings, "vlm_backend", "none") or "none").strip().lower()

    if requested == "g4f" and (_G4F_AUTH_DISABLED or _G4F_TIMEOUT_DISABLED):
        if _local_stack_available(model_id=model_id):
            return "qwen2_vl"
        return "none"

    if requested in {"qwen2_vl", "qwen3_vl", "llava", "phi3_vision"} and _LOCAL_VLM_DISABLED:
        return "none"

    if requested == "auto":
        if model_id and ("Qwen/Qwen3-VL" in model_id or "Qwen3-VL" in model_id):
            return "qwen3_vl" if _local_stack_available(model_id=model_id) else ("g4f" if _has_g4f() else "none")
        if _local_stack_available(model_id=model_id):
            return "qwen2_vl"
        if _has_g4f():
            return "g4f"
        return "none"

    if requested == "g4f" and not _has_g4f():
        if _local_stack_available(model_id=model_id):
            console.print("[yellow]g4f не установлен; переключаю VLM на локальный Transformers backend.[/yellow]")
            return "qwen2_vl"
        console.print("[yellow]g4f не установлен; продолжу без VLM-captioning.[/yellow]")
        return "none"

    if requested in {"qwen2_vl", "qwen3_vl", "llava", "phi3_vision"} and not _local_stack_available(model_id=model_id):
        if requested != "none":
            _, reason = _local_vlm_runtime_check(model_id=model_id)
            suffix = f" Причина: {reason}." if reason else ""
            console.print(f"[yellow]Локальный VLM-стек недоступен; продолжу без VLM-captioning.{suffix}[/yellow]")
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


def _describe_image_qwen_inprocess(image_path: Path, prompt: str, model_id: str, max_new_tokens: int) -> VLMResult:
    try:
        import torch  # type: ignore
        from PIL import Image  # type: ignore
    except Exception:
        _require("torch/pillow")

    processor, model, family = _load_transformers_vlm(model_id)
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
        finally:
            # Release per-page tensors / images aggressively to keep notebook RAM stable.
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
    m = caption.split("TABLES:", 1)
    if len(m) == 2:
        caption, rest = m[0].strip(), m[1].strip()
        m2 = rest.split("EQUATIONS:", 1)
        if len(m2) == 2:
            tables_md, equations_md = m2[0].strip(), m2[1].strip()
        else:
            tables_md = rest

    return VLMResult(caption=caption, extracted_tables_md=tables_md, extracted_equations_md=equations_md)




def _describe_image_qwen(image_path: Path, prompt: str, model_id: str, max_new_tokens: int) -> VLMResult:
    if _prefer_isolated_local_vlm():
        try:
            return _describe_image_qwen_worker(
                image_path=image_path,
                prompt=prompt,
                model_id=model_id,
                max_new_tokens=max_new_tokens,
            )
        except Exception as exc:
            if _allow_inprocess_local_vlm_fallback() and _local_stack_available(model_id=model_id):
                console.print(
                    f"[yellow]Isolated local VLM worker failed for {image_path.name}: {type(exc).__name__}: {exc}. "
                    "Пробую in-process fallback.[/yellow]"
                )
            else:
                raise
    return _describe_image_qwen_inprocess(
        image_path=image_path,
        prompt=prompt,
        model_id=model_id,
        max_new_tokens=max_new_tokens,
    )


@contextmanager
def temporary_vlm_selection(*, vlm_backend: Optional[str] = None, vlm_model_id: Optional[str] = None):
    """Temporarily override VLM routing for notebook/CLI execution."""

    prev_backend = getattr(settings, "vlm_backend", "none")
    prev_model = getattr(settings, "vlm_model_id", "")
    reset_vlm_runtime_state()
    try:
        if vlm_backend and str(vlm_backend).strip():
            settings.vlm_backend = str(vlm_backend).strip()
        if vlm_model_id and str(vlm_model_id).strip():
            settings.vlm_model_id = str(vlm_model_id).strip()
        yield
    finally:
        settings.vlm_backend = prev_backend
        settings.vlm_model_id = prev_model
        reset_vlm_runtime_state()


def describe_image(
    image_path: Path,
    prompt: str,
    backend: Optional[Backend] = None,
    model_id: Optional[str] = None,
    max_new_tokens: int = 512,
) -> VLMResult:
    """Описывает изображение (страница PDF / figure / table) через VL-модель.

    Никогда не роняет общий ingest из-за отсутствия опционального VLM backend.
    Дополнительно защищён от бесконечного ожидания ответа remote VLM.
    """
    requested_model_id = model_id or settings.vlm_model_id  # type: ignore[attr-defined]
    effective_backend = _resolve_backend(backend or settings.vlm_backend, requested_model_id)  # type: ignore[attr-defined]
    effective_model_id = _resolve_model_id_for_backend(effective_backend, requested_model_id)

    if effective_backend == "none":
        return VLMResult(caption="")

    try:
        if effective_backend == "g4f":
            timeout_seconds = float(getattr(settings, "vlm_request_timeout_seconds", 45) or 45)
            return _run_with_timeout(
                timeout_seconds,
                _describe_image_g4f,
                image_path=image_path,
                prompt=prompt,
                model_id=effective_model_id,
            )

        return _describe_image_qwen(
            image_path=image_path,
            prompt=prompt,
            model_id=effective_model_id,
            max_new_tokens=max_new_tokens,
        )
    except Exception as e:
        global _G4F_AUTH_DISABLED, _G4F_TIMEOUT_DISABLED, _LOCAL_VLM_DISABLED
        if effective_backend == "g4f" and (_missing_auth_error(e) or _provider_unavailable_error(e)):
            _G4F_AUTH_DISABLED = True
            if _has_local_vlm_stack():
                console.print(
                    f"[yellow]g4f требует аутентификацию или не нашёл рабочий provider/model; переключаю VLM на локальный Transformers backend начиная с {image_path.name}.[/yellow]"
                )
                try:
                    fallback_model = _resolve_model_id_for_backend("qwen2_vl", None)
                    return _describe_image_qwen(
                        image_path=image_path,
                        prompt=prompt,
                        model_id=fallback_model,
                        max_new_tokens=max_new_tokens,
                    )
                except Exception as inner_e:
                    console.print(
                        f"[yellow]Локальный VLM fallback для {image_path.name} тоже недоступен: {type(inner_e).__name__}: {inner_e}. Продолжаю без caption/tables/equations.[/yellow]"
                    )
                    return VLMResult(caption="")
            console.print(
                f"[yellow]g4f требует API key или другой provider; отключаю g4f-captioning для оставшихся страниц после {image_path.name}.[/yellow]"
            )
            return VLMResult(caption="")

        if effective_backend == "g4f" and isinstance(e, TimeoutError):
            _G4F_TIMEOUT_DISABLED = True
            if _has_local_vlm_stack():
                console.print(
                    f"[yellow]g4f не ответил вовремя на {image_path.name}; переключаю VLM на локальный Transformers backend для оставшихся страниц.[/yellow]"
                )
                try:
                    fallback_model = _resolve_model_id_for_backend("qwen2_vl", None)
                    return _describe_image_qwen(
                        image_path=image_path,
                        prompt=prompt,
                        model_id=fallback_model,
                        max_new_tokens=max_new_tokens,
                    )
                except Exception as inner_e:
                    console.print(
                        f"[yellow]Локальный VLM fallback для {image_path.name} тоже недоступен: {type(inner_e).__name__}: {inner_e}. Продолжаю без caption/tables/equations.[/yellow]"
                    )
                    return VLMResult(caption="")
            console.print(
                f"[yellow]g4f не ответил вовремя на {image_path.name}; отключаю g4f-captioning для оставшихся страниц.[/yellow]"
            )
            return VLMResult(caption="")

        if effective_backend in {"qwen2_vl", "qwen3_vl", "llava", "phi3_vision"}:
            _LOCAL_VLM_DISABLED = True
            _LOCAL_VLM_DISABLE_REASON = f"{type(e).__name__}: {e}"
            console.print(
                f"[yellow]Локальный VLM backend недоступен для {image_path.name}: {type(e).__name__}: {e}. "
                "Отключаю local VLM для оставшихся страниц, чтобы не повторять предупреждение на каждой странице. "
                "Переустановите runtime (torch/transformers/qwen-vl-utils) и перезапустите notebook, если нужен local VLM.[/yellow]"
            )
            return VLMResult(caption="")

        console.print(
            f"[yellow]VLM warning for {image_path.name} ({effective_backend}): {type(e).__name__}: {e}. "
            "Продолжаю без caption/tables/equations для этой страницы.[/yellow]"
        )
        return VLMResult(caption="")
