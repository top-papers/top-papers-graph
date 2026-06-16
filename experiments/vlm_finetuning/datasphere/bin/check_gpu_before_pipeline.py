#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def run_nvidia_smi() -> dict:
    try:
        proc = subprocess.run(
            ["nvidia-smi"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        output = proc.stdout or ""
        print("[gpu-check] nvidia-smi output:")
        print(output.rstrip() or "<empty>")
        return {"available": True, "returncode": proc.returncode, "output_tail": output[-4000:]}
    except FileNotFoundError:
        print("[gpu-check] nvidia-smi not found")
        return {"available": False, "returncode": None, "output_tail": "nvidia-smi not found"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail fast if DataSphere GPU/CUDA/BF16 runtime is not usable.")
    parser.add_argument("--report-dir", default=None, help="Directory where gpu_preflight_status.json will be written.")
    parser.add_argument("--require-bf16", action="store_true", help="Fail if torch.cuda.is_bf16_supported() is false.")
    parser.add_argument("--skip-env", default="SKIP_GPU_PREFLIGHT", help="Environment variable name that can bypass failures when true/1/yes/on.")
    args = parser.parse_args()

    out_prefix = os.environ.get("OUT_PREFIX", "hf_top_papers_qwen3vl_8b")
    report_dir = Path(args.report_dir or os.environ.get("REPORT_DIR", f"reports/{out_prefix}_datasphere"))
    report_dir.mkdir(parents=True, exist_ok=True)
    status_path = report_dir / "gpu_preflight_status.json"

    status: dict = {"ok": False, "nvidia_smi": run_nvidia_smi()}

    try:
        import torch
    except Exception as exc:  # pragma: no cover - executed in DataSphere runtime
        status.update({
            "torch_import_ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(f"[gpu-check] torch import failed: {type(exc).__name__}: {exc}") from exc

    status.update({
        "torch_import_ok": True,
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "bf16_supported": False,
    })

    if torch.cuda.is_available():
        try:
            status["device_name_0"] = torch.cuda.get_device_name(0)
        except Exception as exc:
            status["device_name_0_error"] = f"{type(exc).__name__}: {exc}"
        try:
            if hasattr(torch.cuda, "is_bf16_supported"):
                status["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        except Exception as exc:
            status["bf16_supported_error"] = f"{type(exc).__name__}: {exc}"

    status["ok"] = bool(status["cuda_available"] and (not args.require_bf16 or status["bf16_supported"]))
    print("[gpu-check] torch/CUDA status:")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    skip_value = os.environ.get(args.skip_env, "").lower()
    if skip_value in {"1", "true", "yes", "on"}:
        print(f"[gpu-check] {args.skip_env}={skip_value}; continuing despite GPU preflight status")
        return

    if not status["cuda_available"]:
        raise SystemExit(
            "[gpu-check] CUDA is not available inside DataSphere job. "
            "Most likely the PyTorch CUDA wheel is incompatible with the DataSphere NVIDIA driver. "
            "Check reports/*/gpu_preflight_status.json and pin torch/torchvision/torchaudio to a compatible CUDA wheel."
        )

    if args.require_bf16 and not status["bf16_supported"]:
        raise SystemExit(
            "[gpu-check] CUDA is available, but BF16 is not supported. "
            "This Qwen3-VL-8B recipe expects an A100/H100-class GPU or another BF16-capable accelerator."
        )

    print(f"[gpu-check] OK: wrote {status_path}")


if __name__ == "__main__":
    main()
