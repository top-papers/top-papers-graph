#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

DEFAULT_CONFIG = Path("experiments/vlm_finetuning/datasphere/job_configs/hf_top_papers_sft_grpo_full_g2_2.yaml")
JOB_ID_PATTERNS = [
    re.compile(r"job[_-][a-z0-9-]+", re.IGNORECASE),
    re.compile(r"\b[a-z0-9]{20,}\b", re.IGNORECASE),
]


def run(cmd: List[str], log_file: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as f:
            f.write("\n$ " + " ".join(cmd) + "\n")
            f.write(proc.stdout or "")
    if proc.stdout:
        print(proc.stdout, end="", flush=True)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout)
    return proc


def parse_job_id(text: str) -> str | None:
    # Prefer explicit `id: ...`/`job id ...` lines when DataSphere CLI prints them.
    explicit = re.search(r"(?:job\s*)?(?:id|идентификатор)[\s:=]+([a-z0-9_-]{10,})", text, re.IGNORECASE)
    if explicit:
        return explicit.group(1)
    for pattern in JOB_ID_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def ensure_datasphere_cli() -> None:
    try:
        run(["datasphere", "version"], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise SystemExit(
            "DataSphere CLI is not available. Install it with `pip install datasphere` "
            "and authenticate via Yandex Cloud CLI/OAuth before running this pipeline."
        ) from exc


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the full HF top-papers VLM SFT+GRPO pipeline via Yandex DataSphere Jobs.")
    ap.add_argument("--project-id", default=os.environ.get("DATASPHERE_PROJECT_ID"), help="DataSphere project id. Can also be set via DATASPHERE_PROJECT_ID.")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--ttl-days", type=int, default=1, help="TTL for DataSphere job data after completion.")
    ap.add_argument("--log-dir", type=Path, default=Path("reports/datasphere_cli_runs"))
    ap.add_argument("--no-download", action="store_true", help="Skip explicit `download-files`; outputs are still declared in job config.")
    ap.add_argument("--dry-run", action="store_true", help="Validate local inputs and print commands without launching the job.")
    args = ap.parse_args()

    if not args.project_id:
        raise SystemExit("Set --project-id or DATASPHERE_PROJECT_ID.")
    if not args.config.exists():
        raise SystemExit(f"Job config not found: {args.config}")

    started_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_file = args.log_dir / f"hf_top_papers_sft_grpo_{started_at}.log"
    manifest_file = args.log_dir / f"hf_top_papers_sft_grpo_{started_at}.manifest.json"
    execute_cmd = ["datasphere", "project", "job", "execute", "-p", args.project_id, "-c", str(args.config)]

    manifest = {
        "started_at_utc": started_at,
        "project_id": args.project_id,
        "config": str(args.config),
        "job_id": None,
        "ttl_days": args.ttl_days,
        "commands": {"execute": execute_cmd},
        "status": "planned" if args.dry_run else "running",
    }
    write_manifest(manifest_file, manifest)

    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    ensure_datasphere_cli()
    job_id = None
    try:
        proc = run(execute_cmd, log_file=log_file, check=True)
        job_id = parse_job_id(proc.stdout or "")
        manifest["job_id"] = job_id
        manifest["status"] = "execute_finished"
        write_manifest(manifest_file, manifest)
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        write_manifest(manifest_file, manifest)
        if job_id:
            run(["datasphere", "project", "job", "cancel", "--id", job_id], log_file=log_file, check=False)
        raise
    except subprocess.CalledProcessError as exc:
        job_id = parse_job_id(exc.output or "")
        manifest["job_id"] = job_id
        manifest["status"] = "execute_failed"
        write_manifest(manifest_file, manifest)
        if job_id:
            run(["datasphere", "project", "job", "cancel", "--id", job_id], log_file=log_file, check=False)
        raise SystemExit(exc.returncode) from exc
    finally:
        if job_id:
            # A finished DataSphere Job releases its ephemeral VM; the TTL below removes cached job data quickly.
            run(["datasphere", "project", "job", "set-data-ttl", "--id", job_id, "--days", str(args.ttl_days)], log_file=log_file, check=False)

    if job_id and not args.no_download:
        run(["datasphere", "project", "job", "download-files", "--id", job_id], log_file=log_file, check=False)
        manifest["status"] = "download_requested"
    elif not job_id:
        manifest["status"] = "finished_but_job_id_not_parsed"
        manifest["warning"] = "The job completed, but the CLI output did not expose a parsable job id. Declared outputs should still be present after blocking execute."
    else:
        manifest["status"] = "finished_no_download"

    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_manifest(manifest_file, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
