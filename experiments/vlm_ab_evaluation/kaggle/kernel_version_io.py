#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exact-version Kaggle kernel I/O missing from the Kaggle 2.2.3 CLI wrapper."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

import requests


_KERNEL_REF_RE = re.compile(
    r"(?P<owner>[A-Za-z0-9][A-Za-z0-9_-]*)/"
    r"(?P<slug>[a-z0-9][a-z0-9-]{1,49})/"
    r"(?P<version>[1-9][0-9]*)"
)
_DATASET_REF_RE = re.compile(
    r"(?P<owner>[A-Za-z0-9][A-Za-z0-9_-]*)/(?P<slug>[a-z0-9][a-z0-9-]{1,49})"
)
_MAX_FILES = 100_000
_MAX_BYTES = 50 * 1024 * 1024 * 1024
_WINDOWS_RESERVED_NAMES = {
    "AUX",
    "CON",
    "NUL",
    "PRN",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def _kernel_ref(value: str) -> tuple[str, str, str]:
    match = _KERNEL_REF_RE.fullmatch(value)
    if match is None:
        raise ValueError("kernel must be owner/slug/positive-version")
    return match.group("owner"), match.group("slug"), match.group("version")


def _dataset_ref(value: str) -> tuple[str, str]:
    match = _DATASET_REF_RE.fullmatch(value)
    if match is None:
        raise ValueError("dataset must be owner/slug")
    return match.group("owner"), match.group("slug")


def _output_path(root: Path, raw: str) -> Path:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise ValueError(f"unsafe kernel output path: {raw!r}")
    relative = PurePosixPath(raw)
    if (
        relative.is_absolute()
        or relative.as_posix() != raw
        or any(part in {"", ".", ".."} for part in relative.parts)
        or any(
            ":" in part
            or part.endswith((" ", "."))
            or part.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES
            for part in relative.parts
        )
    ):
        raise ValueError(f"unsafe kernel output path: {raw!r}")
    return root.joinpath(*relative.parts)


def _prepare_output_parent(root: Path, output: Path) -> None:
    relative = output.relative_to(root)
    current = root
    for part in relative.parts[:-1]:
        current /= part
        if current.is_symlink():
            raise RuntimeError(f"kernel output path traverses a symlink: {relative.as_posix()!r}")
        current.mkdir(exist_ok=True)
        if not current.is_dir():
            raise RuntimeError(f"kernel output parent is not a directory: {relative.as_posix()!r}")
    current.resolve(strict=True).relative_to(root.resolve(strict=True))


def _api() -> Any:
    if importlib.metadata.version("kaggle") != "2.2.3":
        raise RuntimeError("kernel_version_io requires kaggle==2.2.3")
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def auth_info(api: Any) -> dict[str, Any]:
    api.dataset_list(mine=True, page=1)
    username = str(api.config_values.get("username") or "").strip()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", username) is None:
        raise RuntimeError("authenticated Kaggle client has no valid username")
    return {"authenticated": True, "username": username}


def kernel_status(api: Any, kernel: str) -> dict[str, Any]:
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest

    owner, slug, version = _kernel_ref(kernel)
    request = ApiGetKernelSessionStatusRequest()
    request.user_name = owner
    request.kernel_slug = slug
    request.version_label = version
    with api.build_kaggle_client() as client:
        response = client.kernels.kernels_api_client.get_kernel_session_status(request)
    raw_status = response.status
    status = getattr(raw_status, "name", None)
    if not isinstance(status, str) or not status:
        status = str(raw_status).rsplit(".", 1)[-1]
    status = status.strip().lower()
    if re.fullmatch(r"[a-z][a-z0-9_]*", status) is None:
        raise RuntimeError(f"Kaggle returned an invalid kernel status: {raw_status!r}")
    return {
        "kernel": kernel,
        "status": status,
        "failure_message": response.failure_message or None,
    }


def describe_kernel(api: Any, kernel: str) -> dict[str, Any]:
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

    owner, slug, version = _kernel_ref(kernel)
    request = ApiGetKernelRequest()
    request.user_name = owner
    request.kernel_slug = slug
    request.version_label = version
    with api.build_kaggle_client() as client:
        response = client.kernels.kernels_api_client.get_kernel(request)
    if response.blob is None or not isinstance(response.blob.source, str):
        raise RuntimeError("Kaggle returned no source for the exact kernel version")
    metadata = response.metadata
    if metadata is None:
        raise RuntimeError("Kaggle returned no metadata for the exact kernel version")
    canonical_source = response.blob.source.replace("\r\n", "\n").replace("\r", "\n")
    return {
        "kernel": kernel,
        "source_sha256": hashlib.sha256(canonical_source.encode("utf-8")).hexdigest(),
        "ref": metadata.ref,
        "is_private": metadata.is_private,
        "enable_gpu": metadata.enable_gpu,
        "enable_internet": metadata.enable_internet,
        "machine_shape": metadata.machine_shape,
        "dataset_sources": list(metadata.dataset_data_sources or []),
    }


def dataset_info(api: Any, dataset: str) -> dict[str, Any]:
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest

    owner, slug = _dataset_ref(dataset)
    request = ApiGetDatasetRequest()
    request.owner_slug = owner
    request.dataset_slug = slug
    with api.build_kaggle_client() as client:
        response = client.datasets.dataset_api_client.get_dataset(request)
    return {
        "dataset": dataset,
        "ref": response.ref,
        "is_private": response.is_private,
        "current_version_number": response.current_version_number,
    }


def dataset_status(api: Any, dataset: str) -> dict[str, Any]:
    from kagglesdk.datasets.types.dataset_api_service import (
        ApiGetDatasetRequest,
        ApiGetDatasetStatusRequest,
    )

    owner, slug = _dataset_ref(dataset)
    info_request = ApiGetDatasetRequest()
    info_request.owner_slug = owner
    info_request.dataset_slug = slug
    status_request = ApiGetDatasetStatusRequest()
    status_request.owner_slug = owner
    status_request.dataset_slug = slug
    with api.build_kaggle_client() as client:
        before = client.datasets.dataset_api_client.get_dataset(info_request)
        response = client.datasets.dataset_api_client.get_dataset_status(status_request)
        after = client.datasets.dataset_api_client.get_dataset(info_request)
    if before.current_version_number != after.current_version_number:
        raise RuntimeError("dataset version changed while its status was queried")
    raw_status = response.status
    status = getattr(raw_status, "name", None)
    if not isinstance(status, str) or not status:
        status = str(raw_status).rsplit(".", 1)[-1]
    status = status.strip().lower()
    if re.fullmatch(r"[a-z][a-z0-9_]*", status) is None:
        raise RuntimeError(f"Kaggle returned an invalid dataset status: {raw_status!r}")
    return {
        "dataset": dataset,
        "status": status,
        "current_version_number": before.current_version_number,
    }


def download_output(api: Any, kernel: str, target: Path) -> dict[str, Any]:
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

    owner, slug, version = _kernel_ref(kernel)
    destination = target.resolve(strict=True)
    if not destination.is_dir() or any(destination.iterdir()):
        raise ValueError("output target must be an existing empty directory")
    page_token: str | None = None
    page_tokens: set[str] = set()
    seen: set[str] = set()
    files: list[tuple[str, str]] = []
    total_bytes = 0
    log: str | None = None
    while True:
        request = ApiListKernelSessionOutputRequest()
        request.user_name = owner
        request.kernel_slug = slug
        request.version_label = version
        request.page_size = 200
        if page_token:
            request.page_token = page_token
        with api.build_kaggle_client() as client:
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
        if response.log:
            if log is not None and response.log != log:
                raise RuntimeError("Kaggle returned inconsistent logs across output pages")
            log = response.log
        for item in response.files or []:
            relative = item.file_name
            output = _output_path(destination, relative)
            canonical = relative.casefold()
            if canonical in seen or len(seen) >= _MAX_FILES:
                raise RuntimeError(f"duplicate or excessive kernel output path: {relative!r}")
            seen.add(canonical)
            parsed = urlparse(item.url or "")
            if parsed.scheme != "https" or not parsed.netloc:
                raise RuntimeError(f"Kaggle returned an unsafe output URL for {relative!r}")
            files.append((relative, str(item.url)))
        page_token = response.next_page_token or None
        if page_token is None:
            break
        if page_token in page_tokens:
            raise RuntimeError("Kaggle returned a repeated output page token")
        page_tokens.add(page_token)

    for relative, _url in files:
        parts = PurePosixPath(relative.casefold()).parts
        for index in range(1, len(parts)):
            if "/".join(parts[:index]) in seen:
                raise RuntimeError(
                    f"kernel output file is also an ancestor of another file: {relative!r}"
                )
    for relative, url in files:
        output = _output_path(destination, relative)
        _prepare_output_parent(destination, output)
        with requests.get(url, stream=True, timeout=120) as download:
            download.raise_for_status()
            final_url = urlparse(download.url)
            if final_url.scheme != "https" or not final_url.netloc:
                raise RuntimeError(f"Kaggle redirected to an unsafe URL for {relative!r}")
            with output.open("xb") as handle:
                for block in download.iter_content(1024 * 1024):
                    if not block:
                        continue
                    total_bytes += len(block)
                    if total_bytes > _MAX_BYTES:
                        raise RuntimeError("kernel output exceeds the download limit")
                    handle.write(block)
    if log is not None:
        log_bytes = log.encode("utf-8")
        total_bytes += len(log_bytes)
        if total_bytes > _MAX_BYTES:
            raise RuntimeError("kernel output and log exceed the download limit")
        log_path = destination / f"{slug}.log"
        if log_path.name.casefold() in seen or log_path.exists():
            raise RuntimeError("kernel log collides with an output file")
        log_path.write_text(log, encoding="utf-8", newline="\n")
    return {"kernel": kernel, "files": len(files), "bytes": total_bytes}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("auth")
    for name in ("status", "describe"):
        command = subparsers.add_parser(name)
        command.add_argument("--kernel", required=True)
    dataset = subparsers.add_parser("dataset-info")
    dataset.add_argument("--dataset", required=True)
    dataset_status_parser = subparsers.add_parser("dataset-status")
    dataset_status_parser.add_argument("--dataset", required=True)
    output = subparsers.add_parser("output")
    output.add_argument("--kernel", required=True)
    output.add_argument("--target", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        api = _api()
        if args.command == "auth":
            result = auth_info(api)
        elif args.command == "status":
            result = kernel_status(api, args.kernel)
        elif args.command == "describe":
            result = describe_kernel(api, args.kernel)
        elif args.command == "dataset-info":
            result = dataset_info(api, args.dataset)
        elif args.command == "dataset-status":
            result = dataset_status(api, args.dataset)
        else:
            result = download_output(api, args.kernel, args.target)
    except (OSError, RuntimeError, ValueError, requests.RequestException) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
