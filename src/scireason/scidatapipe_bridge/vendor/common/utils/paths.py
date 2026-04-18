"""Layout helpers for the ``data/`` folder.

New folder tree (see README)::

    data/
      raw/
        incoming_task1/
        incoming_task2/
      processed/
        normalized_task1/
          <submission_id>/
            <submission_id>.yaml
            sft.jsonl                   # per-submission
            data/
              images/<paper_slug>/…
              pdfs/<paper_slug>.pdf
              html/<paper_slug>.html
              cache/
        normalized_task2/
          <submission_id>/
            gold.json
            auto.json
            sft.jsonl                   # per-bundle gold-reconstruction
            grpo.jsonl                  # per-bundle auto-review
            data/ …
        sft.jsonl                       # merged across submissions
        grpo.jsonl                      # merged across bundles
"""
from __future__ import annotations

import os
from pathlib import Path


def _join(base: Path, *parts: str) -> Path:
    return Path(os.path.join(str(base), *parts))


def raw_dir(root: Path) -> Path:
    return _join(root, "raw")


def processed_dir(root: Path) -> Path:
    return _join(root, "processed")


def raw_task1_dir(root: Path) -> Path:
    return _join(raw_dir(root), "incoming_task1")


def raw_task2_dir(root: Path) -> Path:
    return _join(raw_dir(root), "incoming_task2")


def normalized_task1_dir(root: Path) -> Path:
    return _join(processed_dir(root), "normalized_task1")


def normalized_task2_dir(root: Path) -> Path:
    return _join(processed_dir(root), "normalized_task2")


def submission_root(task_dir: Path, submission_id: str) -> Path:
    return _join(task_dir, submission_id)


def submission_data_dir(submission_root: Path) -> Path:
    return _join(submission_root, "data")


def submission_images_dir(submission_root: Path) -> Path:
    return _join(submission_data_dir(submission_root), "images")


def submission_pdfs_dir(submission_root: Path) -> Path:
    return _join(submission_data_dir(submission_root), "pdfs")


def submission_html_dir(submission_root: Path) -> Path:
    return _join(submission_data_dir(submission_root), "html")


def submission_cache_dir(submission_root: Path) -> Path:
    return _join(submission_data_dir(submission_root), "cache")


def paper_images_dir(submission_root: Path, paper_slug: str) -> Path:
    return _join(submission_images_dir(submission_root), paper_slug)


def merged_sft_path(root: Path) -> Path:
    return _join(processed_dir(root), "sft.jsonl")


def merged_grpo_path(root: Path) -> Path:
    return _join(processed_dir(root), "grpo.jsonl")


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
