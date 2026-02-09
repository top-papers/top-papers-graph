from __future__ import annotations

"""Domain configuration utilities.

Why this exists
---------------
SciReason is designed to support many scientific domains (battery, bio, ...). Most things that are
domain-sensitive (seed queries, ontologies, experiment backends, validation rules for expert
artifacts) should be configured via YAML rather than hard-coded.

The loader below is intentionally lightweight: it tries to load a YAML config file from the
repository (when running from source) but also works when the config is missing (e.g. when the
package is installed without the `configs/` directory).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml  # type: ignore

from .config import settings


@dataclass
class DomainConfig:
    domain_id: str
    title: str = "Science"
    keywords: List[str] = field(default_factory=list)
    seed_queries: List[str] = field(default_factory=list)

    # Optional sections (free-form dictionaries)
    kg: Dict[str, Any] = field(default_factory=dict)
    evaluation: Dict[str, Any] = field(default_factory=dict)
    artifact_validation: Dict[str, Any] = field(default_factory=dict)


def _candidate_paths(domain_id: str) -> List[Path]:
    """Resolve config candidate paths.

    Priority:
    1) explicit settings.domain_config_path
    2) configs/domains/<domain_id>.yaml relative to CWD
    3) configs/domains/<domain_id>.yaml relative to repository root (best effort)
    """
    paths: List[Path] = []
    if settings.domain_config_path:
        paths.append(Path(settings.domain_config_path))

    paths.append(Path("configs") / "domains" / f"{domain_id}.yaml")

    # When running from source, this file lives in <repo>/src/scireason/domain.py
    try:
        repo_root = Path(__file__).resolve().parents[2]
        paths.append(repo_root / "configs" / "domains" / f"{domain_id}.yaml")
    except Exception:
        pass

    # Unique, existing
    seen: set[str] = set()
    out: List[Path] = []
    for p in paths:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.exists():
            out.append(p)
    return out


def load_domain_config(domain_id: Optional[str] = None) -> DomainConfig:
    """Load domain configuration.

    If the config file is missing, returns a minimal DomainConfig so the system stays usable.
    """
    did = (domain_id or settings.domain_id or "science").strip()

    for path in _candidate_paths(did):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return DomainConfig(
            domain_id=str(data.get("domain_id") or did),
            title=str(data.get("title") or did),
            keywords=list(data.get("keywords") or []),
            seed_queries=list(data.get("seed_queries") or []),
            kg=dict(data.get("kg") or {}),
            evaluation=dict(data.get("evaluation") or {}),
            artifact_validation=dict(data.get("artifact_validation") or {}),
        )

    # Fallback
    return DomainConfig(domain_id=did, title=did)
