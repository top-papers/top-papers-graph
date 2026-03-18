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

    # Term graph / temporal KG knobs (optional)
    term_graph: Dict[str, Any] = field(default_factory=dict)


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

    Supports both canonical `domain_id` values (for example `science`) and Wikidata QIDs
    stored in expert trajectory artifacts (for example `Q336`).

    If the config file is missing, returns a minimal DomainConfig so the system stays usable.
    """
    did = (domain_id or settings.domain_id or "science").strip()

    def _from_data(data: dict[str, Any], fallback: str) -> DomainConfig:
        return DomainConfig(
            domain_id=str(data.get("domain_id") or fallback),
            title=str(data.get("title") or fallback),
            keywords=list(data.get("keywords") or []),
            seed_queries=list(data.get("seed_queries") or []),
            kg=dict(data.get("kg") or {}),
            evaluation=dict(data.get("evaluation") or {}),
            artifact_validation=dict(data.get("artifact_validation") or {}),
            term_graph=dict(data.get("term_graph") or {}),
        )

    for path in _candidate_paths(did):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return _from_data(data, did)

    # Expert trajectories store a Wikidata QID in `domain`. Support that form directly.
    qid = did.upper()
    if qid.startswith("Q") and qid[1:].isdigit():
        candidate_dirs: list[Path] = []
        try:
            repo_root = Path(__file__).resolve().parents[2]
            candidate_dirs.append(repo_root / "configs" / "domains")
        except Exception:
            pass
        candidate_dirs.append(Path("configs") / "domains")

        seen: set[str] = set()
        for directory in candidate_dirs:
            key = str(directory)
            if key in seen or not directory.exists():
                continue
            seen.add(key)
            for cfg_path in sorted(directory.glob("*.yaml")):
                try:
                    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                except Exception:
                    continue
                if str(data.get("wikidata_qid") or "").strip().upper() == qid:
                    return _from_data(data, did)

    # Fallback
    return DomainConfig(domain_id=did, title=did)
