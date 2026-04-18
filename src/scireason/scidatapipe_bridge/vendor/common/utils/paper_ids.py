"""Canonical paper-id resolver.

Handles the messy raw identifiers that appear in Task 1 YAMLs:
- ``arXiv:1701.07875v2``, ``https://arxiv.org/abs/1905.00158``, bare ``2407.08693``
- ``DOI:10.1088/0022-3719/6/7/010``, ``DOI: https://doi.org/10.1103/PhysRevLett.73.652``
- ``https://en.wikipedia.org/wiki/Word2vec``
- ``https://www.pi.website/download/pi0.pdf`` and other plain URLs

Returns a stable record with ``paper_type`` / canonical ``id`` / ``arxiv_id`` when
applicable. The goal is that two different strings pointing at the same paper
collapse to the same ``id``.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, asdict
from typing import Optional
from urllib.parse import unquote, urlparse


ARXIV_ID_RE = re.compile(
    r"(?:arxiv[:/]|abs/|pdf/)?(?P<id>\d{4}\.\d{4,5})(?P<version>v\d+)?",
    re.IGNORECASE,
)
# Older-style arxiv ids like math.GT/0309136 or hep-th/9901001.
ARXIV_OLD_RE = re.compile(
    r"(?:arxiv[:/]|abs/|pdf/)?(?P<id>[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?P<version>v\d+)?",
    re.IGNORECASE,
)
DOI_RE = re.compile(r"(10\.\d{4,9}/[^\s]+)", re.IGNORECASE)
WIKI_HOST_RE = re.compile(r"(?:[a-z]{2,}\.)?wikipedia\.org", re.IGNORECASE)


@dataclass
class PaperRef:
    """Canonical representation of a referenced paper."""

    id: str  # canonical id string, e.g. ``arxiv:1701.07875`` / ``doi:10.1038/379806a0``
    paper_type: str  # arxiv | doi | wiki | url
    arxiv_id: Optional[str] = None  # digits-only arxiv id (no version suffix)
    version: Optional[str] = None  # e.g. ``v2``
    raw: str = ""  # original string for audit

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v not in (None, "")}


def _clean(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").strip())


def resolve(raw: str) -> PaperRef:
    """Map a raw identifier string into a :class:`PaperRef`.

    The function never raises: unknown inputs fall back to ``paper_type='url'``.
    """
    original = (raw or "").strip()
    if not original:
        return PaperRef(id="", paper_type="url", raw="")

    # Legacy auto-generated bundles sometimes serialise the source as
    # ``"id:['https://en.wikipedia.org/wiki/GPT-3']"`` — strip the prefix and
    # unwrap the single-element Python list literal before further parsing.
    stripped = original
    while True:
        m = re.match(r"^id\s*:\s*(.*)$", stripped, re.IGNORECASE)
        if not m:
            break
        stripped = m.group(1).strip()
    m = re.match(r"^\[\s*['\"]?(?P<inner>.*?)['\"]?\s*\]$", stripped)
    if m:
        stripped = m.group("inner").strip()
    original = stripped or original

    # Recognise already-canonical identifiers (e.g. when we pass the output
    # of a previous ``resolve()`` back through the resolver).
    canonical_match = re.match(
        r"^(arxiv|doi|wiki|url)\s*:\s*(?P<body>.+)$", original, re.IGNORECASE
    )
    if canonical_match:
        ctype = canonical_match.group(1).lower()
        body = canonical_match.group("body").strip()
        if ctype == "arxiv":
            m = ARXIV_ID_RE.fullmatch(body) or ARXIV_OLD_RE.fullmatch(body)
            if m:
                aid = m.group("id")
                version = m.group("version") or None
                return PaperRef(
                    id=f"arxiv:{aid}",
                    paper_type="arxiv",
                    arxiv_id=aid,
                    version=version,
                    raw=original,
                )
        if ctype == "doi":
            return PaperRef(id=f"doi:{body.lower()}", paper_type="doi", raw=original)
        if ctype == "wiki":
            return PaperRef(id=f"wiki:{body}", paper_type="wiki", raw=original)
        if ctype == "url":
            return PaperRef(id=f"url:{body}", paper_type="url", raw=original)

    cleaned = _clean(original)

    # 1) arxiv — check both URL forms and bare ids
    parsed = urlparse(cleaned) if "://" in cleaned else None
    if parsed and parsed.netloc.lower().endswith("arxiv.org"):
        tail = parsed.path.rstrip("/").split("/")[-1]
        m = ARXIV_ID_RE.match(tail) or ARXIV_OLD_RE.match(tail)
        if m:
            aid = m.group("id")
            version = m.group("version") or None
            return PaperRef(
                id=f"arxiv:{aid}",
                paper_type="arxiv",
                arxiv_id=aid,
                version=version,
                raw=original,
            )

    if re.match(r"^arxiv[:/\s]", cleaned, re.IGNORECASE):
        m = ARXIV_ID_RE.search(cleaned) or ARXIV_OLD_RE.search(cleaned)
        if m:
            aid = m.group("id")
            version = m.group("version") or None
            return PaperRef(
                id=f"arxiv:{aid}",
                paper_type="arxiv",
                arxiv_id=aid,
                version=version,
                raw=original,
            )

    # Bare new-style arxiv id (like ``2407.08693`` or ``2201.12220v3``)
    m = re.fullmatch(ARXIV_ID_RE, cleaned)
    if m:
        aid = m.group("id")
        version = m.group("version") or None
        return PaperRef(
            id=f"arxiv:{aid}",
            paper_type="arxiv",
            arxiv_id=aid,
            version=version,
            raw=original,
        )

    # 2) DOI — "DOI:...", "DOI ...", "https://doi.org/..."
    if parsed and parsed.netloc.lower().endswith("doi.org"):
        doi = parsed.path.lstrip("/")
        return PaperRef(id=f"doi:{doi.lower()}", paper_type="doi", raw=original)

    m = DOI_RE.search(original)  # original, because cleaned lost spaces in DOIs
    if m:
        doi = m.group(1).rstrip(".,;)")
        # Strip trailing URL fragments if the DOI was embedded in a larger URL.
        doi = doi.split("#", 1)[0].split("?", 1)[0]
        return PaperRef(id=f"doi:{doi.lower()}", paper_type="doi", raw=original)

    # 3) Wikipedia
    if parsed and WIKI_HOST_RE.match(parsed.netloc or ""):
        slug = unquote(parsed.path.split("/wiki/", 1)[-1])
        slug = slug.rstrip("/")
        return PaperRef(id=f"wiki:{slug}", paper_type="wiki", raw=original)

    # 4) Generic URL fallback
    if parsed and parsed.scheme in {"http", "https"}:
        return PaperRef(id=f"url:{original}", paper_type="url", raw=original)

    # 5) Any other opaque string — treat as opaque id.
    return PaperRef(id=f"id:{original}", paper_type="url", raw=original)


def extract_figure_locator(locator: str) -> Optional[tuple[str, int]]:
    """Parse a locator string like ``"Fig. 2"`` / ``"Table 1 / Table 2"``.

    Returns a tuple ``(kind, number)`` where ``kind`` is ``figure`` or ``table``.
    Returns ``None`` when nothing figure/table-shaped is present.
    """
    if not locator:
        return None
    m = re.search(r"(figure|fig\.?|table|tab\.?)\s*(\d+)", locator, re.IGNORECASE)
    if not m:
        return None
    kind_raw = m.group(1).lower()
    number = int(m.group(2))
    kind = "table" if kind_raw.startswith("tab") else "figure"
    return kind, number


def has_figure_ref(locator: str) -> bool:
    return extract_figure_locator(locator) is not None


_SLUG_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def paper_slug(ref: "PaperRef | str") -> str:
    """Return a file-system safe slug for a paper reference.

    Examples::

        arxiv:1701.07875          -> arxiv_1701.07875
        doi:10.1039/d4na00784k    -> doi_10.1039_d4na00784k
        wiki:Word2vec             -> wiki_Word2vec
        url:https://a.b/p.pdf     -> url_<sha1[:10]>
    """
    if isinstance(ref, PaperRef):
        paper_type = ref.paper_type
        canonical = ref.id
    else:
        canonical = (ref or "").strip()
        paper_type = canonical.split(":", 1)[0] if ":" in canonical else "url"

    if not canonical:
        return "unknown"

    body = canonical.split(":", 1)[1] if ":" in canonical else canonical
    if paper_type == "url":
        digest = hashlib.sha1(body.encode("utf-8")).hexdigest()[:10]
        return f"url_{digest}"
    clean = _SLUG_SAFE_RE.sub("_", body).strip("_")
    return f"{paper_type}_{clean}" if clean else paper_type
