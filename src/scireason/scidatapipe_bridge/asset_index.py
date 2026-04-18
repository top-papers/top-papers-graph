from __future__ import annotations

import ast
import json
import re
import shutil
from dataclasses import dataclass, field
from html import unescape
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from .vendor.common.utils.paper_ids import paper_slug, resolve


_PAGE_IMG_RE = re.compile(r"page_(\d+)\.png$", re.IGNORECASE)
_FIG_LOC_RE = re.compile(r"(figure|fig\.?|table|tab\.?)\s*(\d+)", re.IGNORECASE)
_IMG_TAG_RE = re.compile(r"<img[^>]+src=[\"'](?P<src>[^\"']+)[\"'][^>]*>", re.IGNORECASE)
_CAPTION_RE = re.compile(r"<figcaption[^>]*>(?P<caption>.*?)</figcaption>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class AssetRecord:
    paper_id: str
    record_id: str
    source: str
    modality: str
    page: Optional[int] = None
    locator: str = ""
    image_path: str = ""
    text: str = ""
    caption: str = ""
    tables_md: str = ""
    equations_md: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_image(self) -> bool:
        return bool(self.image_path)

    def summary_text(self) -> str:
        parts: list[str] = []
        if self.caption:
            parts.append(f"caption={self.caption}")
        if self.text:
            parts.append(f"text={self.text}")
        if self.tables_md:
            parts.append(f"tables={self.tables_md}")
        if self.equations_md:
            parts.append(f"equations={self.equations_md}")
        return " | ".join(part for part in parts if part)


class AssetIndex:
    def __init__(self) -> None:
        self._by_key: dict[str, list[AssetRecord]] = {}
        self._by_paper_id: dict[str, list[AssetRecord]] = {}

    def add(self, record: AssetRecord, aliases: Iterable[str]) -> None:
        seen_keys: set[str] = set()
        for alias in aliases:
            key = _canonical_key(alias)
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            self._by_key.setdefault(key, []).append(record)
        paper_key = _canonical_key(record.paper_id)
        if paper_key:
            self._by_paper_id.setdefault(paper_key, []).append(record)

    def records_for_ref(self, ref: str) -> list[AssetRecord]:
        key = _canonical_key(ref)
        if not key:
            return []
        seen: set[str] = set()
        out: list[AssetRecord] = []
        for rec in self._by_key.get(key, []):
            if rec.record_id in seen:
                continue
            seen.add(rec.record_id)
            out.append(rec)
        return out

    def records_for_refs(self, refs: Sequence[str]) -> list[AssetRecord]:
        seen: set[str] = set()
        out: list[AssetRecord] = []
        for ref in refs:
            for rec in self.records_for_ref(ref):
                if rec.record_id in seen:
                    continue
                seen.add(rec.record_id)
                out.append(rec)
        return out

    def scan_processed_papers(self, roots: Sequence[Path]) -> None:
        for root in roots:
            self._scan_processed_root(root)

    def scan_bundle_inputs(self, roots: Sequence[Path]) -> None:
        for root in roots:
            if root.is_file() and root.suffix.lower() == ".html":
                self._scan_html_file(root)
                continue
            if root.is_file() and root.name == "edge_reviews.json":
                self._scan_edge_reviews(root)
                continue
            if not root.exists():
                continue
            if root.is_dir() and root.name == "processed_papers":
                self._scan_processed_root(root)
            for candidate in root.rglob("edge_reviews.json"):
                self._scan_edge_reviews(candidate)
            for candidate in root.rglob("processed_papers"):
                self._scan_processed_root(candidate)
            html_dirs = [candidate for candidate in root.rglob("html") if candidate.is_dir()]
            for html_dir in html_dirs:
                self._scan_html_root(html_dir)

    def _scan_processed_root(self, root: Path) -> None:
        if not root.exists() or not root.is_dir():
            return
        paper_dirs: list[Path] = []
        if (root / "meta.json").exists():
            paper_dirs = [root]
        else:
            paper_dirs = [p for p in root.iterdir() if p.is_dir() and (p / "meta.json").exists()]
        for paper_dir in paper_dirs:
            try:
                meta = json.loads((paper_dir / "meta.json").read_text(encoding="utf-8"))
            except Exception:
                meta = {}
            aliases = _aliases_from_meta(meta, paper_dir.name)
            page_records = list(_iter_pages(paper_dir))
            for rec in page_records:
                self.add(rec, aliases)

    def _scan_edge_reviews(self, path: Path) -> None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        assertions = payload.get("assertions") if isinstance(payload, dict) else None
        if not isinstance(assertions, list):
            return
        for idx, item in enumerate(assertions, start=1):
            if not isinstance(item, dict):
                continue
            evidence = _safe_parse_mapping(item.get("evidence_payload_full") or item.get("evidence") or item.get("evidence_text"))
            image_path_raw = str(evidence.get("image_path") or "").strip()
            if not image_path_raw:
                continue
            image_path = Path(image_path_raw)
            if not image_path.is_absolute():
                image_path = (path.parent / image_path).resolve()
            if not image_path.exists():
                continue
            paper_id = str(evidence.get("paper_id") or "").strip()
            if not paper_id:
                paper_ids = item.get("paper_ids") if isinstance(item.get("paper_ids"), list) else []
                if paper_ids:
                    paper_id = str(paper_ids[0])
            if not paper_id:
                continue
            page = _coerce_int(evidence.get("page"))
            locator = str(evidence.get("figure_or_table") or "").strip()
            rec = AssetRecord(
                paper_id=_canonical_display_id(paper_id),
                record_id=f"bundle:{path}:{idx}:{image_path.name}",
                source="task2_bundle",
                modality="figure",
                page=page,
                locator=locator,
                image_path=str(image_path.resolve().as_posix()),
                text=str(evidence.get("snippet_or_summary") or "").strip(),
                caption=locator,
            )
            self.add(rec, [paper_id, image_path.stem])

    def _scan_html_root(self, root: Path) -> None:
        for html_path in root.glob("*.html"):
            self._scan_html_file(html_path)

    def _scan_html_file(self, html_path: Path) -> None:
        try:
            text = html_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return
        slug = html_path.stem
        srcs = _IMG_TAG_RE.findall(text)
        captions = [re.sub(r"<[^>]+>", " ", unescape(c)).strip() for c in _CAPTION_RE.findall(text)]
        for idx, src in enumerate(srcs, start=1):
            if src.startswith("data:"):
                continue
            locator = f"Figure {idx}"
            caption = captions[idx - 1] if idx - 1 < len(captions) else ""
            image_path = src
            if not re.match(r"https?://", src):
                candidate = (html_path.parent / src).resolve()
                image_path = str(candidate.as_posix()) if candidate.exists() else src
            rec = AssetRecord(
                paper_id=slug,
                record_id=f"html:{html_path}:{idx}",
                source="html",
                modality="figure",
                locator=locator,
                image_path=image_path if re.match(r"https?://", image_path) or Path(image_path).exists() else "",
                caption=caption,
                text=caption,
            )
            self.add(rec, [slug, f"wiki:{slug}"])


def _aliases_from_meta(meta: dict[str, Any], fallback_name: str) -> list[str]:
    aliases: list[str] = []
    for key in ("id", "doi", "url", "paper_id", "source_id"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            aliases.append(value.strip())
            if key == "doi" and not value.lower().startswith("doi:"):
                aliases.append(f"doi:{value.strip()}")
    aliases.append(fallback_name)
    for alias in list(aliases):
        try:
            aliases.append(paper_slug(resolve(alias)))
        except Exception:
            pass
    seen: set[str] = set()
    out: list[str] = []
    for alias in aliases:
        if not alias or alias in seen:
            continue
        seen.add(alias)
        out.append(alias)
    return out


def _iter_pages(paper_dir: Path) -> Iterator[AssetRecord]:
    pages_path = paper_dir / "mm" / "pages.jsonl"
    if not pages_path.exists():
        return iter(())
    try:
        meta = json.loads((paper_dir / "meta.json").read_text(encoding="utf-8"))
    except Exception:
        meta = {}
    paper_id = _canonical_display_id(str(meta.get("id") or paper_dir.name))

    def _generator() -> Iterator[AssetRecord]:
        for idx, line in enumerate(pages_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            page = _coerce_int(payload.get("page"))
            img = str(payload.get("image_path") or "").strip()
            resolved_img = ""
            if img:
                img_path = Path(img)
                if not img_path.is_absolute():
                    img_path = (paper_dir / img).resolve()
                if img_path.exists():
                    resolved_img = str(img_path.as_posix())
            if not resolved_img and page is not None:
                for candidate in _page_image_candidates(paper_dir, page):
                    if candidate.exists():
                        resolved_img = str(candidate.resolve().as_posix())
                        break
            yield AssetRecord(
                paper_id=paper_id,
                record_id=f"page:{paper_id}:{page if page is not None else idx}",
                source="processed_papers",
                modality="page",
                page=page,
                locator=f"page {page}" if page is not None else "",
                image_path=resolved_img,
                text=str(payload.get("text") or "").strip(),
                caption=str(payload.get("vlm_caption") or "").strip(),
                tables_md=str(payload.get("tables_md") or "").strip(),
                equations_md=str(payload.get("equations_md") or "").strip(),
                extra={"raw": payload},
            )
    return _generator()


def _page_image_candidates(paper_dir: Path, page: int) -> list[Path]:
    mm_images = paper_dir / "mm" / "images"
    return [
        mm_images / f"page_{page:03d}.png",
        mm_images / f"page_{page + 1:03d}.png",
    ]


def _canonical_display_id(raw: str) -> str:
    try:
        return resolve(raw).id or raw
    except Exception:
        return raw


def _canonical_key(raw: str) -> str:
    raw = str(raw or "").strip()
    if not raw:
        return ""
    try:
        ref = resolve(raw)
        if ref.id:
            return ref.id
        slug = paper_slug(ref)
        return slug
    except Exception:
        return raw


def _safe_parse_mapping(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if raw in (None, ""):
        return {}
    if isinstance(raw, str):
        text = raw.strip()
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
    return {}


def _coerce_int(value: Any) -> Optional[int]:
    if value in (None, "", "unknown"):
        return None
    try:
        return int(str(value).strip())
    except Exception:
        return None


def parse_locator_number(locator: str) -> tuple[str | None, int | None]:
    m = _FIG_LOC_RE.search(locator or "")
    if not m:
        return None, None
    kind = "table" if m.group(1).lower().startswith("tab") else "figure"
    try:
        return kind, int(m.group(2))
    except Exception:
        return kind, None


def clone_image(image_path: str, out_dir: Path) -> str:
    if not image_path:
        return ""
    if re.match(r"https?://", image_path):
        return image_path
    src = Path(image_path)
    if not src.exists():
        return ""
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / src.name
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return str(dst.resolve().as_posix())
