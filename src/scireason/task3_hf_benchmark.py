"""Build a Hugging Face-ready Task 3 VLM generation benchmark dataset.

The builder converts Task 3 creator manifests into model-facing generation
samples.  It intentionally separates the expert review metadata from the
messages shown to the model, because the expert rationale and expected error
modes would leak the purpose of the A/B case.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import os
import re
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SYSTEM_PROMPT_RU = (
    "Вы — vision-language модель для анализа научных публикаций. "
    "Отвечайте только на основе предоставленных вводных данных, изображений страниц, таблиц, "
    "рисунков и текста. Не выдумывайте численные значения и не делайте выводы, если evidence "
    "недостаточно. Ответ должен быть полезен эксперту, который затем будет сравнивать два ответа "
    "в blind A/B тесте."
)

OUTPUT_SCHEMA_TEXT_RU = """Верните ответ в JSON-объекте со следующими ключами:
{
  "answer": "краткий предметный ответ на задачу",
  "evidence_used": [
    {"kind": "figure|table|page|text|unknown", "locator": "например Fig. 2 / Table 1 / p. 3", "description": "какая evidence использована"}
  ],
  "visual_facts": ["факты, извлечённые из изображений/таблиц/графиков"],
  "temporal_facts": ["временные метки, фазы, лаги или последовательности, если применимо"],
  "uncertainty": "low|medium|high",
  "missing_evidence": ["что не удалось проверить по предоставленным данным"]
}
"""

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
SUPPORTED_MANIFEST_SUFFIXES = {".json", ".yaml", ".yml"}
PDF_SUFFIXES = {".pdf"}
ARCHIVE_SUFFIXES = {".zip"}


@dataclass(frozen=True)
class BuildConfig:
    input_paths: tuple[Path, ...]
    output_dir: Path
    dataset_repo_id: str = "top-papers/top-papers-graph-benchmark"
    dataset_config_name: str = "task3_vlm_generation"
    split_name: str = "test"
    include_disabled_cases: bool = False
    include_incomplete_cases: bool = False
    copy_source_manifests: bool = True
    copy_explicit_images: bool = True
    copy_explicit_pdfs: bool = False
    render_pdf_pages: bool = True
    max_images_per_case: int = 4
    render_zoom: float = 2.0
    max_pdf_search_pages: int = 40
    fallback_first_pages: int = 0
    download_papers_from_ids: bool = False
    unpaywall_email: str = ""
    request_timeout: int = 30


@dataclass
class BuildStats:
    manifests_discovered: int = 0
    cases_total: int = 0
    cases_written: int = 0
    cases_skipped_disabled: int = 0
    cases_skipped_incomplete: int = 0
    cases_with_images: int = 0
    explicit_images_copied: int = 0
    rendered_pages: int = 0
    pdfs_discovered: int = 0
    pdfs_downloaded: int = 0
    paper_download_errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def slugify(value: Any, *, fallback: str = "item", max_len: int = 90) -> str:
    text = str(value or "").strip().lower()
    # keep ASCII for portable file names
    try:
        import unicodedata

        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    except Exception:
        pass
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-._")
    return (text[:max_len].strip("-._") or fallback)


def stable_hash(value: Any, n: int = 12) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:n]


def read_json_or_yaml(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("PyYAML is required for YAML manifests") from exc
        return yaml.safe_load(text)
    return json.loads(text)


def is_task3_case_manifest(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if not isinstance(payload.get("cases"), list):
        return False
    schema = str(payload.get("schema_version") or "").lower()
    if "task3" in schema and "case" in schema:
        return True
    meta = payload.get("experiment_meta")
    return isinstance(meta, dict) and any("creator_prompt" in c for c in payload.get("cases") if isinstance(c, dict))


def safe_rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def iter_input_files(input_paths: Sequence[Path]) -> Iterable[Path]:
    for raw in input_paths:
        p = Path(raw)
        if not p.exists():
            continue
        if p.is_file():
            yield p
        elif p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file() and child.name != ".DS_Store":
                    yield child


def extract_archives(input_paths: Sequence[Path], work_dir: Path) -> list[Path]:
    expanded: list[Path] = []
    extracted_root = work_dir / "extracted_archives"
    extracted_root.mkdir(parents=True, exist_ok=True)
    for p in iter_input_files(input_paths):
        if p.suffix.lower() == ".zip" and zipfile.is_zipfile(p):
            target = extracted_root / f"{slugify(p.stem)}__{stable_hash(p)}"
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)
            try:
                with zipfile.ZipFile(p, "r") as zf:
                    zf.extractall(target)
                expanded.append(target)
            except Exception as exc:
                expanded.append(p)
        else:
            expanded.append(p)
    return expanded


def discover_manifests_and_assets(input_paths: Sequence[Path], work_dir: Path) -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    roots = extract_archives(input_paths, work_dir)
    manifest_paths: list[Path] = []
    image_paths: list[Path] = []
    pdf_paths: list[Path] = []
    other_paths: list[Path] = []

    seen: set[str] = set()
    for path in iter_input_files(roots):
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        suffix = path.suffix.lower()
        if suffix in SUPPORTED_MANIFEST_SUFFIXES:
            try:
                payload = read_json_or_yaml(path)
                if is_task3_case_manifest(payload):
                    manifest_paths.append(path)
                    continue
            except Exception:
                pass
        if suffix in SUPPORTED_IMAGE_SUFFIXES:
            image_paths.append(path)
        elif suffix in PDF_SUFFIXES:
            pdf_paths.append(path)
        else:
            other_paths.append(path)
    return sorted(manifest_paths), sorted(image_paths), sorted(pdf_paths), sorted(other_paths)


def normalize_prompt_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def adapt_creator_prompt_for_model(prompt: str) -> str:
    """Turn expert-comparison wording into a direct model-facing task.

    Expert manifests often say "Сравните, какой вариант лучше извлекает ...".
    A single VLM should not be asked to compare variants; it should perform the
    evidence extraction.  These conservative rewrites preserve the domain object
    of the task and leave already-direct prompts untouched.
    """
    p = normalize_prompt_text(prompt)
    replacements: list[tuple[str, str]] = [
        (r"^Сравните,\s*какой вариант лучше\s+извлекает\s+(.*)$", r"Извлеките \1."),
        (r"^Сравните,\s*какой вариант лучше\s+определяет\s+(.*)$", r"Определите \1."),
        (r"^Сравните,\s*какой вариант лучше\s+описывает\s+(.*)$", r"Опишите \1."),
        (r"^Сравнить,\s*какой вариант точнее\s+описывает\s+(.*)$", r"Опишите \1."),
        (r"^Сравните,\s*насколько правильно каждый VLM\s+извлекает\s+(.*)$", r"Извлеките \1."),
        (r"^Сравните,\s*как каждый VLM\s+определяет\s+(.*)$", r"Определите \1."),
        (r"^Сравните,\s*как каждый VLM\s+интерпретирует\s+(.*)$", r"Интерпретируйте \1."),
        (r"^Сравните,\s*как каждый VLM\s+связывает\s+(.*)$", r"Свяжите \1."),
        (r"^Сравните,\s*правильно ли каждый VLM\s+определяет\s+(.*)$", r"Определите \1."),
        (r"^Сравните,\s*сколько\s+(.*)$", r"Определите, сколько \1."),
    ]
    for pattern, replacement in replacements:
        if re.search(pattern, p, flags=re.IGNORECASE):
            out = re.sub(pattern, replacement, p, flags=re.IGNORECASE).strip()
            return out if out.endswith((".", "?", "!")) else out + "."
    return p


def build_user_prompt(case: Mapping[str, Any], manifest_meta: Mapping[str, Any]) -> str:
    adapted_task = adapt_creator_prompt_for_model(str(case.get("creator_prompt") or ""))
    lines = [
        "Задача для VLM по Task 3 benchmark.",
        "",
        f"Тема эксперта: {manifest_meta.get('topic') or ''}",
        f"Статья: {case.get('paper_title') or ''}",
        f"Идентификатор статьи: {case.get('paper_id') or ''}",
        f"Год: {case.get('year') or ''}",
        f"Тип evidence: {case.get('evidence_kind') or ''}",
        f"Подсказка по странице/рисунку/таблице: {case.get('page_hint') or ''}",
        f"Слой сложности: {case.get('stratum') or ''}",
        "",
        "Адаптированная задача:",
        adapted_task,
        "",
        "Используйте приложенные изображения страниц/рисунков/таблиц, если они есть. "
        "Если изображений нет или evidence недостаточно, явно укажите это в missing_evidence и uncertainty.",
        "",
        OUTPUT_SCHEMA_TEXT_RU,
    ]
    return "\n".join(lines).strip()


def parse_page_numbers(page_hint: str) -> list[int]:
    text = str(page_hint or "")
    pages: list[int] = []
    for match in re.finditer(r"(?:^|\b)(?:p|pp|page|pages|стр|страница|страницы)\.?:?\s*(\d{1,4})(?:\s*[-–]\s*(\d{1,4}))?", text, flags=re.IGNORECASE):
        a = int(match.group(1))
        b = int(match.group(2)) if match.group(2) else a
        if b < a:
            a, b = b, a
        pages.extend(range(a, min(b, a + 5) + 1))
    # handle compact hints like "p.2, Fig.1"
    for match in re.finditer(r"p\.\s*(\d{1,4})", text, flags=re.IGNORECASE):
        pages.append(int(match.group(1)))
    out: list[int] = []
    for p in pages:
        if p > 0 and p not in out:
            out.append(p)
    return out


def figure_table_labels(page_hint: str) -> list[str]:
    labels: list[str] = []
    text = str(page_hint or "")
    for match in re.finditer(r"\b(Fig(?:ure)?\.?|Рис\.?|Table|Tab\.?|Табл\.?)\s*(\d+[a-zA-Z]?)", text, flags=re.IGNORECASE):
        prefix = match.group(1).lower().replace(".", "")
        number = match.group(2)
        if prefix.startswith(("fig", "рис")):
            labels.extend([f"fig {number}", f"figure {number}", f"рис {number}"])
        else:
            labels.extend([f"table {number}", f"tab {number}", f"табл {number}"])
    return labels


def paper_id_candidates(paper_id: str) -> list[str]:
    text = str(paper_id or "").strip()
    if not text:
        return []
    values = [text]
    low = text.lower()
    if low.startswith("doi:"):
        values.append(text.split(":", 1)[1].strip())
    if low.startswith("arxiv:"):
        values.append(text.split(":", 1)[1].strip())
    doi = re.search(r"10\.\d{4,9}/[^\s]+", text, flags=re.IGNORECASE)
    if doi:
        values.append(doi.group(0).rstrip(".,;)"))
    arx = re.search(r"(?:arxiv[:\s]*)?([0-9]{4}\.[0-9]{4,5})(v\d+)?", text, flags=re.IGNORECASE)
    if arx:
        values.append(arx.group(1) + (arx.group(2) or ""))
    out: list[str] = []
    for v in values:
        v = v.strip()
        if v and v not in out:
            out.append(v)
    return out


def asset_match_score(path: Path, case: Mapping[str, Any]) -> int:
    text = " ".join([
        path.name.lower(),
        str(case.get("paper_id") or "").lower(),
        str(case.get("paper_title") or "").lower(),
        str(case.get("page_hint") or "").lower(),
    ])
    score = 0
    name = path.stem.lower()
    for cand in paper_id_candidates(str(case.get("paper_id") or "")):
        token = slugify(cand).lower()
        if token and token in slugify(name).lower():
            score += 10
    for token in re.split(r"[^a-zA-Z0-9]+", str(case.get("paper_title") or "").lower()):
        if len(token) >= 5 and token in name:
            score += 1
    for label in figure_table_labels(str(case.get("page_hint") or "")):
        compact = re.sub(r"[^a-z0-9]+", "", label.lower())
        if compact and compact in re.sub(r"[^a-z0-9]+", "", name):
            score += 5
    for p in parse_page_numbers(str(case.get("page_hint") or "")):
        if re.search(rf"(?:page|p|стр|_)0*{p}(?:\D|$)", name, flags=re.IGNORECASE):
            score += 3
    if score == 0 and any(x in text for x in ["figure", "fig", "table", "page", "рис", "таб"]):
        score += 1
    return score


def copy_image_assets(image_paths: Sequence[Path], case: Mapping[str, Any], sample_dir: Path, output_root: Path, max_images: int) -> list[str]:
    if max_images <= 0:
        return []
    ranked = sorted(((asset_match_score(p, case), p) for p in image_paths), key=lambda x: (-x[0], x[1].name))
    selected = [p for score, p in ranked if score > 0][:max_images]
    if not selected and image_paths:
        selected = list(sorted(image_paths))[: min(max_images, 1)]
    rels: list[str] = []
    sample_dir.mkdir(parents=True, exist_ok=True)
    for idx, src in enumerate(selected, start=1):
        dst = sample_dir / f"explicit_{idx:02d}_{slugify(src.stem)}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        rels.append(safe_rel(dst, output_root))
    return rels


def find_pdf_for_case(pdf_paths: Sequence[Path], case: Mapping[str, Any]) -> Path | None:
    if not pdf_paths:
        return None
    ranked = sorted(((asset_match_score(p, case), p) for p in pdf_paths), key=lambda x: (-x[0], x[1].name))
    if ranked and ranked[0][0] > 0:
        return ranked[0][1]
    if len(pdf_paths) == 1:
        return pdf_paths[0]
    return None


def render_pdf_pages_for_case(pdf_path: Path, case: Mapping[str, Any], sample_dir: Path, output_root: Path, *, max_images: int, zoom: float, max_search_pages: int, fallback_first_pages: int) -> list[str]:
    if max_images <= 0:
        return []
    try:
        import fitz  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyMuPDF (`pymupdf`) is required for PDF page rendering") from exc

    doc = fitz.open(str(pdf_path))
    if len(doc) == 0:
        return []
    wanted_pages = parse_page_numbers(str(case.get("page_hint") or ""))
    page_indices: list[int] = []
    for p in wanted_pages:
        idx = p - 1
        if 0 <= idx < len(doc) and idx not in page_indices:
            page_indices.append(idx)

    labels = figure_table_labels(str(case.get("page_hint") or ""))
    if labels and len(page_indices) < max_images:
        limit = min(len(doc), max_search_pages)
        for idx in range(limit):
            if idx in page_indices:
                continue
            try:
                txt = " ".join((doc[idx].get_text("text") or "").lower().split())
            except Exception:
                txt = ""
            if any(label in txt for label in labels):
                page_indices.append(idx)
                if len(page_indices) >= max_images:
                    break

    if not page_indices and fallback_first_pages > 0:
        page_indices = list(range(min(len(doc), fallback_first_pages, max_images)))

    page_indices = page_indices[:max_images]
    sample_dir.mkdir(parents=True, exist_ok=True)
    rels: list[str] = []
    matrix = fitz.Matrix(float(zoom), float(zoom))
    for idx in page_indices:
        page = doc[idx]
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        dst = sample_dir / f"rendered_page_{idx+1:04d}.png"
        pix.save(str(dst))
        rels.append(safe_rel(dst, output_root))
    return rels


def direct_pdf_url_for_paper_id(paper_id: str) -> str | None:
    text = str(paper_id or "").strip()
    low = text.lower()
    if not text:
        return None
    if low.startswith("http") and ".pdf" in low:
        return text
    arx = re.search(r"(?:arxiv[:\s]*)?([0-9]{4}\.[0-9]{4,5})(v\d+)?", text, flags=re.IGNORECASE)
    if arx:
        return f"https://arxiv.org/pdf/{arx.group(1)}{arx.group(2) or ''}.pdf"
    return None


def resolve_unpaywall_pdf_url(doi: str, email: str, timeout: int = 30) -> str | None:
    if not doi or not email:
        return None
    try:
        import requests
    except Exception:
        return None
    doi_clean = doi.strip().removeprefix("doi:").strip()
    url = f"https://api.unpaywall.org/v2/{doi_clean}"
    response = requests.get(url, params={"email": email}, timeout=timeout, headers={"User-Agent": "top-papers-graph-task3-benchmark/1.0"})
    response.raise_for_status()
    payload = response.json()
    candidates: list[str] = []
    best = payload.get("best_oa_location") if isinstance(payload, dict) else None
    if isinstance(best, dict):
        for key in ("url_for_pdf", "url"):
            if best.get(key):
                candidates.append(str(best[key]))
    for loc in payload.get("oa_locations") or []:
        if isinstance(loc, dict):
            for key in ("url_for_pdf", "url"):
                if loc.get(key):
                    candidates.append(str(loc[key]))
    for candidate in candidates:
        if candidate and candidate not in {"None", "null"}:
            return candidate
    return None


def download_file(url: str, dst: Path, timeout: int = 30) -> Path:
    try:
        import requests
    except Exception as exc:
        raise RuntimeError("requests is required for file download") from exc
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout, headers={"User-Agent": "top-papers-graph-task3-benchmark/1.0"}) as response:
        response.raise_for_status()
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        with tmp.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 512):
                if chunk:
                    f.write(chunk)
        tmp.replace(dst)
    return dst


def maybe_download_pdf_for_case(case: Mapping[str, Any], download_dir: Path, *, unpaywall_email: str, timeout: int, stats: BuildStats) -> Path | None:
    paper_id = str(case.get("paper_id") or "").strip()
    if not paper_id:
        return None
    candidates: list[str] = []
    direct = direct_pdf_url_for_paper_id(paper_id)
    if direct:
        candidates.append(direct)
    doi_match = re.search(r"10\.\d{4,9}/[^\s]+", paper_id, flags=re.IGNORECASE)
    if doi_match and unpaywall_email:
        try:
            resolved = resolve_unpaywall_pdf_url(doi_match.group(0).rstrip(".,;)"), unpaywall_email, timeout=timeout)
            if resolved:
                candidates.append(resolved)
        except Exception as exc:
            stats.paper_download_errors.append({"paper_id": paper_id, "stage": "unpaywall", "error": repr(exc)})
    if not candidates:
        return None
    stem = slugify(paper_id, fallback="paper") + "__" + stable_hash(paper_id)
    dst = download_dir / f"{stem}.pdf"
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    for url in candidates:
        try:
            download_file(url, dst, timeout=timeout)
            stats.pdfs_downloaded += 1
            return dst
        except Exception as exc:
            stats.paper_download_errors.append({"paper_id": paper_id, "url": url, "error": repr(exc)})
    return None


def make_generation_record(
    *,
    manifest_path: Path,
    manifest_payload: Mapping[str, Any],
    case: Mapping[str, Any],
    output_root: Path,
    image_paths: Sequence[Path],
    pdf_paths: Sequence[Path],
    config: BuildConfig,
    stats: BuildStats,
) -> dict[str, Any] | None:
    meta = manifest_payload.get("experiment_meta") if isinstance(manifest_payload.get("experiment_meta"), dict) else {}
    creator_id = str(meta.get("creator_id") or "")
    submission_id = str(meta.get("submission_id") or "") or f"manifest__{stable_hash(manifest_path)}"
    case_id = str(case.get("case_id") or f"case__{stable_hash(case)}")
    prompt = str(case.get("creator_prompt") or "").strip()
    paper_id = str(case.get("paper_id") or "").strip()
    paper_title = str(case.get("paper_title") or "").strip()

    if not config.include_incomplete_cases and not (prompt and (paper_id or paper_title)):
        stats.cases_skipped_incomplete += 1
        return None

    sample_id = f"task3:{slugify(submission_id, fallback='submission')}:{slugify(case_id, fallback='case')}"
    sample_dir = output_root / "assets" / "images" / slugify(submission_id, fallback="submission") / slugify(case_id, fallback="case")
    images: list[str] = []
    if config.copy_explicit_images:
        copied = copy_image_assets(image_paths, case, sample_dir, output_root, config.max_images_per_case)
        images.extend(copied)
        stats.explicit_images_copied += len(copied)

    if config.render_pdf_pages and len(images) < config.max_images_per_case:
        pdf = find_pdf_for_case(pdf_paths, case)
        if pdf is None and config.download_papers_from_ids:
            pdf = maybe_download_pdf_for_case(
                case,
                output_root / "_work" / "downloaded_pdfs",
                unpaywall_email=config.unpaywall_email,
                timeout=config.request_timeout,
                stats=stats,
            )
        if pdf is not None:
            try:
                rendered = render_pdf_pages_for_case(
                    pdf,
                    case,
                    sample_dir,
                    output_root,
                    max_images=max(0, config.max_images_per_case - len(images)),
                    zoom=config.render_zoom,
                    max_search_pages=config.max_pdf_search_pages,
                    fallback_first_pages=config.fallback_first_pages,
                )
                images.extend(rendered)
                stats.rendered_pages += len(rendered)
            except Exception as exc:
                stats.warnings.append(f"Failed to render {pdf} for {sample_id}: {exc!r}")
            if config.copy_explicit_pdfs and pdf.exists():
                pdf_dir = output_root / "assets" / "papers"
                pdf_dir.mkdir(parents=True, exist_ok=True)
                pdf_dst = pdf_dir / f"{slugify(sample_id)}.pdf"
                if pdf.resolve() != pdf_dst.resolve():
                    shutil.copy2(pdf, pdf_dst)

    user_text = build_user_prompt(case, meta)
    user_content = [{"type": "text", "text": user_text}]
    user_content.extend({"type": "image"} for _ in images)

    record = {
        "sample_id": sample_id,
        "benchmark_version": "task3_hf_benchmark_v1",
        "task_family": "task3_vlm_ab_generation",
        "language": "ru",
        "split": config.split_name,
        "topic": str(meta.get("topic") or ""),
        "submission_id": submission_id,
        "creator_id": creator_id,
        "case_id": case_id,
        "stratum": str(case.get("stratum") or ""),
        "primary_endpoint": bool(case.get("primary_endpoint")),
        "paper_title": paper_title,
        "paper_id": paper_id,
        "year": str(case.get("year") or ""),
        "evidence_kind": str(case.get("evidence_kind") or ""),
        "page_hint": str(case.get("page_hint") or ""),
        "model_task_prompt": adapt_creator_prompt_for_model(prompt),
        "input_text": user_text,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_RU}]},
            {"role": "user", "content": user_content},
        ],
        "images": images,
        "generation_target_schema": {
            "answer": "string",
            "evidence_used": "list[{kind, locator, description}]",
            "visual_facts": "list[string]",
            "temporal_facts": "list[string]",
            "uncertainty": "low|medium|high",
            "missing_evidence": "list[string]",
        },
        "review_metadata": {
            "review_focus": list(case.get("review_focus") or []),
            "expected_error_modes": list(case.get("expected_error_modes") or []),
            "creator_rationale_available": bool(str(case.get("creator_rationale") or "").strip()),
            "source_manifest": safe_rel(manifest_path, manifest_path.parent),
            "original_creator_prompt": prompt,
            "notes_available": bool(str(case.get("notes") or "").strip()),
        },
    }
    if images:
        stats.cases_with_images += 1
    return record


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "topic",
        "submission_id",
        "case_id",
        "stratum",
        "primary_endpoint",
        "paper_title",
        "paper_id",
        "year",
        "evidence_kind",
        "page_hint",
        "image_count",
        "model_task_prompt",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (len(row.get("images") or []) if k == "image_count" else row.get(k, "")) for k in fields})


def dataset_card_content(repo_id: str, config_name: str, split_name: str, stats: Mapping[str, Any]) -> str:
    size_category = "n<1K"
    n = int(stats.get("cases_written", 0) or 0)
    if n >= 100000:
        size_category = "n>100K"
    elif n >= 10000:
        size_category = "10K<n<100K"
    elif n >= 1000:
        size_category = "1K<n<10K"
    return f"""---
pretty_name: top-papers-graph Task 3 VLM A/B benchmark
language:
- ru
tags:
- scientific-papers
- vision-language
- multimodal
- benchmark
- ab-testing
- top-papers-graph
size_categories:
- {size_category}
configs:
- config_name: {config_name}
  data_files:
  - split: {split_name}
    path: data/task3_vlm_generation.jsonl
- config_name: task3_cases_flat
  data_files:
  - split: {split_name}
    path: data/task3_cases_flat.jsonl
---

# top-papers-graph Task 3 VLM A/B benchmark

Dataset repository target: `{repo_id}`.

This dataset contains Task 3 case-based prompts adapted for **single VLM generation**. Expert-created A/B review prompts are transformed into model-facing evidence extraction or reasoning tasks. Expert rationales and expected error modes are kept in metadata, not in the model messages, to reduce prompt leakage during A/B generation.

## Main files

- `data/task3_vlm_generation.jsonl` — primary model-facing records.
- `data/task3_cases_flat.jsonl` — compact case metadata for auditing.
- `metadata/build_summary.json` — build statistics and warnings.
- `review_metadata/task3_case_rationales.jsonl` — expert-only rationale/diagnostic metadata; do not feed this file to the evaluated VLM.
- `assets/images/` — optional rendered pages or explicitly supplied image files referenced by `images`.

## Loading

```python
from datasets import load_dataset
from huggingface_hub import snapshot_download
from pathlib import Path

repo_id = "{repo_id}"
ds = load_dataset(repo_id, "{config_name}", split="{split_name}")
repo_root = Path(snapshot_download(repo_id, repo_type="dataset"))
row = ds[0]
image_paths = [repo_root / rel for rel in row["images"]]
messages = row["messages"]
```

Each row follows the TRL-style multimodal convention used elsewhere in this repository: `messages` contains text and `{{"type": "image"}}` placeholders, while top-level `images` contains relative paths in the same order.

## Build summary

- Cases written: {stats.get('cases_written', 0)}
- Cases with images: {stats.get('cases_with_images', 0)}
- Manifests discovered: {stats.get('manifests_discovered', 0)}
- Rendered PDF pages: {stats.get('rendered_pages', 0)}
"""


def build_dataset(config: BuildConfig) -> dict[str, Any]:
    output_root = config.output_dir.resolve()
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    work_dir = output_root / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths, image_paths, pdf_paths, other_paths = discover_manifests_and_assets(config.input_paths, work_dir)
    stats = BuildStats(manifests_discovered=len(manifest_paths), pdfs_discovered=len(pdf_paths))

    records: list[dict[str, Any]] = []
    flat_cases: list[dict[str, Any]] = []
    rationales: list[dict[str, Any]] = []

    raw_manifest_dir = output_root / "source_manifests"
    if config.copy_source_manifests:
        raw_manifest_dir.mkdir(parents=True, exist_ok=True)

    for manifest_path in manifest_paths:
        try:
            payload = read_json_or_yaml(manifest_path)
        except Exception as exc:
            stats.warnings.append(f"Cannot read manifest {manifest_path}: {exc!r}")
            continue
        if not is_task3_case_manifest(payload):
            continue
        meta = payload.get("experiment_meta") if isinstance(payload.get("experiment_meta"), dict) else {}
        submission_id = str(meta.get("submission_id") or f"manifest__{stable_hash(manifest_path)}")
        if config.copy_source_manifests:
            dst = raw_manifest_dir / f"{slugify(submission_id, fallback='submission')}__{manifest_path.name}"
            try:
                shutil.copy2(manifest_path, dst)
            except Exception:
                pass
        for case in payload.get("cases") or []:
            if not isinstance(case, dict):
                continue
            stats.cases_total += 1
            if not config.include_disabled_cases and not case.get("enabled", True):
                stats.cases_skipped_disabled += 1
                continue
            rec = make_generation_record(
                manifest_path=manifest_path,
                manifest_payload=payload,
                case=case,
                output_root=output_root,
                image_paths=image_paths,
                pdf_paths=pdf_paths,
                config=config,
                stats=stats,
            )
            flat = {
                "submission_id": submission_id,
                "creator_id": str(meta.get("creator_id") or ""),
                "topic": str(meta.get("topic") or ""),
                "case_id": str(case.get("case_id") or ""),
                "enabled": bool(case.get("enabled", True)),
                "primary_endpoint": bool(case.get("primary_endpoint")),
                "stratum": str(case.get("stratum") or ""),
                "paper_title": str(case.get("paper_title") or ""),
                "paper_id": str(case.get("paper_id") or ""),
                "year": str(case.get("year") or ""),
                "evidence_kind": str(case.get("evidence_kind") or ""),
                "page_hint": str(case.get("page_hint") or ""),
                "creator_prompt": str(case.get("creator_prompt") or ""),
                "model_task_prompt": adapt_creator_prompt_for_model(str(case.get("creator_prompt") or "")),
                "review_focus": list(case.get("review_focus") or []),
            }
            flat_cases.append(flat)
            rationales.append({
                **flat,
                "creator_rationale": str(case.get("creator_rationale") or ""),
                "expected_error_modes": list(case.get("expected_error_modes") or []),
                "match": case.get("match") if isinstance(case.get("match"), dict) else {},
                "notes": str(case.get("notes") or ""),
            })
            if rec is not None:
                records.append(rec)
                stats.cases_written += 1

    data_dir = output_root / "data"
    write_jsonl(data_dir / "task3_vlm_generation.jsonl", records)
    write_jsonl(data_dir / "task3_cases_flat.jsonl", flat_cases)
    write_csv_summary(data_dir / "task3_cases_summary.csv", records)
    write_jsonl(output_root / "review_metadata" / "task3_case_rationales.jsonl", rationales)

    stats_payload = {
        "generated_at": utc_now(),
        "dataset_repo_id": config.dataset_repo_id,
        "dataset_config_name": config.dataset_config_name,
        "split_name": config.split_name,
        "input_paths": [str(p) for p in config.input_paths],
        "manifests": [str(p) for p in manifest_paths],
        "explicit_images": [str(p) for p in image_paths],
        "explicit_pdfs": [str(p) for p in pdf_paths],
        "other_files_discovered": len(other_paths),
        **stats.__dict__,
    }
    (output_root / "metadata").mkdir(parents=True, exist_ok=True)
    (output_root / "metadata" / "build_summary.json").write_text(json.dumps(stats_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "README.md").write_text(dataset_card_content(config.dataset_repo_id, config.dataset_config_name, config.split_name, stats_payload), encoding="utf-8")
    (output_root / ".gitattributes").write_text("*.jsonl filter=lfs diff=lfs merge=lfs -text\n*.png filter=lfs diff=lfs merge=lfs -text\n*.jpg filter=lfs diff=lfs merge=lfs -text\n*.jpeg filter=lfs diff=lfs merge=lfs -text\n*.webp filter=lfs diff=lfs merge=lfs -text\n*.pdf filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")

    # Remove temporary files unless downloaded PDFs are useful for debugging.
    try:
        shutil.rmtree(work_dir / "extracted_archives", ignore_errors=True)
    except Exception:
        pass
    return stats_payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Hugging Face-ready Task 3 VLM benchmark dataset from creator manifests.")
    parser.add_argument("--input", dest="inputs", action="append", required=True, help="Input file or directory. Can be passed multiple times.")
    parser.add_argument("--out-dir", required=True, help="Output dataset folder.")
    parser.add_argument("--dataset-repo-id", default="top-papers/top-papers-graph-benchmark")
    parser.add_argument("--dataset-config-name", default="task3_vlm_generation")
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--include-disabled-cases", action="store_true")
    parser.add_argument("--include-incomplete-cases", action="store_true")
    parser.add_argument("--no-copy-source-manifests", action="store_true")
    parser.add_argument("--no-copy-explicit-images", action="store_true")
    parser.add_argument("--copy-explicit-pdfs", action="store_true")
    parser.add_argument("--no-render-pdf-pages", action="store_true")
    parser.add_argument("--max-images-per-case", type=int, default=4)
    parser.add_argument("--render-zoom", type=float, default=2.0)
    parser.add_argument("--max-pdf-search-pages", type=int, default=40)
    parser.add_argument("--fallback-first-pages", type=int, default=0)
    parser.add_argument("--download-papers-from-ids", action="store_true")
    parser.add_argument("--unpaywall-email", default="")
    parser.add_argument("--request-timeout", type=int, default=30)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = BuildConfig(
        input_paths=tuple(Path(x) for x in args.inputs),
        output_dir=Path(args.out_dir),
        dataset_repo_id=args.dataset_repo_id,
        dataset_config_name=args.dataset_config_name,
        split_name=args.split_name,
        include_disabled_cases=bool(args.include_disabled_cases),
        include_incomplete_cases=bool(args.include_incomplete_cases),
        copy_source_manifests=not bool(args.no_copy_source_manifests),
        copy_explicit_images=not bool(args.no_copy_explicit_images),
        copy_explicit_pdfs=bool(args.copy_explicit_pdfs),
        render_pdf_pages=not bool(args.no_render_pdf_pages),
        max_images_per_case=int(args.max_images_per_case),
        render_zoom=float(args.render_zoom),
        max_pdf_search_pages=int(args.max_pdf_search_pages),
        fallback_first_pages=int(args.fallback_first_pages),
        download_papers_from_ids=bool(args.download_papers_from_ids),
        unpaywall_email=str(args.unpaywall_email or ""),
        request_timeout=int(args.request_timeout),
    )
    stats = build_dataset(config)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if int(stats.get("cases_written", 0) or 0) <= 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
