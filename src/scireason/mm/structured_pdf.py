from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..config import settings
from ..ingest.chunking import simple_chunks
from .pdf_mm_extract import extract_pages as fallback_extract_pages
from .vlm import describe_image


@dataclass
class StructuredChunk:
    """Universal multimodal chunk used by the expert pipeline.

    The repository historically stored only plain text chunks (`chunks.jsonl`) and page-level
    multimodal artifacts (`mm/pages.jsonl`). For the expert task we need a richer, *reviewable*
    unit that keeps page / modality / figure-table provenance in one place.
    """

    chunk_id: str
    paper_id: str
    modality: str  # text | table | figure | page
    text: str = ""
    page: Optional[int] = None  # human-friendly page number (1-based)
    order: int = 0
    image_path: Optional[str] = None
    figure_or_table: Optional[str] = None
    table_markdown: Optional[str] = None
    section: Optional[str] = None
    summary: str = ""
    backend: str = "docling"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def searchable_text(self) -> str:
        parts = [self.text or "", self.summary or "", self.table_markdown or ""]
        return "\n\n".join([p.strip() for p in parts if p and p.strip()]).strip()

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["searchable_text"] = self.searchable_text()
        return data


DOCS_FRIENDLY_PAGE_FIELDS = ("page_no", "page", "page_num")


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _human_page(value: Any) -> Optional[int]:
    page = _safe_int(value)
    if page is None:
        return None
    # Docling pages are 1-based; fallback page renderers in this repo were 0-based.
    # We normalize to 1-based for expert-facing evidence references.
    return page if page >= 1 else page + 1


def _infer_page_from_element(element: Any) -> Optional[int]:
    for name in DOCS_FRIENDLY_PAGE_FIELDS:
        if hasattr(element, name):
            page = _human_page(getattr(element, name))
            if page is not None:
                return page

    prov = getattr(element, "prov", None)
    if prov:
        try:
            first = prov[0]
            for name in DOCS_FRIENDLY_PAGE_FIELDS:
                if hasattr(first, name):
                    page = _human_page(getattr(first, name))
                    if page is not None:
                        return page
        except Exception:
            pass
    return None


def _element_text(element: Any) -> str:
    candidates: list[str] = []
    # Common attributes on Docling items.
    for name in (
        "text",
        "caption_text",
        "caption",
        "label",
        "title",
        "content",
        "alt_text",
        "description",
    ):
        try:
            value = getattr(element, name, None)
        except Exception:
            value = None
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    candidates.append(item.strip())
        elif value is not None and hasattr(value, "text"):
            try:
                txt = str(getattr(value, "text") or "").strip()
                if txt:
                    candidates.append(txt)
            except Exception:
                pass
    if candidates:
        return "\n".join(candidates)
    return ""


def _save_pil_image(image_obj: Any, path: Path) -> Optional[str]:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        image_obj.save(path, format="PNG")
        return str(path.as_posix())
    except Exception:
        return None


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_structured_chunks(paper_dir: Path) -> List[StructuredChunk]:
    path = paper_dir / "structured_chunks.jsonl"
    if not path.exists():
        return []
    out: list[StructuredChunk] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            out.append(
                StructuredChunk(
                    chunk_id=str(raw.get("chunk_id") or ""),
                    paper_id=str(raw.get("paper_id") or raw.get("metadata", {}).get("paper_id") or ""),
                    modality=str(raw.get("modality") or "text"),
                    text=str(raw.get("text") or ""),
                    page=_safe_int(raw.get("page")),
                    order=int(raw.get("order") or 0),
                    image_path=raw.get("image_path"),
                    figure_or_table=raw.get("figure_or_table"),
                    table_markdown=raw.get("table_markdown"),
                    section=raw.get("section"),
                    summary=str(raw.get("summary") or ""),
                    backend=str(raw.get("backend") or "docling"),
                    metadata=dict(raw.get("metadata") or {}),
                )
            )
        except Exception:
            continue
    return out


def _docling_extract(
    pdf_path: Path,
    paper_id: str,
    out_dir: Path,
    *,
    prompt_context: str = "",
    run_vlm: bool = True,
) -> List[StructuredChunk]:
    # Lazy import: Docling is optional and quite heavy.
    from docling.document_converter import DocumentConverter, PdfFormatOption  # type: ignore
    from docling.datamodel.base_models import InputFormat  # type: ignore
    from docling.datamodel.pipeline_options import PdfPipelineOptions  # type: ignore
    from docling.utils.export import generate_multimodal_pages  # type: ignore
    from docling_core.types.doc import PictureItem  # type: ignore

    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = max(1.0, float(getattr(settings, "pdf_render_dpi", 150) or 150) / 72.0)
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    if hasattr(pipeline_options, "generate_table_images"):
        pipeline_options.generate_table_images = True
    if hasattr(pipeline_options, "do_table_structure"):
        pipeline_options.do_table_structure = True
    if hasattr(pipeline_options, "do_ocr"):
        pipeline_options.do_ocr = True
    if hasattr(pipeline_options, "do_formula_enrichment"):
        pipeline_options.do_formula_enrichment = True
    if hasattr(pipeline_options, "force_backend_text"):
        pipeline_options.force_backend_text = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
    conv_res = converter.convert(pdf_path)
    doc = conv_res.document

    mm_root = out_dir / "mm"
    page_img_dir = mm_root / "images"
    elem_img_dir = mm_root / "elements"
    tables_dir = mm_root / "tables"
    page_img_dir.mkdir(parents=True, exist_ok=True)
    elem_img_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[StructuredChunk] = []

    # --- per-page multimodal records ---
    page_rows: list[dict[str, Any]] = []
    for order, page_bundle in enumerate(generate_multimodal_pages(conv_res)):
        try:
            content_text, content_md, content_dt, page_cells, page_segments, page = page_bundle
        except Exception:
            # API compatibility fallback
            continue
        page_no = _human_page(getattr(page, "page_no", order + 1)) or (order + 1)
        page_path = page_img_dir / f"page_{page_no:03d}.png"
        image_path = None
        try:
            image_obj = getattr(page, "image", None)
            if image_obj is not None:
                if hasattr(image_obj, "pil_image"):
                    image_path = _save_pil_image(image_obj.pil_image, page_path)
                else:
                    image_path = _save_pil_image(image_obj, page_path)
        except Exception:
            image_path = None

        page_text = str(content_text or "").strip()
        page_md = str(content_md or "").strip()
        page_summary = page_text[:500]
        page_chunk = StructuredChunk(
            chunk_id=f"{paper_id}:page:{page_no}",
            paper_id=paper_id,
            modality="page",
            text=page_text,
            page=page_no,
            order=order,
            image_path=image_path,
            summary=page_summary,
            backend="docling",
            metadata={
                "markdown": page_md,
                "cells": len(page_cells) if page_cells is not None else 0,
                "segments": len(page_segments) if page_segments is not None else 0,
                "doc_tags": str(content_dt or "")[:1500],
            },
        )
        chunks.append(page_chunk)
        page_rows.append(
            {
                "paper_id": paper_id,
                "page": page_no,
                "text": page_text,
                "image_path": image_path or "",
                "vlm_caption": "",
                "tables_md": None,
                "equations_md": None,
            }
        )

        # Text chunks keep page provenance for downstream evidence cards.
        for j, text_chunk in enumerate(simple_chunks(page_text, max_chars=1500, overlap=200)):
            chunks.append(
                StructuredChunk(
                    chunk_id=f"{paper_id}:text:p{page_no}:{j}",
                    paper_id=paper_id,
                    modality="text",
                    text=text_chunk,
                    page=page_no,
                    order=order * 100 + j,
                    summary=text_chunk[:400],
                    backend="docling",
                    metadata={"source": "page_text", "markdown": page_md[:1500]},
                )
            )

    # --- tables ---
    try:
        import pandas as pd  # type: ignore  # noqa: F401
    except Exception:
        pd = None  # type: ignore

    for tix, table in enumerate(getattr(doc, "tables", []) or [], start=1):
        page_no = _infer_page_from_element(table)
        md = ""
        html = ""
        csv_path = None
        html_path = None
        try:
            if hasattr(table, "export_to_dataframe"):
                df = table.export_to_dataframe(doc=doc)
                # index=False is better for human-facing evidence snippets.
                try:
                    md = df.to_markdown(index=False)
                except Exception:
                    md = df.to_csv(index=False)
                csv_out = tables_dir / f"table_{tix:03d}.csv"
                df.to_csv(csv_out, index=False)
                csv_path = str(csv_out.as_posix())
        except Exception:
            md = ""
        try:
            if hasattr(table, "export_to_html"):
                html = str(table.export_to_html(doc=doc) or "")
                html_out = tables_dir / f"table_{tix:03d}.html"
                html_out.write_text(html, encoding="utf-8")
                html_path = str(html_out.as_posix())
        except Exception:
            html = ""
        image_path = None
        try:
            if hasattr(table, "get_image"):
                img = table.get_image(doc)
                image_path = _save_pil_image(img, elem_img_dir / f"table_{tix:03d}.png")
        except Exception:
            image_path = None

        table_text = md or html or _element_text(table)
        if not table_text.strip() and image_path and run_vlm and getattr(settings, "vlm_backend", "none") != "none":
            try:
                table_vlm = describe_image(
                    image_path=Path(image_path),
                    prompt=(prompt_context or "Scientific table") + ". Extract the table semantics and key trends.",
                )
                table_text = (table_vlm.extracted_tables_md or table_vlm.caption or "").strip()
            except Exception:
                pass

        chunks.append(
            StructuredChunk(
                chunk_id=f"{paper_id}:table:{tix}",
                paper_id=paper_id,
                modality="table",
                text=table_text,
                page=page_no,
                order=10_000 + tix,
                image_path=image_path,
                figure_or_table=f"Table {tix}",
                table_markdown=md or None,
                summary=(table_text or md or "")[:500],
                backend="docling",
                metadata={
                    "csv_path": csv_path,
                    "html_path": html_path,
                    "raw_element_text": _element_text(table),
                },
            )
        )

    # --- figures / pictures ---
    pic_counter = 0
    try:
        for element, _level in doc.iterate_items():
            if not isinstance(element, PictureItem):
                continue
            pic_counter += 1
            page_no = _infer_page_from_element(element)
            image_path = None
            try:
                if hasattr(element, "get_image"):
                    image_path = _save_pil_image(element.get_image(doc), elem_img_dir / f"figure_{pic_counter:03d}.png")
            except Exception:
                image_path = None

            raw_text = _element_text(element)
            vlm_text = ""
            if image_path and run_vlm and getattr(settings, "vlm_backend", "none") != "none":
                try:
                    vlm_res = describe_image(
                        image_path=Path(image_path),
                        prompt=(prompt_context or "Scientific figure")
                        + ". Describe the figure, chart or diagram; mention axes, variables, and conclusions."
                    )
                    vlm_text = "\n".join(
                        [x for x in [vlm_res.caption, vlm_res.extracted_tables_md, vlm_res.extracted_equations_md] if x]
                    ).strip()
                except Exception:
                    vlm_text = ""

            fig_text = "\n\n".join([x for x in [raw_text, vlm_text] if x]).strip()
            if not fig_text:
                # Avoid empty searchable chunks; still preserve the figure node.
                fig_text = f"Figure {pic_counter} from page {page_no or '?'}"

            chunks.append(
                StructuredChunk(
                    chunk_id=f"{paper_id}:figure:{pic_counter}",
                    paper_id=paper_id,
                    modality="figure",
                    text=fig_text,
                    page=page_no,
                    order=20_000 + pic_counter,
                    image_path=image_path,
                    figure_or_table=f"Figure {pic_counter}",
                    summary=fig_text[:500],
                    backend="docling",
                    metadata={"raw_element_text": raw_text},
                )
            )
    except Exception:
        # Some Docling versions may not expose PictureItem / iterate_items the same way.
        pass

    _write_jsonl(out_dir / "structured_chunks.jsonl", [c.to_dict() for c in chunks])
    _write_jsonl(out_dir / "mm" / "pages.jsonl", page_rows)
    return chunks


def _fallback_extract(
    pdf_path: Path,
    paper_id: str,
    out_dir: Path,
    *,
    prompt_context: str = "",
    run_vlm: bool = True,
) -> List[StructuredChunk]:
    pages = fallback_extract_pages(
        pdf_path=pdf_path,
        paper_id=paper_id,
        out_dir=out_dir,
        prompt_context=prompt_context,
        run_vlm=run_vlm,
    )
    chunks: list[StructuredChunk] = []
    for order, page in enumerate(pages, start=1):
        page_no = _human_page(getattr(page, "page", order)) or order
        page_chunk = StructuredChunk(
            chunk_id=f"{paper_id}:page:{page_no}",
            paper_id=paper_id,
            modality="page",
            text=str(page.text or ""),
            page=page_no,
            order=order,
            image_path=page.image_path,
            summary=(str(page.vlm_caption or "") or str(page.text or ""))[:500],
            backend="pymupdf",
            metadata={
                "vlm_caption": page.vlm_caption,
                "tables_md": page.tables_md,
                "equations_md": page.equations_md,
            },
        )
        chunks.append(page_chunk)

        for j, text_chunk in enumerate(simple_chunks(str(page.text or ""), max_chars=1500, overlap=200)):
            chunks.append(
                StructuredChunk(
                    chunk_id=f"{paper_id}:text:p{page_no}:{j}",
                    paper_id=paper_id,
                    modality="text",
                    text=text_chunk,
                    page=page_no,
                    order=order * 100 + j,
                    summary=text_chunk[:400],
                    backend="pymupdf",
                    metadata={"source": "page_text"},
                )
            )

        if page.tables_md:
            chunks.append(
                StructuredChunk(
                    chunk_id=f"{paper_id}:table:p{page_no}",
                    paper_id=paper_id,
                    modality="table",
                    text=str(page.tables_md),
                    page=page_no,
                    order=10_000 + order,
                    image_path=page.image_path,
                    figure_or_table=f"Table page {page_no}",
                    table_markdown=page.tables_md,
                    summary=str(page.tables_md)[:500],
                    backend="pymupdf",
                )
            )

        if page.image_path and page.vlm_caption:
            chunks.append(
                StructuredChunk(
                    chunk_id=f"{paper_id}:figure:p{page_no}",
                    paper_id=paper_id,
                    modality="figure",
                    text=str(page.vlm_caption),
                    page=page_no,
                    order=20_000 + order,
                    image_path=page.image_path,
                    figure_or_table=f"Figure page {page_no}",
                    summary=str(page.vlm_caption)[:500],
                    backend="pymupdf",
                    metadata={"equations_md": page.equations_md},
                )
            )

    _write_jsonl(out_dir / "structured_chunks.jsonl", [c.to_dict() for c in chunks])
    return chunks


def extract_structured_pdf(
    pdf_path: Path,
    paper_id: str,
    out_dir: Path,
    *,
    prompt_context: str = "",
    run_vlm: bool = True,
    prefer_docling: bool = True,
) -> List[StructuredChunk]:
    """Extract a multimodal, review-friendly chunk inventory from a PDF.

    Priority:
    1) Docling (best accuracy for text / layout / tables / figure export)
    2) existing PyMuPDF page renderer fallback
    """

    if prefer_docling:
        try:
            return _docling_extract(
                pdf_path=pdf_path,
                paper_id=paper_id,
                out_dir=out_dir,
                prompt_context=prompt_context,
                run_vlm=run_vlm,
            )
        except Exception:
            # Silent fallback: the rest of the pipeline still works via page-level extraction.
            pass

    return _fallback_extract(
        pdf_path=pdf_path,
        paper_id=paper_id,
        out_dir=out_dir,
        prompt_context=prompt_context,
        run_vlm=run_vlm,
    )
