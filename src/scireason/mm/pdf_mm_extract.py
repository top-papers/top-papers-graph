from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..config import settings
from rich.console import Console

from .vlm import describe_image, VLMResult


console = Console()


@dataclass
class PageRecord:
    paper_id: str
    page: int
    text: str
    image_path: str
    vlm_caption: str = ""
    tables_md: Optional[str] = None
    equations_md: Optional[str] = None


def _require(pkg: str) -> None:
    raise RuntimeError(
        f"Для извлечения мультимодальности из PDF нужна зависимость '{pkg}'.\n"
        "Установите extras: pip install -e '.[mm]'\n"
    )


def extract_pages(
    pdf_path: Path,
    paper_id: str,
    out_dir: Path,
    prompt_context: str = "",
    dpi: Optional[int] = None,
    run_vlm: bool = True,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[PageRecord]:
    """Извлекает из PDF:
    - текст по страницам
    - рендер каждой страницы в PNG
    - (опционально) подпись/таблицы/формулы через VL-модель

    Результат пишется в out_dir/mm/ (поддиректории images/ и pages.jsonl).
    """
    dpi = dpi or getattr(settings, "pdf_render_dpi", 150)
    mm_root = out_dir / "mm"
    img_dir = mm_root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    try:
        import fitz  # PyMuPDF  # type: ignore
    except Exception:
        _require("pymupdf")

    doc = fitz.open(str(pdf_path))
    pages: List[PageRecord] = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    total_pages = len(doc)

    if progress_callback is not None:
        progress_callback({
            "event": "start",
            "paper_id": paper_id,
            "pdf_path": str(pdf_path),
            "current": 0,
            "total": total_pages,
            "message": f"Страницы PDF: 0/{total_pages}",
        })

    for i in range(total_pages):
        page = doc.load_page(i)
        console.print(f"[cyan]MM page {i + 1}/{total_pages}:[/cyan] {pdf_path.name}")
        text = (page.get_text("text") or "").strip()

        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img_path = img_dir / f"page_{i:03d}.png"
        pix.save(str(img_path))

        rec = PageRecord(
            paper_id=paper_id,
            page=i,
            text=text,
            image_path=str(img_path.as_posix()),
        )

        if run_vlm and getattr(settings, "vlm_backend", "none") != "none":
            try:
                res: VLMResult = describe_image(
                    image_path=img_path,
                    prompt=prompt_context or "Извлеки смысл страницы научной статьи (графики/таблицы/формулы)",
                )
                rec.vlm_caption = res.caption
                rec.tables_md = res.extracted_tables_md
                rec.equations_md = res.extracted_equations_md
            except Exception as e:  # pragma: no cover - belt-and-suspenders fallback
                console.print(
                    f"[yellow]MM page-level fallback for {img_path.name}: {type(e).__name__}: {e}. "
                    "Продолжаю без VLM-данных для страницы.[/yellow]"
                )

        pages.append(rec)

        if progress_callback is not None:
            progress_callback({
                "event": "page",
                "paper_id": paper_id,
                "pdf_path": str(pdf_path),
                "current": i + 1,
                "total": total_pages,
                "message": f"Страницы PDF: {i + 1}/{total_pages}",
            })

    # save
    (mm_root / "pages.jsonl").write_text(
        "\n".join([rec_to_jsonl(p) for p in pages]), encoding="utf-8"
    )

    return pages


def rec_to_jsonl(p: PageRecord) -> str:
    import json
    return json.dumps(
        {
            "paper_id": p.paper_id,
            "page": p.page,
            "text": p.text,
            "image_path": p.image_path,
            "vlm_caption": p.vlm_caption,
            "tables_md": p.tables_md,
            "equations_md": p.equations_md,
        },
        ensure_ascii=False,
    )
