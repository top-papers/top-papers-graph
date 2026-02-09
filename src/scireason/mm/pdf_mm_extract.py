from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import settings
from .vlm import describe_image, VLMResult


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

    for i in range(len(doc)):
        page = doc.load_page(i)
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
            res: VLMResult = describe_image(
                image_path=img_path,
                prompt=prompt_context or "Извлеки смысл страницы научной статьи (графики/таблицы/формулы)",
            )
            rec.vlm_caption = res.caption
            rec.tables_md = res.extracted_tables_md
            rec.equations_md = res.extracted_equations_md

        pages.append(rec)

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
