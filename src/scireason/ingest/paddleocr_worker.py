from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .paddleocr_pipeline import (
    _fallback_pdf_records,
    _load_pipeline,
    _records_from_predict_result,
)


def _record_to_dict(record):
    if hasattr(record, "model_dump"):
        return record.model_dump()
    return record.dict()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lang", default=None)
    args = parser.parse_args(argv)

    pdf_path = Path(args.pdf)
    pipeline, backend_name = _load_pipeline(lang=args.lang)
    try:
        output = pipeline.predict(input=str(pdf_path))
    except TypeError:
        output = pipeline.predict(str(pdf_path))

    records = []
    for page_index, result in enumerate(output):
        records.extend(
            _records_from_predict_result(
                result,
                paper_id=args.paper_id,
                page_index=page_index,
                source_backend=backend_name,
            )
        )

    if not records:
        records = _fallback_pdf_records(pdf_path=pdf_path, paper_id=args.paper_id)

    Path(args.out).write_text(
        json.dumps([_record_to_dict(r) for r in records], ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
