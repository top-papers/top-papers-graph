# Fix: ModuleNotFoundError: scireason.ingest.vlm_ocr

## Problem

Task 2 imports `scireason.task2_validation`, which imports `scireason.pipeline.task2_validation`, which imports `scireason.temporal.temporal_kg_builder`.
`temporal_kg_builder.py` expected `scireason.ingest.vlm_ocr.extract_pdf_page_chunks_vlm_ocr`, but the repository snapshot did not contain `src/scireason/ingest/vlm_ocr.py`.

## Fix

Added `src/scireason/ingest/vlm_ocr.py` with the expected API:

- `VLMOCRPageChunk`
- `extract_pdf_page_chunks_vlm_ocr(...)`

The implementation is safe for notebook use: if a VLM backend is unavailable, it falls back to PyMuPDF page text and does not block Task 2 imports.

The Task 2 notebooks were also updated so the installation/import cell can create this compatibility module automatically when an older repository snapshot is used.

## Quick check

From the repository root:

```bash
PYTHONPATH=src python -c "import scireason.task2_validation; from scireason.ingest.vlm_ocr import extract_pdf_page_chunks_vlm_ocr; print('ok')"
```
