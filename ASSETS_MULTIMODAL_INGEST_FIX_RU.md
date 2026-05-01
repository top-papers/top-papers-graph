# Fix: assets contain PNGs only for two cases

## Problem

Notebook output showed that PDF downloads succeeded, but PDF ingestion into `downloaded_processed_papers` failed because the code path used the legacy GROBID-only ingester:

```text
failed to ingest ... GROBID недоступен по 'http://localhost:8070' (ConnectError: [Errno 111] Connection refused)
```

As a result, `download_ingested_processed_papers` stayed `0`, so the exporter had no generated page PNGs for most samples. Only the two Task 2 bundles that already contained local PNG evidence produced folders under `assets/`.

## Fix

### Repository

`src/scireason/scidatapipe_bridge/download.py`

- switched downloaded-PDF ingestion from `ingest_pdf_multimodal()` to `ingest_pdf_multimodal_auto()`;
- switched text-only ingestion from `ingest_pdf()` to `ingest_pdf_auto()`;
- this avoids the hard dependency on a running local GROBID service in Colab;
- the auto path falls back to local PyMuPDF/pypdf extraction and still renders PDF pages into:

```text
downloaded_processed_papers/<paper>/mm/images/page_XXX.png
```

`src/scireason/scidatapipe_bridge/hf_hub.py`

- added `delete_patterns` passthrough for clean Hub uploads when needed.

### Notebook

- installs `PyMuPDF`, `pypdf`, and `pillow`;
- removes hardcoded HF token;
- defaults `DOWNLOAD_RUN_VLM=False`, because page PNG extraction does not require VLM captions;
- sets `OCR_BACKEND=pymupdf` and `VLM_BACKEND=none` for a stable Colab run;
- validates that downloaded PDFs were actually ingested;
- validates image rows and `assets/**/*.png` before upload;
- uploads to Hugging Face with `delete_patterns`, so the remote `assets/` folder is cleaned and replaced rather than left stale.

## Expected result

After rerunning the notebook with the fixed repository:

- `download_ingested_processed_papers` should be greater than `0` when PDFs are downloaded;
- `sft_rows_with_images` / `grpo_rows_with_images` should increase;
- `assets/` should contain many more per-case folders, not only the two Task 2 bundles with pre-existing PNGs.
