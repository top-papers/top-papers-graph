
## 2026-06-23 — DataSphere DPO attention backend fix

- Fixed DPO Qwen3-VL loading when `ATTN_IMPLEMENTATION=auto` is passed by DataSphere jobs.
- Mirrored SFT/GRPO attention backend resolution in `train_vlm_dpo.py`: `auto` now becomes `flash_attention_2` when `flash_attn` is installed, otherwise `sdpa`.
- Added a regression test to ensure DPO never forwards unsupported `attn_implementation="auto"` into Transformers model loading.

## 2026-06-19 — SciReason fine-tuning v2 pipeline

- Added export-only `build_scireason_alignment_datasets.py` for leakage-safe SFT/DPO/GRPO dataset preparation.
- Disabled dangerous `imagefolder` source mode by default unless `--allow-imagefolder-fallback` is explicitly provided.
- Added text-SFT -> VLM-SFT -> DPO -> optional-GRPO DataSphere pipeline and configs.
- Added `--init-adapter-path` to `train_vlm_sft.py` to continue multimodal SFT from the text-SFT adapter.
- Added v2 tests for leakage-safe splits, relevance-based image selection, and DPO row construction.

# Changelog

## 0.4.1 — Optional GNN mode (PyTorch Geometric)

### Added
- Optional **GNN link prediction** mode for hypothesis discovery via **PyTorch Geometric**:
  - new extra dependencies: `.[gnn]` (+ `.[gnn_ext]` for extension wheels)
  - new module: `scireason.gnn.pyg_link_prediction` (GraphSAGE + negative sampling)
  - env flags: `HYP_GNN_ENABLED`, `HYP_GNN_EPOCHS`, `HYP_GNN_HIDDEN_DIM`, `HYP_GNN_NODE_CAP`
  - docs: `docs/gnn.md` + install helper scripts in `scripts/`

### Fixed
- Base CLI import no longer requires optional Neo4j/Qdrant dependencies; optional stores now fail with a clear runtime error only when actually used.

## 0.2.0 — top-papers-graph rename + new data sources

### Changed
- Project renamed to **top-papers-graph** (distribution name + primary CLI command).
- CLI help/titles updated for new branding.

### Added
- New API connectors:
  - **PubMed** (NCBI E-utilities: ESearch + ESummary; optional EFetch abstracts)
  - **Europe PMC** (REST search; can query PubMed/PMC/preprints)
  - **bioRxiv/medRxiv** (details API: DOI + interval)
- New CLI support in `fetch` for: `crossref`, `pubmed`, `europepmc`, `biorxiv`, `medrxiv`.
- `.env.example` expanded with contact email / polite pool settings and NCBI configuration.
- `docs/sources.md` describing supported sources.

### Fixed
- Semantic Scholar connector now returns `data` (list of papers) by default; raw JSON is available via `search_papers_raw`.

### Packaging
- Added missing `__init__.py` files for subpackages so that setuptools package discovery works reliably.

## 2026-06-15 - DataSphere smoke chat-template and rate-limit fix

- Fixed Qwen3-VL/TRL SFT tokenization crash caused by raw non-dict content items inside multimodal chat messages.
- Added robust SFT/GRPO message-content canonicalization to emit only safe text/image blocks.
- Added smoke-only HF export asset subsetting to avoid downloading the full 15k-file asset tree and reduce Hugging Face 429 rate-limit retries.
- Added smoke dataset caps: MAX_SFT_SAMPLES=96 and MAX_GRPO_SAMPLES=48.
- Added regression tests for raw content normalization and smoke asset allow-pattern collection.

