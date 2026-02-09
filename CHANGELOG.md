# Changelog

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
