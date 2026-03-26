# Live E2E smoke run

This directory contains the artifacts of a live smoke run executed on 2026-03-23.

## What was executed

Input: 3 synthetic multimodal scientific PDFs generated on the fly.

Pipeline:
1. PDF parsing with PyMuPDF fallback
2. Multimodal page extraction (text + rendered page image)
3. Text embeddings via built-in hash encoder
4. Multimodal embeddings via lightweight local hash backend
5. Vector indexing into persistent local Qdrant-compatible storage
6. Temporal triplet extraction with rule-based fallback when no external LLM is available
7. Assertion/event persistence into persistent local Memgraph-compatible state store
8. Temporal KG build and hypothesis generation
9. Local status service exposed at `http://127.0.0.1:8787/report`

## Key outputs

- `live_e2e_report.json` — compact summary of the run
- `temporal_kg.json` — built temporal graph
- `hypotheses.json` — generated hypothesis drafts
- `service_health.json` — health probe of the local status service
- `service_report.json` — counts and analytics snapshot exposed by the local status service

## Result snapshot

- processed papers: 3
- temporal KG nodes: 26
- temporal KG edges: 14
- hypotheses: 6
- Qdrant text points: 3
- Qdrant multimodal points: 6
- graph papers/chunks/assertions/events: 3 / 3 / 14 / 14
