# Validation report

Date: 2026-03-06

## What was changed

- Added `TemporalEvent` schema and event-layer support for temporal KG.
- Extended Neo4j temporal store with:
  - `Event` nodes
  - optional vector index creation for `Chunk` and `Assertion`
  - event stream export
- Updated TG/MMKG build path to:
  - persist chunk embeddings to Neo4j
  - persist assertion embeddings to Neo4j
  - create Event nodes for each extracted temporal triplet
- Added lightweight TGNN/TGN-style temporal link prediction modules:
  - `src/scireason/tgnn/event_dataset.py`
  - `src/scireason/tgnn/tgn_link_prediction.py`
  - `src/scireason/tgnn/evaluation.py`
- Made TGNN/TGN-style prediction the preferred temporal candidate generator.
- Kept static PyG GraphSAGE as optional baseline.
- Added CLI commands:
  - `export-temporal-events`
  - `train-tgn`
- Updated architecture and GNN docs.
- Added tests for temporal events and TGNN generation.
- Added graceful fallback when `tenacity` is absent in this environment.

## Commands executed

### 1) Static validation

```bash
python -m compileall -q src tests
```

Result: success

### 2) Test suite

```bash
pytest -q
```

Result: `9 passed, 4 skipped`

### 3) Offline pipeline smoke run

```bash
PYTHONPATH=src python -m scireason.cli demo-run \
  --query 'temporal graph methods for science' \
  --out-dir /mnt/data/top-papers-graph-work/demo_runs
```

Result: success

Artifacts written under `results/validation_demo/`.

### 4) TGNN/TGN-style temporal prediction run

```bash
PYTHONPATH=src python -m scireason.cli train-tgn \
  --temporal-kg-json /mnt/data/top-papers-graph-work/demo_runs/demo_20260306_173033/temporal_kg.json \
  --out /mnt/data/top-papers-graph-work/demo_runs/demo_20260306_173033/tgn_predictions.json
```

Result: success

## Notes

- Offline/demo pipeline and TGNN command were exercised successfully.
- Live Neo4j vector index creation and `export-temporal-events` were implemented but not executed here,
  because no running Neo4j service was available inside this container session.
