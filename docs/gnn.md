# Optional GNN mode (PyTorch Geometric)

This repository supports an **optional “adult” GNN mode** for hypothesis discovery.

When enabled, the pipeline will run a small **link prediction** model (GraphSAGE + negative sampling)
on the term graph to propose **missing edges** that can be turned into *testable hypotheses*.

## Why this mode exists

Classic graph heuristics (common neighbors / Adamic–Adar / Jaccard) are strong and cheap baselines,
but they often struggle on sparse graphs and in the presence of long-range dependencies.

A GNN can learn node representations by aggregating multi-hop neighborhoods and often produces
better candidate links for “bridge” hypotheses.

## Install

Minimal install (works in most environments):

```bash
pip install -e ".[gnn]"
```

For **best performance** (recommended), also install the PyG extension wheels (`pyg-lib`, `torch-scatter`, ...).

Convenience scripts:

```bash
./scripts/install_pyg_extensions.sh      # Linux/macOS
# Windows PowerShell:
#   .\scripts\install_pyg_extensions.ps1
```

```bash
# 1) check torch and cuda
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"

# 2) install extensions via the official wheel index
# Example for CPU-only PyTorch 2.5.*:
#   pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
#     -f https://data.pyg.org/whl/torch-2.5.0+cpu.html
```

See the official PyG installation notes for the exact `TORCH` and `CUDA` values.

## Enable in the pipeline

Set the env var:

```bash
HYP_GNN_ENABLED=1
```

Optional knobs:

```bash
HYP_GNN_EPOCHS=80
HYP_GNN_HIDDEN_DIM=64
HYP_GNN_LR=0.01
HYP_GNN_NODE_CAP=300
```

## What happens when enabled

During `generate_candidates(...)`:

1. Build a NetworkX term graph from the temporal KG.
2. Take an induced subgraph of top-degree nodes (`HYP_GNN_NODE_CAP`) for speed.
3. Train a small GraphSAGE encoder with a dot-product decoder.
4. Produce top-k predicted non-edges with high probability.
5. Convert them into hypothesis candidates (`kind = gnn_missing_link`).

If PyG is not installed, the pipeline prints a warning and falls back to heuristic methods.
