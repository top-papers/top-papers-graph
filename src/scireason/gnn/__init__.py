"""Optional GNN utilities (PyTorch Geometric).

This package is intentionally optional:
- Base installation must work without PyG.
- When the user installs `.[gnn]`, the pipeline can leverage a GNN-based link prediction
  model to propose higher-quality hypothesis candidates.

The main entry point is:
    scireason.gnn.pyg_link_prediction.pyg_link_prediction
"""

from __future__ import annotations

from .pyg_link_prediction import pyg_available, pyg_link_prediction  # noqa: F401
