"""Temporal GNN / temporal link prediction helpers.

The implementation is deliberately lightweight and course-friendly:
- works from the temporal KG event stream already produced by the repo
- does not require PyTorch Geometric Temporal for the base path
- exposes a TGNN-style predictor that prefers recency-aware event streams over static graphs
"""

from .event_dataset import build_event_stream, chronological_split  # noqa: F401
from .tgn_link_prediction import TGNLinkPredConfig, tgn_link_prediction, tgnn_available  # noqa: F401

from .pygt_temporal_link_prediction import (
    PyGTemporalLinkPredConfig,
    PyGTemporalUnavailableError,
    pygt_temporal_available,
    pygt_temporal_link_prediction,
)  # noqa: F401

from .prediction_types import LinkPredictionRecord, SemanticEdgeKey, predicate_family, same_predicate_family  # noqa: F401
