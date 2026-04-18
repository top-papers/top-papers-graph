"""Task 1 trajectory normalizer (legacy YAMLs -> v4 schema)."""
from scireason.scidatapipe_bridge.vendor.normalize_task1.normalizer import (
    main,
    normalize_file,
)

__all__ = ["main", "normalize_file"]
