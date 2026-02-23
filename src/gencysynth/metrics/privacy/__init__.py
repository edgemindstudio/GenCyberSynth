# src/gencysynth/metrics/privacy/__init__.py
"""
Privacy-oriented metrics for GenCyberSynth.

Rule A
------
- Metrics here must be *pure computation*:
  - No reading from disk (no artifact paths).
  - No writing to disk.
- Any heavy features (e.g., embeddings) must be computed elsewhere
  (feature extractors / utility classifiers) and injected into the metric call payload.

Currently registered
--------------------
- privacy.nn_distance : nearest-neighbor distance from synthetic -> real in feature space.
"""

from __future__ import annotations

from ..registry import REGISTRY
from .nn_distance import NNDistMetric

REGISTRY.register("privacy.nn_distance", NNDistMetric())

__all__ = ["NNDistMetric"]