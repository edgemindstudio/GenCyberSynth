# src/gencysynth/metrics/diversity/__init__.py
"""
Diversity metrics for GenCyberSynth.

Rule A
------
- Metrics are pure computations: arrays + metadata -> MetricResult.
- No direct file I/O; artifact writing is handled by gencysynth.metrics.writer.
- Dataset_agnostic: supports multiple datasets via passed arrays and DatasetMeta.

Registered metrics
------------------
- diversity.duplicates : duplicate / near_duplicate rate via perceptual_ish hashing (fast, deterministic)
- diversity.coverage   : coverage proxy based on kNN distances in a lightweight feature space
"""

from __future__ import annotations

from ..registry import REGISTRY

from .duplicates import DuplicatesMetric
from .coverage import CoverageMetric


REGISTRY.register("diversity.duplicates", DuplicatesMetric())
REGISTRY.register("diversity.coverage", CoverageMetric())

__all__ = ["DuplicatesMetric",
           "CoverageMetric"]
