# src/gencysynth/metrics/distribution/__init__.py
"""
Distribution-level metrics for GenCyberSynth.

Rule A principles
-----------------
- Metric computation is pure (inputs -> result) and does not invent paths.
- Writing to artifacts is handled by gencysynth.metrics.writer via metrics.api.
- Configuration is taken from cfg["metrics"]["options"][<metric_name>] when present.
- Safe defaults for smoke tests and end-to-end sanity runs.

Registered metrics
------------------
- distribution.fid          : FID on lightweight features (pixel features by default)
- distribution.kid          : KID (polynomial-kernel MMD^2) on the same feature space
- distribution.mmd_rbf      : RBF-kernel MMD^2 on feature space
- distribution.js_kl_hist   : JS + KL divergences on pixel-intensity histograms
"""

from __future__ import annotations

from ..registry import REGISTRY

from .fid import FIDMetric
from .kid import KIDMetric
from .mmd import MMDMetric
from .js_kl import JSKLMetrics


# NOTE: registry keys should be stable and human-readable (Rule A)
REGISTRY.register("distribution.fid", FIDMetric())
REGISTRY.register("distribution.kid", KIDMetric())
REGISTRY.register("distribution.mmd_rbf", MMDMetric(kernel="rbf"))
REGISTRY.register("distribution.js_kl_hist", JSKLMetrics())

__all__ = ["FIDMetric", "KIDMetric", "MMDMetric", "JSKLMetrics"]
