# src/gencysynth/metrics/calibration/__init__.py
"""
Calibration metrics for GenCyberSynth.

Calibration answers: "Do predicted probabilities reflect true correctness?"

Rule A
------
- Pure computations only: inputs -> MetricResult.
- No I/O and no artifact path assumptions.
- Dataset-agnostic: accepts integer or one-hot labels, any K classes.

Registered metrics
------------------
- calibration.ece   : Expected Calibration Error (ECE), lower is better
- calibration.brier : Brier score (multi-class), lower is better
"""

from __future__ import annotations

from ..registry import REGISTRY
from .ece import ECEMetric
from .brier import BrierMetric

REGISTRY.register("calibration.ece", ECEMetric())
REGISTRY.register("calibration.brier", BrierMetric())

__all__ = ["ECEMetric", "BrierMetric"]
