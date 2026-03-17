# src/gencysynth/metrics/registry.py
"""
Metric registry + default built_in metrics.

Why a registry?
---------------
- Configure metrics from YAML: cfg.metrics.enabled = [...]
- Let new metric modules register plugins without touching orchestrator
- Keep naming stable across runs (Rule A artifacts)

Default included metrics
------------------------
- sanity.shape_checks
- sanity.basic_stats
- distribution.pixel_hist_l1

These are intentionally lightweight for smoke tests and quick end_to_end validation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .contracts import ShapeSpec, validate_images, validate_labels
from .features import global_stats, pixel_histogram, per_class_mean
from .types import DatasetMeta, Metric, MetricResult, RunMeta


class MetricRegistry:
    def __init__(self) -> None:
        self._metrics: Dict[str, Metric] = {}

    def register(self, name: str, metric: Metric) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Metric name must be a non_empty string.")
        if name in self._metrics:
            raise KeyError(f"Metric '{name}' is already registered.")
        self._metrics[name] = metric

    def get(self, name: str) -> Metric:
        if name not in self._metrics:
            known = ", ".join(sorted(self._metrics.keys()))
            raise KeyError(f"Unknown metric '{name}'. Known: [{known}]")
        return self._metrics[name]

    def list(self) -> List[str]:
        return sorted(self._metrics.keys())


REGISTRY = MetricRegistry()

# ---------------------------------------------------------------------
# Default metrics (small, stable, smoke_test friendly)
# ---------------------------------------------------------------------
class ShapeChecksMetric:
    """
    Verify shapes and label compatibility; no heavy computation.
    """
    def __call__(
        self,
        *,
        x_real01: np.ndarray,
        y_real: Optional[np.ndarray],
        x_synth01: np.ndarray,
        y_synth: Optional[np.ndarray],
        dataset: DatasetMeta,
        run: RunMeta,
        cfg: Dict,
    ) -> MetricResult:
        try:
            spec = ShapeSpec(img_shape=dataset.img_shape, num_classes=dataset.num_classes)
            validate_images(x_real01, spec, name="real")
            validate_images(x_synth01, spec, name="synth")
            validate_labels(y_real, spec, name="real.labels", n=int(x_real01.shape[0]))
            validate_labels(y_synth, spec, name="synth.labels", n=int(x_synth01.shape[0]))

            return MetricResult(
                name="sanity.shape_checks",
                value=1.0,
                details={
                    "real_shape": list(x_real01.shape),
                    "synth_shape": list(x_synth01.shape),
                    "labels_present": {"real": y_real is not None, "synth": y_synth is not None},
                },
                status="ok",
            )
        except Exception as e:
            return MetricResult(name="sanity.shape_checks", status="error", error=str(e))


class BasicStatsMetric:
    """
    Basic pixel distribution statistics (mean/std/min/max) and optional per_class mean/std.
    """
    def __call__(
        self,
        *,
        x_real01: np.ndarray,
        y_real: Optional[np.ndarray],
        x_synth01: np.ndarray,
        y_synth: Optional[np.ndarray],
        dataset: DatasetMeta,
        run: RunMeta,
        cfg: Dict,
    ) -> MetricResult:
        K = int(dataset.num_classes)
        details = {
            "real": global_stats(x_real01),
            "synth": global_stats(x_synth01),
        }

        # If labels exist, include per_class mean/std summaries
        if y_real is not None and y_synth is not None:
            details["real_per_class"] = per_class_mean(x_real01, y_real, num_classes=K)
            details["synth_per_class"] = per_class_mean(x_synth01, y_synth, num_classes=K)

        return MetricResult(name="sanity.basic_stats", value=None, details=details, status="ok")


class PixelHistL1Metric:
    """
    Histogram L1 distance between REAL and SYNTH pixel intensity distributions.

    This is a simple distribution drift measure:
      L1 = sum(|p_real - p_synth|) where p are normalized histogram densities.
    """
    def __call__(
        self,
        *,
        x_real01: np.ndarray,
        y_real: Optional[np.ndarray],
        x_synth01: np.ndarray,
        y_synth: Optional[np.ndarray],
        dataset: DatasetMeta,
        run: RunMeta,
        cfg: Dict,
    ) -> MetricResult:
        opts = ((cfg.get("metrics") or {}).get("options") or {}).get("distribution.pixel_hist_l1", {}) or {}
        bins = int(opts.get("bins", 32))

        h_real = pixel_histogram(x_real01, bins=bins)
        h_syn = pixel_histogram(x_synth01, bins=bins)

        # Convert counts to probability distribution
        cr = np.asarray(h_real["counts"], dtype=np.float64)
        cs = np.asarray(h_syn["counts"], dtype=np.float64)

        pr = cr / max(1.0, float(np.sum(cr)))
        ps = cs / max(1.0, float(np.sum(cs)))

        l1 = float(np.sum(np.abs(pr - ps)))

        return MetricResult(
            name="distribution.pixel_hist_l1",
            value=l1,
            details={
                "bins": bins,
                "hist_real": h_real,
                "hist_synth": h_syn,
                "l1": l1,
            },
            status="ok",
        )


# Register defaults
REGISTRY.register("sanity.shape_checks", ShapeChecksMetric())
REGISTRY.register("sanity.basic_stats", BasicStatsMetric())
REGISTRY.register("distribution.pixel_hist_l1", PixelHistL1Metric())