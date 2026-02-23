# src/gencysynth/metrics/sanity/basic_stats.py
"""
Sanity metric: basic image statistics (global + optional per-class).

Why this exists
---------------
Before expensive metrics (FID/KID/utility), you want a quick health report:

- value range (min/max)
- mean/std
- percentiles (helps detect saturation)
- per-class sample counts (helps detect imbalance)
- NaN/Inf counts (fast failure signal)

This is intentionally lightweight and NumPy-only so it runs everywhere (HPC login,
CPU node, GPU node) and is stable across model families and datasets.

Rule A alignment
----------------
- This module returns a structured dict; it does not decide file locations.
- A separate metrics writer should store this under the run's artifacts folder
  (e.g., artifacts/<dataset>/<run_id>/metrics/sanity_basic_stats.json).

Typical usage
-------------
from gencysynth.metrics.sanity.basic_stats import basic_stats

report = basic_stats(
    x, y=y, num_classes=9,
    percentiles=(0,1,5,50,95,99,100),
    max_rows=10000,
)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[Any]]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _as_np(x: ArrayLike) -> np.ndarray:
    return np.asarray(x)


def _finite_counts(x: np.ndarray) -> Dict[str, int]:
    finite = np.isfinite(x)
    n_total = int(x.size)
    n_finite = int(finite.sum())
    return {
        "n_total": n_total,
        "n_finite": n_finite,
        "n_nonfinite": int(n_total - n_finite),
    }


def _maybe_subsample_rows(x: np.ndarray, max_rows: Optional[int], seed: int = 0) -> np.ndarray:
    if max_rows is None or x.ndim == 0:
        return x
    if x.shape[0] <= int(max_rows):
        return x
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(x.shape[0], size=int(max_rows), replace=False)
    return x[idx]


def _labels_to_int(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Normalize labels into integer ids:
      - one-hot (N,K) -> argmax
      - int (N,) -> as-is
    """
    y = _as_np(y)
    if y.ndim == 2 and y.shape[1] == int(num_classes):
        return np.argmax(y, axis=1).astype(np.int64)
    if y.ndim == 1:
        return y.astype(np.int64)
    raise ValueError(f"labels must be (N,) or (N,{num_classes})")


def _describe_array(
    x: np.ndarray,
    *,
    percentiles: Sequence[float] = (0, 1, 5, 50, 95, 99, 100),
) -> Dict[str, Any]:
    """
    Compute robust stats for numeric arrays.
    Works for images shaped (N,H,W,C) or vectors; flattens everything.
    """
    out: Dict[str, Any] = {}

    if x.size == 0:
        return {"empty": True, "n": 0}

    # Flatten for scalar stats
    xf = x.reshape(-1)

    # Finite counts
    fc = _finite_counts(xf)
    out.update(fc)

    # If there are non-finites, compute stats on finite subset
    if fc["n_finite"] == 0:
        out.update({"min": None, "max": None, "mean": None, "std": None, "percentiles": None})
        return out

    xff = xf[np.isfinite(xf)]

    out["min"] = float(np.min(xff))
    out["max"] = float(np.max(xff))
    out["mean"] = float(np.mean(xff))
    out["std"] = float(np.std(xff))

    # Percentiles
    try:
        ps = np.percentile(xff, list(percentiles)).astype(np.float64)
        out["percentiles"] = {str(p): float(v) for p, v in zip(percentiles, ps)}
    except Exception:
        out["percentiles"] = None

    return out


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def basic_stats(
    x: np.ndarray,
    *,
    y: Optional[np.ndarray] = None,
    num_classes: Optional[int] = None,
    expected_shape: Optional[Tuple[int, int, int]] = None,
    percentiles: Sequence[float] = (0, 1, 5, 50, 95, 99, 100),
    per_class: bool = True,
    max_rows: Optional[int] = 10000,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Compute basic global stats for an image tensor and (optionally) per-class stats.

    Parameters
    ----------
    x : np.ndarray
        Images. Expected (N,H,W,C) or (N,D). We do not reshape; only report.
    y : Optional[np.ndarray]
        Labels (int or one-hot). Required if per_class=True.
    num_classes : Optional[int]
        Needed to interpret one-hot labels and to produce stable per-class keys.
    expected_shape : Optional[(H,W,C)]
        If provided, included for reporting (not strict validation in this module).
    percentiles : sequence
        Percentiles to report.
    per_class : bool
        If True, compute per-class stats when y and num_classes are provided.
    max_rows : Optional[int]
        Subsample N to keep speed stable.
    seed : int
        Deterministic subsampling.

    Returns
    -------
    dict suitable for merging into an eval summary.
    """
    x = _as_np(x)
    xs = _maybe_subsample_rows(x, max_rows=max_rows, seed=seed)

    report: Dict[str, Any] = {
        "kind": "sanity.basic_stats",
        "observed": {
            "shape": tuple(x.shape) if hasattr(x, "shape") else None,
            "dtype": str(getattr(x, "dtype", "unknown")),
            "ndim": int(getattr(x, "ndim", -1)),
        },
        "expected": {
            "img_shape": expected_shape,
        },
        "global": _describe_array(xs, percentiles=percentiles),
        "per_class": None,
    }

    # Per-class block
    if per_class:
        if y is None or num_classes is None:
            report["per_class"] = {"skipped": True, "reason": "y or num_classes not provided"}
            return report

        y_ids = _labels_to_int(y, int(num_classes))
        # Keep alignment with subsampled x if we subsampled rows
        if xs.shape[0] != x.shape[0]:
            # We subsampled x with a deterministic idx in _maybe_subsample_rows.
            # Recompute the same idx to subsample y consistently.
            rng = np.random.default_rng(int(seed))
            idx = rng.choice(x.shape[0], size=int(max_rows), replace=False)
            y_ids_s = y_ids[idx]
        else:
            y_ids_s = y_ids

        per: Dict[str, Any] = {
            "counts": {str(k): 0 for k in range(int(num_classes))},
            "stats": {},
        }

        for k in range(int(num_classes)):
            mask = (y_ids_s == k)
            n_k = int(mask.sum())
            per["counts"][str(k)] = n_k
            if n_k == 0:
                per["stats"][str(k)] = {"empty": True, "n": 0}
                continue
            per["stats"][str(k)] = _describe_array(xs[mask], percentiles=percentiles)

        report["per_class"] = per

    return report


__all__ = ["basic_stats"]
