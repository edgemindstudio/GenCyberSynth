# src/gencysynth/metrics/features.py
"""
Feature extraction used by some metrics.

These are *not* deep learned features (no Inception here). We keep it light:
- global pixel mean/var
- per-class pixel mean (when labels exist)
- histogram summaries

This helps establish:
- "basic stats sanity"
- distribution drift checks between REAL and SYNTH
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def global_stats(x01: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x01, dtype=np.float32)
    if x.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def per_class_mean(
    x01: np.ndarray,
    y_int: np.ndarray,
    num_classes: int,
) -> Dict[str, Dict[str, float]]:
    """
    For each class k: mean/std over all pixels in all images of that class.
    """
    out: Dict[str, Dict[str, float]] = {}
    for k in range(int(num_classes)):
        idx = (y_int == k)
        if int(np.sum(idx)) == 0:
            out[str(k)] = {"mean": float("nan"), "std": float("nan"), "n": 0}
            continue
        xk = x01[idx]
        out[str(k)] = {
            "mean": float(np.mean(xk)),
            "std": float(np.std(xk)),
            "n": int(xk.shape[0]),
        }
    return out


def pixel_histogram(
    x01: np.ndarray,
    bins: int = 32,
    range_: Tuple[float, float] = (0.0, 1.0),
) -> Dict[str, object]:
    """
    Histogram over pixel intensities (flattened).
    Returns JSON-friendly {bins, edges, counts}.
    """
    x = np.asarray(x01, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return {"bins": int(bins), "edges": [], "counts": []}
    counts, edges = np.histogram(x, bins=int(bins), range=range_)
    return {
        "bins": int(bins),
        "edges": edges.astype(np.float64).tolist(),
        "counts": counts.astype(np.int64).tolist(),
    }
