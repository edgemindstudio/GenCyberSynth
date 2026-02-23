# src/gencysynth/metrics/preprocess.py
"""
Preprocessing for metrics.

Metrics should consume a standardized representation:
- images in float32 [0,1]
- channels-last (N,H,W,C)
- labels either int (N,) or None

This file keeps transformations small and explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .contracts import ShapeSpec, validate_images, validate_labels
from .types import as_float01, as_int_labels


@dataclass(frozen=True)
class PreprocessConfig:
    """
    Common preprocessing knobs for metrics.
    """
    binarize: bool = False
    bin_threshold: float = 0.5


def preprocess_for_metrics(
    *,
    x: np.ndarray,
    y: Optional[np.ndarray],
    spec: ShapeSpec,
    pp: PreprocessConfig,
    name: str,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Validate + normalize images/labels for metrics.

    Returns
    -------
    x01: float32 in [0,1] (optionally binarized)
    y_i: int64 labels (N,) or None
    """
    x = validate_images(x, spec, name=name)
    y = validate_labels(y, spec, name=f"{name}.labels", n=int(x.shape[0]))

    x01 = as_float01(x)
    if pp.binarize:
        thr = float(pp.bin_threshold)
        x01 = (x01 >= thr).astype(np.float32)

    y_i = as_int_labels(y, num_classes=spec.num_classes)
    return x01, y_i
