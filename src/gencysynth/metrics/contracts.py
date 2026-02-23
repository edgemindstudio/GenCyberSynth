# src/gencysynth/metrics/contracts.py
"""
Contracts and validation utilities for metrics inputs/outputs.

This module is intentionally strict about:
- channels-last image layout (N,H,W,C)
- dtype normalization expectations
- label shape and class bounds when labels are provided

Rule A
------
Validation errors should be explicit and actionable because metrics are often run
in large suites on HPC. A crisp error message saves hours.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ShapeSpec:
    img_shape: Tuple[int, int, int]
    num_classes: int


def _fail(msg: str) -> None:
    raise ValueError(f"[metrics.contracts] {msg}")


def validate_images(x: np.ndarray, spec: ShapeSpec, name: str) -> np.ndarray:
    """
    Validate images are shaped (N,H,W,C) and match expected H/W/C.
    Returns x unchanged (caller can normalize separately).
    """
    x = np.asarray(x)
    if x.ndim != 4:
        _fail(f"{name} must be rank-4 (N,H,W,C); got shape={x.shape}")
    H, W, C = spec.img_shape
    if tuple(x.shape[1:]) != (H, W, C):
        _fail(f"{name} expected shape (_, {H},{W},{C}); got {x.shape}")
    return x


def validate_labels(y: Optional[np.ndarray], spec: ShapeSpec, name: str, n: int) -> Optional[np.ndarray]:
    """
    Validate labels are either:
      - None
      - (N,) integer labels
      - (N,K) one-hot labels
    And that N matches x.shape[0].
    """
    if y is None:
        return None

    y = np.asarray(y)
    K = int(spec.num_classes)

    if y.ndim == 1:
        if y.shape[0] != n:
            _fail(f"{name} length mismatch: expected N={n}, got {y.shape[0]}")
        return y

    if y.ndim == 2:
        if y.shape[0] != n:
            _fail(f"{name} length mismatch: expected N={n}, got {y.shape[0]}")
        if y.shape[1] != K:
            _fail(f"{name} one-hot width mismatch: expected K={K}, got {y.shape[1]}")
        return y

    _fail(f"{name} must be None, (N,), or (N,K); got shape={y.shape}")
    return None