# src/gencysynth/adapters/normalize.py
"""
Normalization helpers shared across adapters (Rule A friendly).

Goal
----
Adapters should standardize inputs/outputs at the boundary so model code can
focus on modeling, not edge cases:
- label formats (int vs one_hot)
- image shapes (flattened vs (H,W,C))
- value ranges ([0,1] vs [-1,1])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# =============================================================================
# Labels
# =============================================================================
def labels_to_int(y: np.ndarray, *, num_classes: int) -> np.ndarray:
    """
    Convert labels to integer class ids.

    Accepts:
      - (N,) integer labels
      - (N,K) one_hot labels
    """
    y = np.asarray(y)
    if y.ndim == 1:
        return y.astype(np.int64, copy=False)
    if y.ndim == 2 and y.shape[1] == int(num_classes):
        return np.argmax(y, axis=1).astype(np.int64)
    raise ValueError(f"Expected labels (N,) ints or (N,{num_classes}) one_hot; got shape {y.shape}.")


def labels_to_onehot(y: np.ndarray, *, num_classes: int, dtype=np.float32) -> np.ndarray:
    """
    Convert labels to one_hot.

    Accepts:
      - (N,) integer labels
      - (N,K) one_hot labels (returned as_is, dtype cast)
    """
    y = np.asarray(y)
    K = int(num_classes)
    if y.ndim == 2 and y.shape[1] == K:
        return y.astype(dtype, copy=False)

    ids = labels_to_int(y, num_classes=K)
    out = np.zeros((len(ids), K), dtype=dtype)
    out[np.arange(len(ids)), ids] = 1.0
    return out
    

def ensure_onehot(y: np.ndarray, *, num_classes: int, dtype=np.float32) -> np.ndarray:
    """
    Ensure labels are one_hot (N,K). Accepts (N,) ints or (N,K) one_hot.
    Thin wrapper for adapter code readability.
    """
    return labels_to_onehot(y, num_classes=num_classes, dtype=dtype)


def ensure_int_labels(y: np.ndarray, *, num_classes: int) -> np.ndarray:
    """
    Ensure labels are integer ids (N,). Accepts (N,) ints or (N,K) one_hot.
    Thin wrapper for adapter code readability.
    """
    return labels_to_int(y, num_classes=num_classes)


def ensure_onehot(y: np.ndarray, *, num_classes: int, dtype=np.float32) -> np.ndarray:
    """
    Ensure labels are one-hot (N,K). Accepts (N,) ints or (N,K) one-hot.
    Thin wrapper for adapter code readability.
    """
    return labels_to_onehot(y, num_classes=num_classes, dtype=dtype)


def ensure_int_labels(y: np.ndarray, *, num_classes: int) -> np.ndarray:
    """
    Ensure labels are integer ids (N,). Accepts (N,) ints or (N,K) one-hot.
    Thin wrapper for adapter code readability.
    """
    return labels_to_int(y, num_classes=num_classes)


# =============================================================================
# Shapes
# =============================================================================
def ensure_nhwc(x: np.ndarray, img_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Ensure x is shaped (N,H,W,C).

    Accepts:
      - (N,H,W,C)
      - (N, H*W*C) flattened
    """
    x = np.asarray(x)
    H, W, C = map(int, img_shape)
    if x.ndim == 4:
        if tuple(x.shape[1:]) != (H, W, C):
            raise ValueError(f"Expected images shape (N,{H},{W},{C}), got {x.shape}.")
        return x
    if x.ndim == 2:
        if x.shape[1] != H * W * C:
            raise ValueError(f"Expected flattened dim {H*W*C}, got {x.shape}.")
        return x.reshape((-1, H, W, C))
    raise ValueError(f"Expected x as (N,H,W,C) or (N,D); got {x.shape}.")


def flatten_nhwc(x: np.ndarray, img_shape: Tuple[int, int, int]) -> np.ndarray:
    """Flatten (N,H,W,C) -> (N, H*W*C) with validation."""
    x = ensure_nhwc(x, img_shape)
    H, W, C = map(int, img_shape)
    return x.reshape((-1, H * W * C))


def ensure_float32(x: np.ndarray) -> np.ndarray:
    """Ensure array is float32 (no range scaling)."""
    return np.asarray(x).astype(np.float32, copy=False)

# =============================================================================
# Ranges
# =============================================================================
def to_float01(x: np.ndarray) -> np.ndarray:
    """
    Convert to float32 and map to [0,1] if inputs look like [0,255].
    """
    x = np.asarray(x).astype(np.float32, copy=False)
    if np.nanmax(x) > 1.5:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


def to_minus1_1(x01: np.ndarray) -> np.ndarray:
    """Map [0,1] -> [-1,1]."""
    x01 = to_float01(x01)
    return (x01 * 2.0 - 1.0).astype(np.float32, copy=False)


def to_01(x: np.ndarray) -> np.ndarray:
    """
    Map either [-1,1] or [0,1] (or [0,255]) into [0,1].
    """
    x = np.asarray(x).astype(np.float32, copy=False)
    mx = np.nanmax(x)
    mn = np.nanmin(x)

    if mx > 1.5:
        x = x / 255.0
        return np.clip(x, 0.0, 1.0)

    # Heuristic: if values look like [-1,1], rescale.
    if mn < -0.2:
        return np.clip((x + 1.0) / 2.0, 0.0, 1.0)

    return np.clip(x, 0.0, 1.0)
    
    
def from_minus1_1(x: np.ndarray) -> np.ndarray:
    """Alias: map [-1,1] (or [0,255]) to [0,1]."""
    return to_01(x)


def from_minus1_1(x: np.ndarray) -> np.ndarray:
    """Alias: map [-1,1] (or [0,255]) to [0,1]."""
    return to_01(x)


# =============================================================================
# Small convenience struct (optional)
# =============================================================================
@dataclass(frozen=True)
class NormalizedBatch:
    """
    A normalized batch suitable for most models/evaluators.

    - x01: images in [0,1] shaped (N,H,W,C)
    - y_int: integer labels (N,)
    - y_onehot: one_hot labels (N,K)
    """
    x01: np.ndarray
    y_int: np.ndarray
    y_onehot: np.ndarray
