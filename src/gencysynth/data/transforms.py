# src/gencysynth/data/transforms.py
"""
gencysynth.data.transforms
=========================

Pure, dependency-light transforms used by GenCyberSynth data loaders and evaluation.

Why this module exists
----------------------
We want a scalable, multi-dataset repo layout. That means:
- loaders should focus on "where to load from"
- transforms should focus on "how to normalize / reshape / encode labels"

Design goals
------------
- No heavy framework dependency (no TensorFlow required).
- Safe defaults:
    - never mutates caller arrays
    - robust handling of common image ranges: [0,255], [-1,1], [0,1]
- Clear, professional error messages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = [
    "one_hot",
    "onehot_to_int",
    "to_01_hwc",
    "split_val_from_test",
    "dataset_counts",
    "TransformConfig",
]


# -----------------------------------------------------------------------------
# Configuration container (optional but useful)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class TransformConfig:
    """
    Optional transform configuration.

    Attributes
    ----------
    img_shape:
        (H, W, C) expected shape used by to_01_hwc.
    num_classes:
        Number of classes for one_hot.
    """
    img_shape: Tuple[int, int, int]
    num_classes: int


# -----------------------------------------------------------------------------
# Label transforms
# -----------------------------------------------------------------------------
def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert class labels to one-hot (float32) without framework dependencies.

    Accepts:
      - shape (N,) integer class ids in [0, num_classes-1]
      - shape (N, K) that already looks one-hot (K must equal num_classes)

    Returns
    -------
    np.ndarray
        Shape (N, num_classes), dtype float32, values clamped to [0,1].
    """
    lab = np.asarray(labels)

    # Already one-hot?
    if lab.ndim == 2 and lab.shape[1] == int(num_classes):
        y = lab.astype(np.float32, copy=False)
        np.clip(y, 0.0, 1.0, out=y)
        return y

    if lab.ndim != 1:
        raise ValueError(f"Labels must be 1-D ints or 2-D one-hot; got shape {lab.shape}.")

    if lab.size == 0:
        return np.zeros((0, int(num_classes)), dtype=np.float32)

    lab_i = lab.astype(int, copy=False)

    if int(lab_i.min()) < 0 or int(lab_i.max()) >= int(num_classes):
        raise ValueError(
            f"Label ids out of range [0, {int(num_classes)-1}] "
            f"(min={int(lab_i.min())}, max={int(lab_i.max())})."
        )

    eye = np.eye(int(num_classes), dtype=np.float32)
    return eye[lab_i]


def onehot_to_int(y: np.ndarray) -> np.ndarray:
    """
    Convert labels to integer ids.

    Accepts:
      - (N,) ints
      - (N,K) one-hot / probabilities -> argmax(axis=1)

    Returns:
      (N,) int32
    """
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] > 1:
        return y.argmax(axis=1).astype(np.int32, copy=False)
    return y.astype(np.int32, copy=False)


# -----------------------------------------------------------------------------
# Image transforms
# -----------------------------------------------------------------------------
def to_01_hwc(x: np.ndarray, img_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Normalize images to float32 in [0,1] and reshape to (N, H, W, C).

    Accepted input formats:
      - (N, H, W)     -> expanded to (N, H, W, 1)
      - (N, H, W, C)  -> used as-is
      - (H, W)        -> treated as single image -> (1, H, W, 1)
      - (H, W, C)     -> treated as single image -> (1, H, W, C)

    Accepted input ranges:
      - [0,255] -> divide by 255
      - [-1,1]  -> (x + 1)/2
      - [0,1]   -> pass-through

    Parameters
    ----------
    x:
        Image batch (or single image).
    img_shape:
        Target (H, W, C). Used for final reshape sanity.

    Returns
    -------
    np.ndarray
        float32 array in [0,1], shape (N,H,W,C).
    """
    H, W, C = img_shape
    arr = np.asarray(x)

    # Handle single image shapes first
    if arr.ndim == 2:          # (H,W)
        arr = arr[None, ..., None]
    elif arr.ndim == 3:
        # could be (N,H,W) OR (H,W,C)
        if arr.shape[0] == H and arr.shape[1] == W:
            # treat as (H,W,C?) single image
            arr = arr[None, ...]
            if arr.shape[-1] != C and C == 1:
                arr = arr[..., None]  # (1,H,W,1)
        else:
            # treat as (N,H,W)
            arr = arr[..., None]

    arr = arr.astype(np.float32, copy=False)

    # Range normalization heuristics
    x_min = float(np.min(arr)) if arr.size else 0.0
    x_max = float(np.max(arr)) if arr.size else 0.0

    if x_max > 1.5:
        arr = arr / 255.0
    elif x_min < 0.0:
        arr = (arr + 1.0) / 2.0

    # Final reshape + clamp for safety
    arr = arr.reshape((-1, H, W, C))
    np.clip(arr, 0.0, 1.0, out=arr)
    return arr


# -----------------------------------------------------------------------------
# Split helpers
# -----------------------------------------------------------------------------
def split_val_from_test(
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    val_fraction: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split an existing "test" set into (val, test) by val_fraction.

    We do not shuffle here because many datasets already come shuffled.
    If you need shuffling, do it upstream (so it is reproducible and documented).

    Returns:
      x_val, y_val, x_test2, y_test2
    """
    n = int(len(x_test))
    n_val = int(n * float(val_fraction))
    n_val = max(0, min(n_val, n))
    return x_test[:n_val], y_test[:n_val], x_test[n_val:], y_test[n_val:]


# -----------------------------------------------------------------------------
# Counts block (Phase-1/Phase-2 summaries)
# -----------------------------------------------------------------------------
def dataset_counts(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    x_synth: Optional[np.ndarray] = None,
) -> Dict[str, int | None]:
    """
    Build the 'images' counts block used by summary JSON outputs.
    """
    return {
        "train_real": int(len(x_train)),
        "val_real": int(len(x_val)),
        "test_real": int(len(x_test)),
        "synthetic": (int(len(x_synth)) if isinstance(x_synth, np.ndarray) else None),
    }
