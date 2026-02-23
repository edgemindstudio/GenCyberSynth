# src/gencysynth/adapters/datasets/splits.py
"""
Split contract for dataset adapters.

Rule A goal
-----------
Downstream code should always see the same structure:
  - train / val / test splits
  - x in NHWC
  - y as both int labels and one-hot labels
  - x in [0,1] by default (eval-friendly)

If a model needs [-1,1] (tanh decoder), adapters can convert at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import numpy as np


class Split(str, Enum):
    train = "train"
    val = "val"
    test = "test"


@dataclass(frozen=True)
class SplitArrays:
    """
    A single split worth of arrays.

    Conventions
    -----------
    - x01: float32 images in [0,1], shape (N,H,W,C)
    - y_int: int64 labels, shape (N,)
    - y_onehot: float32 one-hot labels, shape (N,K)
    """
    x01: np.ndarray
    y_int: np.ndarray
    y_onehot: np.ndarray

    def n(self) -> int:
        return int(self.x01.shape[0])


@dataclass(frozen=True)
class DatasetSplits:
    """
    Standardized dataset splits.

    - train is required
    - val/test are optional depending on dataset config
    """
    train: SplitArrays
    val: Optional[SplitArrays] = None
    test: Optional[SplitArrays] = None

    def as_dict(self) -> Dict[str, SplitArrays]:
        out: Dict[str, SplitArrays] = {"train": self.train}
        if self.val is not None:
            out["val"] = self.val
        if self.test is not None:
            out["test"] = self.test
        return out
