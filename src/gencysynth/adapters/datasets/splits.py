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
from typing import Any, Tuple
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


    @staticmethod
    def _to_x01_nhwc(x: np.ndarray) -> np.ndarray:
        """Ensure x is float32 NHWC in [0,1]."""
        x = np.asarray(x)

        # NHWC: allow (N,H,W) -> (N,H,W,1)
        if x.ndim == 3:
            x = x[..., None]
        if x.ndim != 4:
            raise ValueError(f"Expected x with ndim 3 or 4; got shape {x.shape}")

        x = x.astype(np.float32, copy=False)

        # If looks like 0..255, scale down
        try:
            if np.nanmax(x) > 1.5:
                x = x / 255.0
        except Exception:
            pass

        # Clip to [0,1] to be eval-friendly
        x = np.clip(x, 0.0, 1.0)
        return x

    @staticmethod
    def _to_y_int_onehot(y: np.ndarray, num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
        """Convert y into (y_int, y_onehot). Accepts int labels or one-hot."""
        y = np.asarray(y)

        K = int(num_classes)
        if K <= 1:
            raise ValueError(f"num_classes must be >= 2; got {K}")

        # One-hot input
        if y.ndim == 2 and y.shape[1] == K:
            y_onehot = y.astype(np.float32, copy=False)
            y_int = np.argmax(y_onehot, axis=1).astype(np.int64, copy=False)
            return y_int, y_onehot

        # Integer input
        if y.ndim == 1:
            y_int = y.astype(np.int64, copy=False)
            y_onehot = np.zeros((y_int.shape[0], K), dtype=np.float32)
            y_onehot[np.arange(y_int.shape[0]), y_int.astype(np.int64)] = 1.0
            return y_int, y_onehot

        # Some datasets might provide (N,1)
        if y.ndim == 2 and y.shape[1] == 1:
            y_int = y.reshape(-1).astype(np.int64, copy=False)
            y_onehot = np.zeros((y_int.shape[0], K), dtype=np.float32)
            y_onehot[np.arange(y_int.shape[0]), y_int.astype(np.int64)] = 1.0
            return y_int, y_onehot

        raise ValueError(f"Expected y as (N,) ints or (N,{K}) one-hot; got shape {y.shape}")

    @classmethod
    def from_dataset_arrays(cls, arrays: Any, cfg: Dict[str, Any]) -> "DatasetSplits":
        """
        Convert a dataset loader output (DatasetArrays-like) into Rule-A DatasetSplits.

        Expects `arrays` to expose attributes:
          x_train, y_train, x_val, y_val, x_test, y_test  (val/test may be None)

        num_classes resolution:
          1) cfg['dataset']['num_classes']
          2) cfg['dataset']['channels'] (NOT num_classes) -> ignored
          3) infer from y_train if possible (fallback)
        """
        # 1) num_classes
        dcfg = cfg.get("dataset", {}) if isinstance(cfg.get("dataset"), dict) else {}
        K = dcfg.get("num_classes", None)

        if K is None:
            # best-effort infer from train labels
            ytr = getattr(arrays, "y_train", None)
            if ytr is None:
                raise ValueError("Cannot infer num_classes: cfg.dataset.num_classes missing and arrays.y_train missing.")
            ytr = np.asarray(ytr)
            if ytr.ndim == 2:
                K = int(ytr.shape[1])
            elif ytr.ndim in (1, 2):
                K = int(np.max(ytr)) + 1
            else:
                raise ValueError("Cannot infer num_classes from y_train.")
        K = int(K)

        def _mk_split(x: Any, y: Any) -> SplitArrays:
            x01 = cls._to_x01_nhwc(np.asarray(x))
            y_int, y_onehot = cls._to_y_int_onehot(np.asarray(y), num_classes=K)
            return SplitArrays(x01=x01, y_int=y_int, y_onehot=y_onehot)

        # 2) required train
        x_train = getattr(arrays, "x_train", None)
        y_train = getattr(arrays, "y_train", None)
        if x_train is None or y_train is None:
            raise ValueError("DatasetArrays must provide x_train and y_train.")
        train = _mk_split(x_train, y_train)

        # 3) optional val/test
        val = None
        if getattr(arrays, "x_val", None) is not None and getattr(arrays, "y_val", None) is not None:
            val = _mk_split(getattr(arrays, "x_val"), getattr(arrays, "y_val"))

        test = None
        if getattr(arrays, "x_test", None) is not None and getattr(arrays, "y_test", None) is not None:
            test = _mk_split(getattr(arrays, "x_test"), getattr(arrays, "y_test"))

        return cls(train=train, val=val, test=test)
