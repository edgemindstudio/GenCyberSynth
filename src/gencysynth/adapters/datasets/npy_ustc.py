# src/gencysynth/adapters/datasets/npy_ustc.py
"""
USTC_TFC2016 NPY dataset adapter.

This bridges the legacy / existing loader(s) in gencysynth.data.* into the
standard DatasetSplits contract.

Expected layout (classic)
-------------------------
<data_root>/
  train_data.npy
  train_labels.npy
  test_data.npy
  test_labels.npy

We optionally split the provided test split into (val, test) using a fraction.

Config keys (supported)
-----------------------
data.root or DATA_DIR:
  Root directory containing the .npy files.

IMG_SHAPE:
  e.g. (40,40,1)

NUM_CLASSES:
  e.g. 9

VAL_FRACTION:
  Fraction of "test" to allocate to validation (default 0.5).
  - val = first portion, test = remaining (deterministic unless shuffle is enabled)

SHUFFLE_TEST_BEFORE_SPLIT:
  Whether to shuffle the provided test split before splitting val/test (default False).
  Determinism: controlled by SEED.

SEED:
  Seed used if shuffling is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from .base import DatasetAdapter, DatasetSpec
from .splits import DatasetSplits, SplitArrays
from gencysynth.adapters.normalize import ensure_nhwc, labels_to_int, labels_to_onehot, to_float01


def _cfg_get(cfg: Dict[str, Any], dotted: str, default=None):
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


@dataclass
class UstcNpyDatasetAdapter:
    spec: DatasetSpec = DatasetSpec(
        dataset_id="ustc_tfc2016_npy",
        description="USTC_TFC2016 malware images stored as train/test .npy arrays",
        default_img_shape=(40, 40, 1),
        default_num_classes=9,
    )

    def load(self, cfg: Dict[str, Any]) -> DatasetSplits:
        # Resolve dataset root
        data_root = Path(cfg.get("DATA_DIR", _cfg_get(cfg, "data.root", "data")))
        img_shape: Tuple[int, int, int] = tuple(cfg.get("IMG_SHAPE", self.spec.default_img_shape))
        num_classes = int(cfg.get("NUM_CLASSES", self.spec.default_num_classes))

        # Load npy arrays
        x_train = np.load(data_root / "train_data.npy")
        y_train = np.load(data_root / "train_labels.npy")
        x_test = np.load(data_root / "test_data.npy")
        y_test = np.load(data_root / "test_labels.npy")

        # Normalize shapes and ranges
        x_train01 = to_float01(ensure_nhwc(x_train, img_shape))
        x_test01 = to_float01(ensure_nhwc(x_test, img_shape))

        y_train_int = labels_to_int(y_train, num_classes=num_classes)
        y_test_int = labels_to_int(y_test, num_classes=num_classes)

        y_train_oh = labels_to_onehot(y_train_int, num_classes=num_classes)
        y_test_oh = labels_to_onehot(y_test_int, num_classes=num_classes)

        # Optionally split test into (val, test)
        val_fraction = float(cfg.get("VAL_FRACTION", cfg.get("val_fraction", 0.5)))
        shuffle = bool(cfg.get("SHUFFLE_TEST_BEFORE_SPLIT", False))
        seed = int(cfg.get("SEED", 42))

        if val_fraction <= 0.0 or x_test01.shape[0] < 2:
            return DatasetSplits(
                train=SplitArrays(x01=x_train01, y_int=y_train_int, y_onehot=y_train_oh),
                val=None,
                test=SplitArrays(x01=x_test01, y_int=y_test_int, y_onehot=y_test_oh),
            )

        n = int(x_test01.shape[0])
        n_val = max(1, int(round(n * val_fraction)))
        n_val = min(n_val, n - 1)  # keep at least 1 for test

        idx = np.arange(n)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(idx)

        val_idx = idx[:n_val]
        test_idx = idx[n_val:]

        val = SplitArrays(
            x01=x_test01[val_idx],
            y_int=y_test_int[val_idx],
            y_onehot=y_test_oh[val_idx],
        )
        test = SplitArrays(
            x01=x_test01[test_idx],
            y_int=y_test_int[test_idx],
            y_onehot=y_test_oh[test_idx],
        )

        return DatasetSplits(
            train=SplitArrays(x01=x_train01, y_int=y_train_int, y_onehot=y_train_oh),
            val=val,
            test=test,
        )


# Register on import (optional but convenient)
from .registry import register_dataset_adapter
register_dataset_adapter("ustc_tfc2016_npy", lambda: UstcNpyDatasetAdapter())

__all__ = ["UstcNpyDatasetAdapter"]
