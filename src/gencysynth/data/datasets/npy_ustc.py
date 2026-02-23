# src/gencysynth/data/datasets/npy_ustc.py
"""
GenCyberSynth — NPY USTC-TFC2016 Dataset

This dataset matches the classic GenCyberSynth format where splits are stored
as numpy files:

  train_data.npy, train_labels.npy
  test_data.npy,  test_labels.npy

We further split provided test into (val, test) using val_fraction.

Dataset scalability
-------------------
- Raw dataset location is config-driven.
- Cache + fingerprints live under artifacts/datasets/<dataset_id>/.

Config contract
---------------
dataset:
  id: "USTC-TFC2016_40x40_gray"
  type: "npy_ustc"
  raw_root: "data/ustc"          # directory containing the *.npy quartet
  image_shape: [40,40,1]
  num_classes: 9
  val_fraction: 0.5
  one_hot: true                   # recommended for evaluation utilities
  cache:
    enabled: true
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from gencysynth.data.datasets.base import BaseDataset, DatasetArrays, DatasetInfo

try:
    from gencysynth.data.transforms import to_01_hwc, one_hot  # type: ignore
except Exception:  # pragma: no cover
    to_01_hwc = None  # type: ignore
    one_hot = None    # type: ignore

try:
    from gencysynth.data.cache import DatasetCache  # type: ignore
except Exception:  # pragma: no cover
    DatasetCache = None  # type: ignore


class NpyUSTCDataset(BaseDataset):
    """
    Loader for USTC-style .npy quartets.

    This is the direct scalable replacement for older gcs_core.data.load_npy_splits.
    """

    def __init__(self, dataset_id: str):
        super().__init__(dataset_id)

    def info(self) -> DatasetInfo:
        # Real values are config-driven; we provide placeholders here.
        return DatasetInfo(
            dataset_id=self.dataset_id,
            image_shape=(40, 40, 1),
            num_classes=9,
            class_names=tuple(str(i) for i in range(9)),
            description="USTC-TFC2016 malware images stored as .npy splits.",
        )

    def load_arrays(self, *, config: Dict[str, Any]) -> DatasetArrays:
        dcfg = self._dataset_cfg(config)

        raw_root = dcfg.get("raw_root") or dcfg.get("root") or dcfg.get("data_dir")
        if not isinstance(raw_root, str) or not raw_root:
            raise ValueError("NpyUSTCDataset requires config['dataset']['raw_root'] (directory with .npy files).")
        data_dir = Path(raw_root)

        # Required dataset invariants
        img_shape = dcfg.get("image_shape") or dcfg.get("img_shape") or [40, 40, 1]
        if not (isinstance(img_shape, (list, tuple)) and len(img_shape) == 3):
            raise ValueError("dataset.image_shape must be a 3-item list/tuple: [H,W,C].")
        H, W, C = int(img_shape[0]), int(img_shape[1]), int(img_shape[2])

        num_classes = int(dcfg.get("num_classes", 9))
        val_fraction = float(dcfg.get("val_fraction", 0.5))
        want_one_hot = bool(dcfg.get("one_hot", True))

        # ---------------------------------------------------------------------
        # 0) Optional cache
        # ---------------------------------------------------------------------
        cache_cfg = dcfg.get("cache") if isinstance(dcfg.get("cache"), dict) else {}
        cache_enabled = bool(cache_cfg.get("enabled", True))

        cache = None
        if cache_enabled and DatasetCache is not None:
            cache = DatasetCache.from_config(config=config, dataset_id=self.dataset_id)
            cache_key = {
                "dataset_id": self.dataset_id,
                "type": "npy_ustc",
                "raw_root": str(data_dir.resolve()),
                "image_shape": [H, W, C],
                "num_classes": num_classes,
                "val_fraction": val_fraction,
                "one_hot": want_one_hot,
            }
            hit = cache.try_load_arrays(cache_key)
            if hit is not None:
                return hit

        # ---------------------------------------------------------------------
        # 1) Load required files
        # ---------------------------------------------------------------------
        req = {
            "train_data": data_dir / "train_data.npy",
            "train_labels": data_dir / "train_labels.npy",
            "test_data": data_dir / "test_data.npy",
            "test_labels": data_dir / "test_labels.npy",
        }
        missing = [k for k, p in req.items() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing required USTC .npy files in {data_dir}: {missing}")

        x_train_raw = np.load(req["train_data"], allow_pickle=False)
        y_train_raw = np.load(req["train_labels"], allow_pickle=False)
        x_test_raw  = np.load(req["test_data"], allow_pickle=False)
        y_test_raw  = np.load(req["test_labels"], allow_pickle=False)

        # ---------------------------------------------------------------------
        # 2) Normalize images + labels
        # ---------------------------------------------------------------------
        if to_01_hwc is None:
            # Conservative fallback: basic float normalization only
            x_train01 = np.asarray(x_train_raw, dtype="float32")
            x_test01  = np.asarray(x_test_raw, dtype="float32")
            if x_train01.max() > 1.5:
                x_train01 /= 255.0
            if x_test01.max() > 1.5:
                x_test01 /= 255.0
            # Ensure NHWC
            if x_train01.ndim == 3:
                x_train01 = x_train01[..., None]
            if x_test01.ndim == 3:
                x_test01 = x_test01[..., None]
        else:
            x_train01 = to_01_hwc(x_train_raw, (H, W, C))
            x_test01  = to_01_hwc(x_test_raw,  (H, W, C))

        # Labels -> int or one-hot
        y_train = np.asarray(y_train_raw)
        y_test  = np.asarray(y_test_raw)

        if want_one_hot and one_hot is not None:
            y_train = one_hot(y_train, num_classes)
            y_test  = one_hot(y_test,  num_classes)
        else:
            # Ensure integer labels (N,)
            if y_train.ndim == 2:
                y_train = y_train.argmax(axis=1)
            if y_test.ndim == 2:
                y_test = y_test.argmax(axis=1)
            y_train = y_train.astype("int32", copy=False)
            y_test  = y_test.astype("int32", copy=False)

        # ---------------------------------------------------------------------
        # 3) Split provided test -> (val, test)
        # ---------------------------------------------------------------------
        n_val = int(len(x_test01) * float(val_fraction))
        n_val = max(0, min(n_val, len(x_test01)))

        x_val, y_val = x_test01[:n_val], y_test[:n_val]
        x_test, y_test = x_test01[n_val:], y_test[n_val:]

        arrays = DatasetArrays(
            x_train=x_train01, y_train=y_train,
            x_val=x_val,       y_val=y_val,
            x_test=x_test,     y_test=y_test,
        )

        # ---------------------------------------------------------------------
        # 4) Save to cache
        # ---------------------------------------------------------------------
        if cache is not None:
            cache.save_arrays(cache_key, arrays)

        return arrays
