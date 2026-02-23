# src/gencysynth/data/loaders.py
"""
gencysynth.data.loaders
======================

Dataset-aware data loading for GenCyberSynth.

Key idea (multi-dataset scalability)
------------------------------------
Loaders must NOT assume a single dataset folder name (like USTC-TFC2016_malware).
Instead, we load from:
  - an explicit dataset root directory in config, OR
  - a structured data root convention, OR
  - a DatasetSpec

This module supports the standard GenCyberSynth "npy quartet" classification dataset:
  train_data.npy / train_labels.npy
  test_data.npy  / test_labels.npy

and then splits the provided test set into (val, test) using val_fraction.

Outputs
-------
- x_* are float32 in [0,1], NHWC (N,H,W,C)
- y_* are float32 one-hot (N,K)

No TensorFlow dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from gencysynth.data.transforms import (
    one_hot,
    to_01_hwc,
    split_val_from_test,
    dataset_counts,
)
from gencysynth.data.specs import DatasetSpec

__all__ = [
    "resolve_dataset_root",
    "load_npy_classification_splits",
    "load_dataset_from_config",
    "dataset_counts",
]


# -----------------------------------------------------------------------------
# Path policy: dataset root resolution (multi-dataset scalable)
# -----------------------------------------------------------------------------
def resolve_dataset_root(
    config: Dict[str, Any],
    *,
    dataset_id: Optional[str] = None,
) -> Path:
    """
    Resolve the dataset root directory in a scalable way.

    Priority order:
      1) config["dataset"]["root"]                 (explicit root; best)
      2) config["data"]["root"]                    (legacy support)
      3) config["paths"]["data_root"] / <dataset_id> (structured convention)
      4) default: "./data/<dataset_id>" (last resort)

    Notes
    -----
    - dataset_id should be stable and used across artifacts paths.
    - This function DOES NOT create directories; it only resolves a path.
    """
    ds = None
    if isinstance(config.get("dataset"), dict):
        ds = config["dataset"]

    dataset_id = dataset_id or (ds.get("id") if isinstance(ds, dict) else None) or "unknown_dataset"

    # 1) explicit dataset.root
    if isinstance(ds, dict) and isinstance(ds.get("root"), str) and ds["root"]:
        return Path(ds["root"])

    # 2) legacy support: data.root
    if isinstance(config.get("data"), dict):
        dr = config["data"].get("root")
        if isinstance(dr, str) and dr:
            return Path(dr)

    # 3) structured convention: paths.data_root/<dataset_id>
    if isinstance(config.get("paths"), dict):
        pr = config["paths"].get("data_root")
        if isinstance(pr, str) and pr:
            return Path(pr) / str(dataset_id)

    # 4) last resort: ./data/<dataset_id>
    return Path("data") / str(dataset_id)


# -----------------------------------------------------------------------------
# Core loader: .npy classification quartet
# -----------------------------------------------------------------------------
def load_npy_classification_splits(
    *,
    data_dir: Path | str,
    img_shape: Tuple[int, int, int],
    num_classes: int,
    val_fraction: float = 0.5,
    filenames: Optional[Dict[str, str]] = None,
    mmap: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load classification splits from the standard .npy quartet:
      - train_data.npy / train_labels.npy
      - test_data.npy  / test_labels.npy

    Then split the provided test set into (val, test) by val_fraction.

    Parameters
    ----------
    data_dir:
        Directory containing the dataset files.
    img_shape:
        (H,W,C) expected output for images.
    num_classes:
        Number of label classes (K).
    val_fraction:
        Fraction of the provided test set assigned to validation.
    filenames:
        Optional override mapping:
          {
            "train_data": "train_data.npy",
            "train_labels": "train_labels.npy",
            "test_data": "test_data.npy",
            "test_labels": "test_labels.npy",
          }
    mmap:
        If True, use numpy memmap for large arrays (good for huge datasets).

    Returns
    -------
    (x_train01, y_train1h, x_val01, y_val1h, x_test01, y_test1h)

    Raises
    ------
    FileNotFoundError if required files are missing.
    ValueError for shape/label issues.
    """
    data_dir = Path(data_dir)

    fn = {
        "train_data": "train_data.npy",
        "train_labels": "train_labels.npy",
        "test_data": "test_data.npy",
        "test_labels": "test_labels.npy",
    }
    if filenames:
        fn.update({k: v for k, v in filenames.items() if isinstance(v, str) and v})

    paths = {
        k: (data_dir / v)
        for k, v in fn.items()
    }
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        pretty = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing required dataset files in {data_dir}: {pretty}. "
            f"Expected npy quartet: {list(paths.values())}"
        )

    load_kwargs = {"allow_pickle": False}
    if mmap:
        load_kwargs["mmap_mode"] = "r"

    x_train_raw = np.load(paths["train_data"], **load_kwargs)
    y_train_raw = np.load(paths["train_labels"], **load_kwargs)
    x_test_raw = np.load(paths["test_data"], **load_kwargs)
    y_test_raw = np.load(paths["test_labels"], **load_kwargs)

    # Normalize to float32 NHWC in [0,1]
    x_train01 = to_01_hwc(x_train_raw, img_shape)
    x_test01 = to_01_hwc(x_test_raw, img_shape)

    # Labels -> one-hot float32
    y_train1h = one_hot(y_train_raw, num_classes)
    y_test1h = one_hot(y_test_raw, num_classes)

    # Split test into val/test
    x_val01, y_val1h, x_test01, y_test1h = split_val_from_test(
        x_test01, y_test1h, val_fraction=val_fraction
    )

    # Sanity checks (clear messages)
    if x_train01.shape[0] != y_train1h.shape[0]:
        raise ValueError(
            f"Train count mismatch: images={x_train01.shape[0]} vs labels={y_train1h.shape[0]}"
        )
    if x_val01.shape[0] != y_val1h.shape[0]:
        raise ValueError(
            f"Val count mismatch: images={x_val01.shape[0]} vs labels={y_val1h.shape[0]}"
        )
    if x_test01.shape[0] != y_test1h.shape[0]:
        raise ValueError(
            f"Test count mismatch: images={x_test01.shape[0]} vs labels={y_test1h.shape[0]}"
        )

    return x_train01, y_train1h, x_val01, y_val1h, x_test01, y_test1h


# -----------------------------------------------------------------------------
# High-level config-driven loader (recommended entrypoint)
# -----------------------------------------------------------------------------
def load_dataset_from_config(
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Recommended single entrypoint: load REAL splits based on config.

    Expected config fields (flexible, multi-dataset scalable)
    --------------------------------------------------------
    dataset:
      id: "USTC-TFC2016_40x40_gray"
      root: "/path/to/dataset"              # optional, preferred
      img_shape: [40,40,1]
      num_classes: 9
      val_fraction: 0.5                     # optional
      files:                                # optional file-name overrides
        train_data: "train_data.npy"
        train_labels: "train_labels.npy"
        test_data: "test_data.npy"
        test_labels: "test_labels.npy"
    paths:
      data_root: "data"                     # optional convention root

    Returns:
      xtr, ytr, xva, yva, xte, yte, meta

    meta includes:
      - dataset_id
      - dataset_root
      - img_shape
      - num_classes
      - val_fraction
      - file_paths used
    """
    ds = config.get("dataset") if isinstance(config.get("dataset"), dict) else {}
    dataset_id = ds.get("id") if isinstance(ds.get("id"), str) and ds.get("id") else "unknown_dataset"

    # img_shape is required for correct NHWC reshaping
    img_shape = ds.get("img_shape")
    if not (isinstance(img_shape, (list, tuple)) and len(img_shape) == 3):
        raise ValueError("config['dataset']['img_shape'] must be [H,W,C].")
    img_shape_t = (int(img_shape[0]), int(img_shape[1]), int(img_shape[2]))

    # num_classes required for one_hot width
    num_classes = ds.get("num_classes")
    if num_classes is None:
        raise ValueError("config['dataset']['num_classes'] must be set.")
    num_classes_i = int(num_classes)

    val_fraction = float(ds.get("val_fraction", 0.5))

    # optional filename overrides
    filenames = ds.get("files") if isinstance(ds.get("files"), dict) else None

    dataset_root = resolve_dataset_root(config, dataset_id=dataset_id)

    # Build file paths meta (for audit)
    fn = {
        "train_data": "train_data.npy",
        "train_labels": "train_labels.npy",
        "test_data": "test_data.npy",
        "test_labels": "test_labels.npy",
    }
    if filenames:
        fn.update({k: v for k, v in filenames.items() if isinstance(v, str) and v})

    file_paths = {k: str(dataset_root / v) for k, v in fn.items()}

    xtr, ytr, xva, yva, xte, yte = load_npy_classification_splits(
        data_dir=dataset_root,
        img_shape=img_shape_t,
        num_classes=num_classes_i,
        val_fraction=val_fraction,
        filenames=fn,
        mmap=bool(ds.get("mmap", False)),
    )

    meta: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "dataset_root": str(dataset_root),
        "img_shape": list(img_shape_t),
        "num_classes": num_classes_i,
        "val_fraction": val_fraction,
        "files": file_paths,
    }
    return xtr, ytr, xva, yva, xte, yte, meta
