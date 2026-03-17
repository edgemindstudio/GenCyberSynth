# src/gencysynth/eval/splits.py
"""
GenCyberSynth — Real Dataset Splits (dataset_scalable)
=====================================================

Purpose
-------
This module provides **one place** to resolve and load REAL dataset splits in a
multi_dataset GenCyberSynth repo.

Why this exists
---------------
In the legacy repo, many scripts assumed a single dataset folder and hard_coded
filenames like `train_data.npy`. GenCyberSynth now supports **multiple datasets**,
so we need a consistent, scalable way to:

1) Resolve which dataset we are evaluating (dataset_id + root path)
2) Load REAL splits into arrays (train / val / test)
3) Provide small helpers to:
   - convert labels to integer ids
   - create val/test splits if the dataset provides only train/test
   - create *per_class capped* subsets (used by FID/cFID/KID protocols)

Design principles
-----------------
- This module **is allowed to read from disk** (unlike eval_common.py).
- It should be robust: clear errors, helpful messages, safe defaults.
- It is dataset_scalable: dataset_id + dataset.root drive everything.
- It does NOT assume a single data format; a simple "format" switch enables growth.

Expected config contract (recommended)
--------------------------------------
paths:
  artifacts: "artifacts"         # used elsewhere, not required here
  data_root: "data"              # optional global base directory for datasets

dataset:
  id: "USTC_TFC2016_40x40_gray"  # REQUIRED for scalable artifacts layout
  root: "data/ustc"              # directory containing dataset files
  format: "npy_ustc"             # dataset loader type
  image_shape: [40, 40, 1]       # H,W,C
  num_classes: 9
  val_fraction: 0.5              # when dataset provides only train/test
  seed: 42                       # deterministic split for val/test
  files:                          # optional override of default filenames
    train_data: "train_data.npy"
    train_labels: "train_labels.npy"
    test_data: "test_data.npy"
    test_labels: "test_labels.npy"

Notes on dataset formats
------------------------
Currently implemented:
- "npy_ustc": the classic USTC_TFC2016_style quartet of .npy files.

You can add additional formats later (e.g., "image_folder", "parquet", "hf_dataset")
without touching eval_common.py — only update this module and data/loaders.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import os
import numpy as np

# Recommended: loaders + transforms live under gencysynth/data/
# (If these modules are not present yet, the import error will clearly tell you.)
from gencysynth.data.loaders import load_npy_ustc_splits
from gencysynth.data.transforms import to_01_hwc
from gencysynth.utils.reproducibility import set_global_seed


# =============================================================================
# Public dataclasses
# =============================================================================
@dataclass(frozen=True)
class RealSplits:
    """
    Container for REAL splits (train/val/test).

    All images are float32 in [0,1] and NHWC. Labels are:
    - y_*_int: (N,) int class ids in [0..K_1]
    - y_*_oh:  (N,K) float32 one_hot (optional; computed on demand)
    """
    dataset_id: str
    img_shape: Tuple[int, int, int]
    num_classes: int

    x_train: np.ndarray
    y_train_int: np.ndarray

    x_val: np.ndarray
    y_val_int: np.ndarray

    x_test: np.ndarray
    y_test_int: np.ndarray

    # Optional provenance (paths, hashes, etc.)
    meta: Dict[str, Any]


@dataclass(frozen=True)
class FidCapSelection:
    """
    Indices of per_class capped subsets used for FID/cFID/KID protocols.
    """
    cap_per_class: int
    seed: int
    idx_val: np.ndarray   # indices into x_val/y_val_int
    idx_synth: np.ndarray # indices into x_synth/y_synth_int


# =============================================================================
# Config helpers
# =============================================================================
def _cfg_get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    """Fetch a nested config value by dotted path, e.g. 'dataset.root'."""
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _resolve_dataset_id(cfg: Dict[str, Any]) -> str:
    dsid = _cfg_get(cfg, "dataset.id", None)
    if isinstance(dsid, str) and dsid.strip():
        return dsid.strip()
    # We avoid hard_crashing here because some legacy scripts may not supply it.
    # But: multi_dataset scaling REALLY wants a stable dataset_id.
    return "unknown_dataset"


def _resolve_dataset_root(cfg: Dict[str, Any]) -> Path:
    """
    Resolve dataset root directory.

    Priority:
      1) cfg.dataset.root (recommended)
      2) cfg.paths.data_root + cfg.dataset.id  (scalable convention)
      3) env GENCSYNTH_DATA_ROOT + cfg.dataset.id
      4) fallback to "./data/<dataset_id>" (reasonable default)
    """
    ds_root = _cfg_get(cfg, "dataset.root", None)
    if isinstance(ds_root, str) and ds_root.strip():
        return Path(ds_root).expanduser()

    dsid = _resolve_dataset_id(cfg)

    data_root = _cfg_get(cfg, "paths.data_root", None)
    if isinstance(data_root, str) and data_root.strip():
        return (Path(data_root).expanduser() / dsid)

    env_root = os.environ.get("GENCSYNTH_DATA_ROOT")
    if isinstance(env_root, str) and env_root.strip():
        return (Path(env_root).expanduser() / dsid)

    return Path("data") / dsid


def _resolve_img_shape(cfg: Dict[str, Any]) -> Tuple[int, int, int]:
    shape = _cfg_get(cfg, "dataset.image_shape", _cfg_get(cfg, "dataset.img_shape", None))
    if isinstance(shape, (list, tuple)) and len(shape) == 3:
        return int(shape[0]), int(shape[1]), int(shape[2])
    # Safe default for your current primary dataset
    return (40, 40, 1)


def _resolve_num_classes(cfg: Dict[str, Any]) -> int:
    k = _cfg_get(cfg, "dataset.num_classes", None)
    if k is None:
        # default for USTC_TFC2016 malware classes in your project
        return 9
    return int(k)


def _resolve_dataset_format(cfg: Dict[str, Any]) -> str:
    fmt = _cfg_get(cfg, "dataset.format", None)
    return str(fmt).strip() if isinstance(fmt, str) and fmt.strip() else "npy_ustc"


def _resolve_split_seed(cfg: Dict[str, Any]) -> int:
    s = _cfg_get(cfg, "dataset.seed", _cfg_get(cfg, "SEED", _cfg_get(cfg, "seed", 42)))
    try:
        return int(s)
    except Exception:
        return 42


def _resolve_val_fraction(cfg: Dict[str, Any]) -> float:
    vf = _cfg_get(cfg, "dataset.val_fraction", 0.5)
    try:
        vf = float(vf)
    except Exception:
        vf = 0.5
    return max(0.0, min(1.0, vf))


# =============================================================================
# Label utilities
# =============================================================================
def _to_int_labels(y: np.ndarray) -> np.ndarray:
    """
    Convert labels to integer ids:
      - (N,) -> unchanged
      - (N,K) -> argmax
    """
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] > 1:
        return y.argmax(axis=1).astype(np.int32, copy=False)
    return y.astype(np.int32, copy=False)


# =============================================================================
# Public API: load REAL splits (dataset_scalable)
# =============================================================================
def load_real_splits(cfg: Dict[str, Any]) -> RealSplits:
    """
    Load REAL dataset splits using cfg.dataset.{id,root,format,...}.

    Returns
    -------
    RealSplits
        Images are float32 in [0,1], NHWC.
        Labels are int class ids (N,).
    """
    dataset_id = _resolve_dataset_id(cfg)
    dataset_root = _resolve_dataset_root(cfg)
    fmt = _resolve_dataset_format(cfg)

    img_shape = _resolve_img_shape(cfg)
    num_classes = _resolve_num_classes(cfg)
    seed = _resolve_split_seed(cfg)
    val_fraction = _resolve_val_fraction(cfg)

    # Ensure deterministic split decisions
    set_global_seed(seed)

    meta: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "dataset_root": str(dataset_root),
        "dataset_format": fmt,
        "img_shape": img_shape,
        "num_classes": num_classes,
        "seed": seed,
        "val_fraction": val_fraction,
    }

    if fmt == "npy_ustc":
        # The loader encapsulates file naming and the val/test split behavior.
        xtr, ytr, xva, yva, xte, yte, files_meta = load_npy_ustc_splits(
            data_dir=dataset_root,
            img_shape=img_shape,
            num_classes=num_classes,
            val_fraction=val_fraction,
            seed=seed,
            files=_cfg_get(cfg, "dataset.files", None),
        )
        meta.update(files_meta or {})
    else:
        raise ValueError(
            f"Unsupported dataset.format='{fmt}'. "
            "Add a loader in gencysynth.data.loaders and register it here."
        )

    # Normalize images (belt & suspenders). Most loaders already do this.
    xtr = to_01_hwc(xtr, img_shape)
    xva = to_01_hwc(xva, img_shape)
    xte = to_01_hwc(xte, img_shape)

    # Labels -> int ids (callers can one_hot inside eval_common if needed)
    ytr_int = _to_int_labels(ytr)
    yva_int = _to_int_labels(yva)
    yte_int = _to_int_labels(yte)

    # Minimal sanity checks (catch path mistakes early)
    if len(xtr) != len(ytr_int):
        raise ValueError(f"Train count mismatch: x={len(xtr)} vs y={len(ytr_int)}")
    if len(xva) != len(yva_int):
        raise ValueError(f"Val count mismatch: x={len(xva)} vs y={len(yva_int)}")
    if len(xte) != len(yte_int):
        raise ValueError(f"Test count mismatch: x={len(xte)} vs y={len(yte_int)}")

    return RealSplits(
        dataset_id=dataset_id,
        img_shape=img_shape,
        num_classes=num_classes,
        x_train=xtr,
        y_train_int=ytr_int,
        x_val=xva,
        y_val_int=yva_int,
        x_test=xte,
        y_test_int=yte_int,
        meta=meta,
    )


# =============================================================================
# Public API: per_class capped selections (FID/cFID/KID protocol helper)
# =============================================================================
def select_per_class_cap(
    y_int: np.ndarray,
    cap_per_class: int,
    *,
    seed: int = 42,
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """
    Return indices selecting up to `cap_per_class` samples per class.

    This is a core building block for FID/cFID/KID fairness protocols:
      - always use the same cap per class across models
      - deterministic selection given (seed)

    Parameters
    ----------
    y_int : np.ndarray
        Integer labels shape (N,)
    cap_per_class : int
        Max samples per class (>=1)
    seed : int
        RNG seed for deterministic selection
    num_classes : Optional[int]
        If provided, iterates classes 0..K_1 in stable order. Otherwise uses np.unique.

    Returns
    -------
    np.ndarray
        1_D array of selected indices into the original arrays.
    """
    y = _to_int_labels(y_int)
    cap = max(1, int(cap_per_class))

    rng = np.random.default_rng(int(seed))
    selected: list[int] = []

    classes = list(range(int(num_classes))) if num_classes is not None else list(np.unique(y))
    for k in classes:
        idx = np.where(y == int(k))[0]
        if len(idx) == 0:
            continue
        if len(idx) <= cap:
            selected.extend(idx.tolist())
        else:
            chosen = rng.choice(idx, size=cap, replace=False)
            selected.extend(chosen.tolist())

    # Stable output: sort indices so downstream slicing keeps deterministic order.
    return np.asarray(sorted(selected), dtype=np.int64)


def make_fid_cap_selection(
    *,
    y_val_int: np.ndarray,
    y_synth_int: np.ndarray,
    cap_per_class: int,
    seed: int,
    num_classes: Optional[int] = None,
) -> FidCapSelection:
    """
    Build deterministic per_class capped index selections for:
      - REAL validation subset
      - SYNTH subset

    This supports:
      - FID macro (pooled across selected indices)
      - cFID per_class (computed class_by_class with same cap)
      - KID (same pooled selection)

    Returns
    -------
    FidCapSelection
        Contains idx_val and idx_synth.
    """
    idx_val = select_per_class_cap(
        y_val_int,
        cap_per_class,
        seed=seed,
        num_classes=num_classes,
    )
    idx_synth = select_per_class_cap(
        y_synth_int,
        cap_per_class,
        seed=seed + 999,  # deterministic but not identical sampling stream
        num_classes=num_classes,
    )
    return FidCapSelection(
        cap_per_class=int(cap_per_class),
        seed=int(seed),
        idx_val=idx_val,
        idx_synth=idx_synth,
    )


__all__ = [
    "RealSplits",
    "FidCapSelection",
    "load_real_splits",
    "select_per_class_cap",
    "make_fid_cap_selection",
]