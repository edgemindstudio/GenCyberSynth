# src/gencysynth/data/datasets/image_folder.py
"""
GenCyberSynth — ImageFolder Dataset

Use case
--------
Datasets stored as folders of images with class subfolders, typically like:

<raw_root>/
  train/
    class0/
      img1.png
      img2.png
    class1/
      ...
  val/
    class0/
    class1/
  test/
    class0/
    class1/

This loader converts the folder structure into arrays:
  - x_* float32 in [0,1] NHWC
  - y_* int labels (N,) by default (optionally one_hot)

Config contract
---------------
dataset:
  id: "my_dataset_v1"
  type: "image_folder"
  raw_root: "/path/to/raw_root"        # REQUIRED
  image_hw: [H, W]                     # optional (defaults to discovered size)
  channels: 1 or 3                     # optional (default: 3)
  splits: ["train","val","test"]       # optional
  one_hot: false                       # optional
  class_names: ["a","b","c"]           # optional (if you want a stable mapping)
  cache:
    enabled: true
    format: "npz"                      # default
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from gencysynth.data.datasets.base import BaseDataset, DatasetArrays, DatasetInfo

# Recommended shared helpers (optional but preferred)
try:
    from gencysynth.utils.paths import ensure_dir  # type: ignore
except Exception:  # pragma: no cover
    def ensure_dir(p: Path) -> Path:
        p.mkdir(parents=True, exist_ok=True)
        return p

try:
    from gencysynth.data.transforms import to_01_hwc, one_hot  # type: ignore
except Exception:  # pragma: no cover
    to_01_hwc = None  # type: ignore
    one_hot = None    # type: ignore

try:
    from gencysynth.data.cache import DatasetCache  # type: ignore
except Exception:  # pragma: no cover
    DatasetCache = None  # type: ignore


# =============================================================================
# Internals: reading images
# =============================================================================

def _read_image_rgb_or_gray(path: Path, *, channels: int, target_hw: Optional[Tuple[int, int]]) -> Optional[np.ndarray]:
    """
    Read one image file as float32 HWC in [0,1].

    - channels=3 -> RGB
    - channels=1 -> grayscale stored as (H,W,1)

    target_hw:
      - if provided, resize to (H,W) for consistency across samples
    """
    try:
        from PIL import Image  # type: ignore

        img = Image.open(path)
        img = img.convert("RGB") if int(channels) == 3 else img.convert("L")

        if target_hw is not None:
            # PIL expects (W,H) order
            img = img.resize((int(target_hw[1]), int(target_hw[0])), Image.NEAREST)

        arr = np.asarray(img).astype("float32") / 255.0

        if int(channels) == 1:
            # (H,W) -> (H,W,1)
            if arr.ndim == 2:
                arr = arr[..., None]
        else:
            # Ensure RGB (H,W,3)
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=-1)

        return arr
    except Exception:
        return None


def _discover_classes(split_dir: Path) -> List[str]:
    """Discover class folder names (sorted) from a split directory."""
    if not split_dir.exists():
        return []
    classes = [p.name for p in split_dir.iterdir() if p.is_dir()]
    classes.sort()
    return classes


def _gather_image_paths(split_dir: Path, class_names: List[str]) -> Tuple[List[Path], List[int]]:
    """
    Gather image file paths and integer labels for a split.

    Label mapping:
      class_names[i] -> label i
    """
    xs: List[Path] = []
    ys: List[int] = []

    for i, cname in enumerate(class_names):
        cdir = split_dir / cname
        if not cdir.exists():
            continue
        for p in cdir.rglob("*"):
            if not p.is_file():
                continue
            # Basic image extension filter (expand as needed)
            if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"):
                continue
            xs.append(p)
            ys.append(int(i))

    return xs, ys


# =============================================================================
# Dataset implementation
# =============================================================================

class ImageFolderDataset(BaseDataset):
    """
    ImageFolder dataset implementation.

    The primary goal is clarity + scalability:
    - raw data lives somewhere dataset_specific
    - dataset artifacts/caches live under artifacts/datasets/<dataset_id>/
    """

    def __init__(self, dataset_id: str):
        super().__init__(dataset_id)

    def info(self) -> DatasetInfo:
        # NOTE: full metadata is usually config_driven for image folder datasets,
        # because classes and shapes depend on the actual files.
        # We return placeholders here; the loader fills concrete values.
        return DatasetInfo(
            dataset_id=self.dataset_id,
            image_shape=(0, 0, 0),
            num_classes=0,
            class_names=None,
            description="ImageFolder dataset (train/val/test with class subfolders).",
        )

    def load_arrays(self, *, config: Dict[str, Any]) -> DatasetArrays:
        """
        Load splits into memory (optionally cached).

        The cache key includes:
        - dataset_id
        - raw_root path
        - image_hw/channels/one_hot options
        - (optionally) dataset fingerprint if you wire it in
        """
        dcfg = self._dataset_cfg(config)

        raw_root = dcfg.get("raw_root")
        if not isinstance(raw_root, str) or not raw_root:
            raise ValueError("ImageFolderDataset requires config['dataset']['raw_root'] as a valid path string.")

        raw_root_p = Path(raw_root)

        # Split names (default to train/val/test)
        splits = dcfg.get("splits") or ["train", "val", "test"]
        if not isinstance(splits, list) or len(splits) == 0:
            splits = ["train", "val", "test"]
        splits = [str(s) for s in splits]

        # Channel policy (default: 3)
        channels = int(dcfg.get("channels", 3))
        if channels not in (1, 3):
            raise ValueError("dataset.channels must be 1 or 3 for ImageFolderDataset.")

        # Optional fixed size (recommended for stability)
        hw = dcfg.get("image_hw")
        target_hw: Optional[Tuple[int, int]] = None
        if isinstance(hw, (list, tuple)) and len(hw) == 2:
            try:
                target_hw = (int(hw[0]), int(hw[1]))
            except Exception:
                target_hw = None

        want_one_hot = bool(dcfg.get("one_hot", False))

        # ---------------------------------------------------------------------
        # 0) Optional cache (dataset_scoped artifacts)
        # ---------------------------------------------------------------------
        cache_cfg = dcfg.get("cache") if isinstance(dcfg.get("cache"), dict) else {}
        cache_enabled = bool(cache_cfg.get("enabled", True))

        cache = None
        if cache_enabled and DatasetCache is not None:
            cache = DatasetCache.from_config(config=config, dataset_id=self.dataset_id)

            # Build a cache key that changes when important knobs change.
            # NOTE: we keep it simple; you can include dataset fingerprint hash later.
            cache_key = {
                "dataset_id": self.dataset_id,
                "type": "image_folder",
                "raw_root": str(raw_root_p.resolve()),
                "splits": splits,
                "channels": channels,
                "target_hw": list(target_hw) if target_hw else None,
                "one_hot": want_one_hot,
            }

            hit = cache.try_load_arrays(cache_key)
            if hit is not None:
                return hit

        # ---------------------------------------------------------------------
        # 1) Determine stable class mapping
        # ---------------------------------------------------------------------
        # If user provides class_names, we respect it (stable label mapping).
        # Otherwise, discover from train split folders (sorted).
        class_names_cfg = dcfg.get("class_names")
        if isinstance(class_names_cfg, list) and len(class_names_cfg) > 0:
            class_names = [str(x) for x in class_names_cfg]
        else:
            train_dir = raw_root_p / "train"
            class_names = _discover_classes(train_dir)
            if not class_names:
                raise FileNotFoundError(
                    f"No class folders discovered under {train_dir}. "
                    "Provide dataset.class_names or ensure train/<class>/ exists."
                )

        num_classes = len(class_names)

        # ---------------------------------------------------------------------
        # 2) Load each split to arrays
        # ---------------------------------------------------------------------
        def _load_split(split_name: str) -> Tuple[np.ndarray, np.ndarray]:
            split_dir = raw_root_p / split_name
            paths, y_int = _gather_image_paths(split_dir, class_names)

            if not paths:
                # Return empty arrays with correct rank
                x_empty = np.zeros((0, *(target_hw or (0, 0)), channels), dtype="float32")
                y_empty = np.zeros((0,), dtype="int32")
                return x_empty, y_empty

            xs: List[np.ndarray] = []
            ys: List[int] = []

            for p, y in zip(paths, y_int):
                arr = _read_image_rgb_or_gray(p, channels=channels, target_hw=target_hw)
                if arr is None:
                    continue
                xs.append(arr)
                ys.append(int(y))

            if not xs:
                x_empty = np.zeros((0, *(target_hw or (0, 0)), channels), dtype="float32")
                y_empty = np.zeros((0,), dtype="int32")
                return x_empty, y_empty

            x = np.stack(xs, axis=0).astype("float32", copy=False)
            y = np.asarray(ys, dtype="int32")

            # Normalize/reshape to NHWC contract if shared transform exists
            if to_01_hwc is not None and target_hw is not None:
                x = to_01_hwc(x, (target_hw[0], target_hw[1], channels))

            if want_one_hot and one_hot is not None:
                y = one_hot(y, num_classes)

            return x, y

        x_train, y_train = _load_split("train")
        x_val, y_val     = _load_split("val")
        x_test, y_test   = _load_split("test")

        arrays = DatasetArrays(
            x_train=x_train, y_train=y_train,
            x_val=x_val,     y_val=y_val,
            x_test=x_test,   y_test=y_test,
        )

        # ---------------------------------------------------------------------
        # 3) Save to cache (if enabled)
        # ---------------------------------------------------------------------
        if cache is not None:
            cache.save_arrays(cache_key, arrays)

        return arrays
