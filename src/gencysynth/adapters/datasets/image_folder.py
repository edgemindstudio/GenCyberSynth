# src/gencysynth/adapters/datasets/image_folder.py
"""
Generic image-folder dataset adapter.

Expected layout
---------------
<data_root>/
  class0/
    img1.png
    img2.png
  class1/
    ...

This adapter delegates heavy lifting to gencysynth.data.datasets.image_folder
(where available), then converts into DatasetSplits.

Config keys
-----------
data.root or DATA_DIR:
  Root folder.

IMG_SHAPE:
  Expected output shape (H,W,C). Resize is handled by gencysynth.data transforms.

NUM_CLASSES:
  Optional if inferable; otherwise required.

VAL_FRACTION / TEST_FRACTION:
  Fractions used to split the dataset into train/val/test (defaults: 0.1/0.1)

SEED, SHUFFLE:
  Deterministic splitting behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from .base import DatasetAdapter, DatasetSpec
from .splits import DatasetSplits, SplitArrays
from gencysynth.adapters.normalize import labels_to_onehot, to_float01

# Bridge to your existing dataset infra.
# If your internal API differs, adjust here; the adapter contract remains stable.
from gencysynth.data.loaders import load_image_folder_dataset  # <-- expected helper


def _cfg_get(cfg: Dict[str, Any], dotted: str, default=None):
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


@dataclass
class ImageFolderDatasetAdapter:
    spec: DatasetSpec = DatasetSpec(
        dataset_id="image_folder",
        description="Generic image-folder dataset (class subfolders) -> standardized splits",
        default_img_shape=(40, 40, 1),
        default_num_classes=0,
    )

    def load(self, cfg: Dict[str, Any]) -> DatasetSplits:
        data_root = Path(cfg.get("DATA_DIR", _cfg_get(cfg, "data.root", "data")))
        img_shape: Tuple[int, int, int] = tuple(cfg.get("IMG_SHAPE", self.spec.default_img_shape))

        val_fraction = float(cfg.get("VAL_FRACTION", 0.1))
        test_fraction = float(cfg.get("TEST_FRACTION", 0.1))
        seed = int(cfg.get("SEED", 42))
        shuffle = bool(cfg.get("SHUFFLE", True))

        # Delegate to core data loaders
        # Expected return:
        #   x_train, y_train_int, x_val, y_val_int, x_test, y_test_int, num_classes
        x_train, y_train_int, x_val, y_val_int, x_test, y_test_int, num_classes = load_image_folder_dataset(
            root=str(data_root),
            img_shape=img_shape,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
            shuffle=shuffle,
        )

        x_train01 = to_float01(x_train)
        y_train_int = np.asarray(y_train_int).astype(np.int64, copy=False)
        y_train_oh = labels_to_onehot(y_train_int, num_classes=int(num_classes))

        train = SplitArrays(x01=x_train01, y_int=y_train_int, y_onehot=y_train_oh)

        val = None
        if x_val is not None and y_val_int is not None:
            xv01 = to_float01(x_val)
            yv_int = np.asarray(y_val_int).astype(np.int64, copy=False)
            yv_oh = labels_to_onehot(yv_int, num_classes=int(num_classes))
            val = SplitArrays(x01=xv01, y_int=yv_int, y_onehot=yv_oh)

        test = None
        if x_test is not None and y_test_int is not None:
            xt01 = to_float01(x_test)
            yt_int = np.asarray(y_test_int).astype(np.int64, copy=False)
            yt_oh = labels_to_onehot(yt_int, num_classes=int(num_classes))
            test = SplitArrays(x01=xt01, y_int=yt_int, y_onehot=yt_oh)

        return DatasetSplits(train=train, val=val, test=test)


from .registry import register_dataset_adapter
register_dataset_adapter("image_folder", lambda: ImageFolderDatasetAdapter())

__all__ = ["ImageFolderDatasetAdapter"]
