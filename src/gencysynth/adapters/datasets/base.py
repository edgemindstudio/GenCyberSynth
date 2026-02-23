# src/gencysynth/adapters/datasets/base.py
"""
DatasetAdapter protocol (front-door for adapters).

This module defines a stable, adapter-facing dataset interface.
It is intentionally small so dataset formats can evolve behind it.

Rule A
------
Dataset identity is always `dataset_id`, and each dataset adapter is responsible
for mapping that ID + cfg into concrete data loading behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable

from .splits import DatasetSplits


@dataclass(frozen=True)
class DatasetSpec:
    """
    Static identity/metadata for a dataset adapter.
    """
    dataset_id: str                   # e.g. "ustc_tfc2016_npy"
    description: str = ""
    default_img_shape: Tuple[int, int, int] = (40, 40, 1)
    default_num_classes: int = 9


@runtime_checkable
class DatasetAdapter(Protocol):
    """
    Adapter-facing dataset loader.

    Responsibilities
    ----------------
    - load raw dataset via gencysynth.data.* tools
    - return standardized DatasetSplits:
        * x01 in [0,1], NHWC float32
        * y_int int64
        * y_onehot float32
    - honor cfg (paths, split fractions, class mapping, etc.)

    Non-responsibilities
    --------------------
    - writing run artifacts (that belongs to orchestration / adapters)
    - computing dataset fingerprints (handled by gencysynth.data.fingerprint + schema writers)
    """

    spec: DatasetSpec

    def load(self, cfg: Dict[str, Any]) -> DatasetSplits:
        """
        Load dataset splits as standardized arrays.

        cfg should already be merged, and typically includes:
          - data.root or DATA_DIR
          - IMG_SHAPE, NUM_CLASSES
          - split controls (val_fraction, test_fraction, or explicit splits)

        Returns DatasetSplits with at least .train set.
        """
        ...
