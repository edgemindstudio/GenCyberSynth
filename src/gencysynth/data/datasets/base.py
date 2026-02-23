# src/gencysynth/data/datasets/base.py
"""
GenCyberSynth — Dataset Abstractions (Scalable, Multi-Dataset)

Why this exists
---------------
GenCyberSynth is designed to support multiple datasets without code forks.
This module defines a small, stable interface for dataset implementations.

Key ideas
---------
1) Every dataset must have a stable dataset_id (e.g. "USTC-TFC2016_40x40_gray")
2) Every dataset implementation must be able to:
   - Describe itself (id, image shape, num classes, class names if known)
   - Resolve where its raw data lives (config-based)
   - Load splits into arrays (train/val/test), optionally cached

This file does NOT assume any particular dataset layout.
Concrete datasets (npy_ustc, image_folder, etc.) implement the abstract hooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Protocol, runtime_checkable

import numpy as np


# =============================================================================
# Split container (arrays)
# =============================================================================

@dataclass(frozen=True)
class DatasetArrays:
    """
    A simple container for dataset splits loaded into memory.

    Shapes (recommended contract)
    -----------------------------
    x_*: np.ndarray, float32, in [0,1], NHWC -> (N,H,W,C)
    y_*: np.ndarray, either:
         - int labels (N,) OR
         - one-hot float32 (N,K)

    NOTE:
    - We do not force one-hot here because some pipelines prefer int labels.
    - Downstream evaluation utilities can convert as needed.
    """
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


# =============================================================================
# Dataset metadata
# =============================================================================

@dataclass(frozen=True)
class DatasetInfo:
    """
    Dataset identity + invariants that should not change between runs.

    This information is also helpful for:
    - fingerprinting datasets
    - validating configs
    - ensuring models know expected input sizes
    """
    dataset_id: str
    image_shape: Tuple[int, int, int]         # (H,W,C)
    num_classes: int
    class_names: Optional[Tuple[str, ...]] = None

    # Optional descriptive fields (nice for reports/logs)
    description: Optional[str] = None
    homepage: Optional[str] = None
    license: Optional[str] = None


# =============================================================================
# Dataset interface
# =============================================================================

@runtime_checkable
class Dataset(Protocol):
    """
    Minimal protocol for datasets in GenCyberSynth.

    Why Protocol?
    -------------
    - Keeps the code flexible and easy to test.
    - Concrete datasets don't need to inherit from a base class if you prefer
      composition. But we provide an optional BaseDataset below for convenience.
    """

    def info(self) -> DatasetInfo:
        """Return stable dataset metadata (id/shape/classes)."""
        ...

    def load_arrays(self, *, config: Dict[str, Any]) -> DatasetArrays:
        """
        Load train/val/test arrays for this dataset.

        Rules:
        - Must read from a dataset-specific raw location (defined by config).
        - Should normalize images to float32 in [0,1] NHWC.
        - Should support caching when config enables it (recommended).
        """
        ...


# =============================================================================
# Optional convenience base class
# =============================================================================

class BaseDataset:
    """
    Convenience base class implementing small shared behaviors:
    - standard config reads for artifacts_root and dataset section
    - safe dataset_id handling

    Concrete dataset loaders should inherit and implement `load_arrays(...)`.
    """

    def __init__(self, dataset_id: str):
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise ValueError("dataset_id must be a non-empty string.")
        self._dataset_id = dataset_id.strip()

    @property
    def dataset_id(self) -> str:
        return self._dataset_id

    def _artifacts_root(self, config: Dict[str, Any]) -> Path:
        """
        Where artifacts live (outputs, cache, fingerprints, etc.)

        Convention:
          config["paths"]["artifacts"] (preferred)
          else default "artifacts"
        """
        paths = config.get("paths") if isinstance(config.get("paths"), dict) else {}
        return Path(paths.get("artifacts", "artifacts"))

    def _dataset_cfg(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Return config['dataset'] dict safely."""
        d = config.get("dataset")
        return d if isinstance(d, dict) else {}

    def _dataset_root(self, config: Dict[str, Any]) -> Path:
        """
        Dataset-scoped artifacts root (NOT the raw dataset location).

        This is where we store:
          - fingerprints
          - caches
          - dataset metadata snapshots

        Rule:
          <artifacts_root>/datasets/<dataset_id>/
        """
        return self._artifacts_root(config) / "datasets" / self.dataset_id
