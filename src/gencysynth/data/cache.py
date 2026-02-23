# src/gencysynth/data/cache.py
"""
GenCyberSynth — Dataset Cache (Scalable, Dataset-Scoped)

Why this exists
---------------
Loading datasets (especially large ones) repeatedly on HPC can be expensive.
This module provides a small caching layer for dataset splits.

Scalable location policy
------------------------
We store cached dataset artifacts under:

  <artifacts_root>/datasets/<dataset_id>/cache/

This keeps caches dataset-scoped (no collisions between datasets), and makes it
easy to purge or version dataset caches independently.

What is cached
--------------
- Train/Val/Test arrays (x/y)
- A cache_key.json with the parameters that produced the cached arrays

Cache key
---------
The cache_key is a JSON-serializable dict. We hash it (SHA1) to get a stable filename.

Example cache file:
  artifacts/datasets/USTC-TFC2016_40x40_gray/cache/arrays_<sha1>.npz
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from gencysynth.data.datasets.base import DatasetArrays

try:
    from gencysynth.utils.paths import ensure_dir  # type: ignore
except Exception:  # pragma: no cover
    def ensure_dir(p: Path) -> Path:
        p.mkdir(parents=True, exist_ok=True)
        return p

try:
    from gencysynth.utils.io import write_json, read_json  # type: ignore
except Exception:  # pragma: no cover
    write_json = None  # type: ignore
    read_json = None   # type: ignore


def _stable_json(obj: Any) -> str:
    """Stable JSON string used for hashing cache keys."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DatasetCachePaths:
    """Concrete file paths for a single cached item."""
    npz_path: Path
    key_path: Path


class DatasetCache:
    """
    Dataset-scoped cache helper.

    This class is intentionally simple:
    - It does not attempt eviction policies.
    - It does not attempt file locks (HPC users may prefer deterministic pre-warming).
    """

    def __init__(self, *, artifacts_root: Path, dataset_id: str):
        self.artifacts_root = Path(artifacts_root)
        self.dataset_id = str(dataset_id)

    @classmethod
    def from_config(cls, *, config: Dict[str, Any], dataset_id: str) -> "DatasetCache":
        """
        Build a DatasetCache from the global config.

        Uses:
          config["paths"]["artifacts"] or "artifacts"
        """
        paths = config.get("paths") if isinstance(config.get("paths"), dict) else {}
        artifacts_root = Path(paths.get("artifacts", "artifacts"))
        return cls(artifacts_root=artifacts_root, dataset_id=dataset_id)

    # -------------------------------------------------------------------------
    # Directory policy
    # -------------------------------------------------------------------------
    def cache_dir(self) -> Path:
        """
        Dataset-scoped cache directory:
          <artifacts_root>/datasets/<dataset_id>/cache/
        """
        return self.artifacts_root / "datasets" / self.dataset_id / "cache"

    def _paths_for_key(self, cache_key: Dict[str, Any]) -> DatasetCachePaths:
        """
        Map cache_key -> deterministic file paths.

        We hash the key so filenames stay short and filesystem-safe.
        """
        key_json = _stable_json(cache_key)
        h = _sha1_text(key_json)

        cdir = ensure_dir(self.cache_dir())
        npz_path = cdir / f"arrays_{h}.npz"
        key_path = cdir / f"arrays_{h}.key.json"
        return DatasetCachePaths(npz_path=npz_path, key_path=key_path)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def try_load_arrays(self, cache_key: Dict[str, Any]) -> Optional[DatasetArrays]:
        """
        Try to load cached arrays for the given cache_key.

        Returns DatasetArrays if cache exists and is readable; otherwise None.
        """
        paths = self._paths_for_key(cache_key)

        if not paths.npz_path.exists() or not paths.key_path.exists():
            return None

        try:
            with np.load(paths.npz_path, allow_pickle=False) as z:
                arrays = DatasetArrays(
                    x_train=z["x_train"],
                    y_train=z["y_train"],
                    x_val=z["x_val"],
                    y_val=z["y_val"],
                    x_test=z["x_test"],
                    y_test=z["y_test"],
                )
            return arrays
        except Exception:
            return None

    def save_arrays(self, cache_key: Dict[str, Any], arrays: DatasetArrays) -> DatasetCachePaths:
        """
        Save dataset arrays to cache (npz + key.json).

        Notes:
        - This is an overwrite-safe operation: writing npz is atomic enough for most cases
          (but if you want strict atomicity, we can route via gencysynth.utils.io later).
        """
        paths = self._paths_for_key(cache_key)
        ensure_dir(paths.npz_path.parent)

        # Save arrays
        np.savez_compressed(
            paths.npz_path,
            x_train=arrays.x_train,
            y_train=arrays.y_train,
            x_val=arrays.x_val,
            y_val=arrays.y_val,
            x_test=arrays.x_test,
            y_test=arrays.y_test,
        )

        # Save the key (helps debugging and cache audits)
        payload = {"cache_key": cache_key}
        if write_json is not None:
            write_json(paths.key_path, payload, indent=2, sort_keys=True, atomic=True)
        else:
            paths.key_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        return paths