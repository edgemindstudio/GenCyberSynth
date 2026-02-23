# src/gencysynth/adapters/datasets/registry.py
"""
Dataset adapter registry.

We resolve dataset adapters by dataset_id, e.g.:
  - "ustc_tfc2016_npy"
  - "image_folder"

This mirrors the model adapter registry pattern and keeps orchestration generic.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from .base import DatasetAdapter
from gencysynth.adapters.errors import AdapterNotFoundError


DatasetAdapterFactory = Callable[[], DatasetAdapter]
_REGISTRY: Dict[str, DatasetAdapterFactory] = {}


def register_dataset_adapter(dataset_id: str, factory: DatasetAdapterFactory) -> None:
    """
    Register a dataset adapter factory.

    Overwrites are allowed to support local development and experimentation.
    """
    _REGISTRY[str(dataset_id)] = factory


def resolve_dataset_adapter(dataset_id: str) -> DatasetAdapter:
    """
    Instantiate and return the dataset adapter for a given dataset_id.
    """
    key = str(dataset_id)
    if key not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY.keys())) if _REGISTRY else "(none)"
        raise AdapterNotFoundError(
            f"No dataset adapter registered for dataset_id='{key}'. Known: {known}. "
            "Did you import the dataset registry module (e.g., gencysynth.adapters.datasets.npy_ustc)?"
        )
    return _REGISTRY[key]()


def list_dataset_adapters() -> List[str]:
    """Return registered dataset_ids (sorted)."""
    return sorted(_REGISTRY.keys())
