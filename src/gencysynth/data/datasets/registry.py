# src/gencysynth/data/datasets/registry.py
"""
GenCyberSynth — Dataset Registry (Scalable)

Purpose
-------
A dataset registry lets the rest of the codebase remain dataset_agnostic.

Instead of writing:
    if dataset == "ustc": ...
    elif dataset == "cifar": ...
    ...

we do:
    ds = make_dataset_from_config(cfg)
    splits = ds.load_arrays(config=cfg)

This is essential for scalability when you add:
- new datasets
- new raw layouts (NPY, image folders, LMDB, webdataset, etc.)
- per_dataset options (image shape, class names, split policies)

Config contract
---------------
dataset:
  id: "USTC_TFC2016_40x40_gray"          # REQUIRED for scalable artifacts layout
  type: "npy_ustc"                       # REQUIRED (selects loader)
  raw_root: "data/ustc"                  # dataset_specific

We treat dataset_id as *the* stable identity for:
  artifacts/datasets/<dataset_id>/...
  artifacts/runs/<dataset_id>/...
  artifacts/eval/<dataset_id>/...

Design goals
------------
- Clear errors (helps in HPC environments)
- Easy extension: add dataset class + register it
- No heavy dependencies
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type

from gencysynth.data.datasets.base import Dataset, BaseDataset

# Concrete dataset implementations
from gencysynth.data.datasets.npy_ustc import NpyUSTCDataset
from gencysynth.data.datasets.image_folder import ImageFolderDataset


# =============================================================================
# Registry types
# =============================================================================

DatasetFactory = Callable[[str], Dataset]


@dataclass(frozen=True)
class DatasetRegistryEntry:
    """
    One registered dataset loader.

    `factory` takes dataset_id and returns a Dataset instance.
    `description` is used for debugging and CLI help text.
    """
    factory: DatasetFactory
    description: str


# =============================================================================
# Global registry
# =============================================================================

_REGISTRY: Dict[str, DatasetRegistryEntry] = {}


def register_dataset_type(
    dataset_type: str,
    factory: DatasetFactory,
    *,
    description: str = "",
    aliases: Optional[list[str]] = None,
) -> None:
    """
    Register a dataset type string -> factory.

    Example:
        register_dataset_type(
            "npy_ustc",
            lambda dataset_id: NpyUSTCDataset(dataset_id),
            aliases=["ustc_npy", "npy"],
        )

    Notes
    -----
    - dataset_type and aliases are normalized to lowercase.
    - Later registrations overwrite earlier ones (intentional for experiments).
    """
    key = str(dataset_type).strip().lower()
    if not key:
        raise ValueError("dataset_type must be a non_empty string.")

    _REGISTRY[key] = DatasetRegistryEntry(factory=factory, description=description or key)

    for a in aliases or []:
        akey = str(a).strip().lower()
        if akey:
            _REGISTRY[akey] = _REGISTRY[key]


def known_dataset_types() -> Dict[str, str]:
    """
    Return a mapping of known dataset type tokens -> description.
    Useful for CLI help or debugging.
    """
    out: Dict[str, str] = {}
    for k, v in sorted(_REGISTRY.items(), key=lambda kv: kv[0]):
        out[k] = v.description
    return out


# =============================================================================
# Default registrations (core supported dataset formats)
# =============================================================================

def _register_defaults() -> None:
    """
    Register built_in dataset types.

    Keep this list small and stable.
    Additional datasets can be registered by:
    - importing this module and calling register_dataset_type(...) at runtime, OR
    - adding new imports + registrations here.
    """
    register_dataset_type(
        "npy_ustc",
        lambda dataset_id: NpyUSTCDataset(dataset_id),
        description="USTC_style .npy quartet: train_data/train_labels/test_data/test_labels (+ val split)",
        aliases=["ustc_npy", "ustc", "npy"],
    )

    register_dataset_type(
        "image_folder",
        lambda dataset_id: ImageFolderDataset(dataset_id),
        description="ImageFolder: raw_root/{train,val,test}/<class>/*.png|jpg|...",
        aliases=["folder", "images", "img_folder"],
    )


# Ensure defaults registered on import
_register_defaults()


# =============================================================================
# Config parsing + dataset creation
# =============================================================================

def _cfg_dataset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    d = cfg.get("dataset")
    return d if isinstance(d, dict) else {}


def _require_dataset_id(cfg: Dict[str, Any]) -> str:
    """
    dataset_id is mandatory for scalable artifacts layout.
    """
    d = _cfg_dataset(cfg)
    dataset_id = d.get("id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError(
            "Missing required config['dataset']['id']. "
            "This must be a stable identifier, e.g. 'USTC_TFC2016_40x40_gray'."
        )
    return dataset_id.strip()


def _require_dataset_type(cfg: Dict[str, Any]) -> str:
    """
    dataset.type selects the loader.
    """
    d = _cfg_dataset(cfg)
    t = d.get("type")
    if not isinstance(t, str) or not t.strip():
        known = ", ".join(sorted(set(_REGISTRY.keys())))
        raise ValueError(
            "Missing required config['dataset']['type']. "
            f"Known types include: {known}"
        )
    return t.strip().lower()


def make_dataset_from_config(cfg: Dict[str, Any]) -> Dataset:
    """
    Create a Dataset instance from config.

    This is the standard entrypoint used by:
    - orchestration code
    - evaluation split loaders
    - any pipeline that wants dataset_agnostic behavior

    Raises
    ------
    ValueError with clear message if config is invalid.
    """
    dataset_id = _require_dataset_id(cfg)
    dataset_type = _require_dataset_type(cfg)

    entry = _REGISTRY.get(dataset_type)
    if entry is None:
        known = ", ".join(sorted(set(_REGISTRY.keys())))
        raise ValueError(
            f"Unknown dataset type '{dataset_type}'. "
            f"Known types include: {known}"
        )

    ds = entry.factory(dataset_id)

    # Defensive sanity: ensure it looks like a dataset implementation
    if not hasattr(ds, "load_arrays"):
        raise TypeError(f"Dataset factory for '{dataset_type}' did not return a Dataset_like object.")

    return ds


__all__ = [
    "DatasetRegistryEntry",
    "register_dataset_type",
    "known_dataset_types",
    "make_dataset_from_config",
]
