# src/gencysynth/adapters/datasets/__init__.py
"""
Dataset adapters (front-door for model adapters).

Why this exists
---------------
We support multiple datasets (Rule A), and each dataset may have different:
- storage format (npy files, image folders, webdataset, etc.)
- label encoding
- split conventions

Model adapters should NOT care about that. They should receive:
  - standardized splits (train/val/test)
  - standardized image shape
  - standardized label encodings (int + one-hot)
  - standardized ranges ([0,1] for eval/metrics; optionally [-1,1] for tanh models)

Public API
----------
- DatasetAdapter protocol + DatasetSplits dataclasses
- register_dataset_adapter / resolve_dataset_adapter
- built-in examples:
    - npy_ustc (USTC-TFC2016 npy format)
    - image_folder (generic class-folder dataset)
"""

from .splits import DatasetSplits, Split, SplitArrays
from .base import DatasetAdapter, DatasetSpec
from .registry import register_dataset_adapter, resolve_dataset_adapter, list_dataset_adapters

__all__ = [
    "DatasetSplits",
    "Split",
    "SplitArrays",
    "DatasetAdapter",
    "DatasetSpec",
    "register_dataset_adapter",
    "resolve_dataset_adapter",
    "list_dataset_adapters",
]
