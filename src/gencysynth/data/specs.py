# src/gencysynth/data/specs.py
"""
gencysynth.data.specs
====================

A small dataset contract for scalable, multi-dataset loading.

Why this file exists
--------------------
As GenCyberSynth grows, you will support multiple datasets with different:
- root folders
- file naming conventions
- shapes / channels
- number of classes
- train/val/test composition rules

DatasetSpec is the minimal structure that keeps loaders clean and scalable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class DatasetSpec:
    """
    Describes how to load a dataset from a root directory.

    Attributes
    ----------
    dataset_id:
        Stable identifier used across artifacts paths (e.g. "USTC-TFC2016_40x40_gray").
    root_dir:
        Where the dataset files live on disk (can be absolute or relative).
    img_shape:
        (H,W,C) expected for training/evaluation.
    num_classes:
        Number of classes (K).
    train_data_fname, train_labels_fname, test_data_fname, test_labels_fname:
        Standard quartet for .npy classification datasets.
    val_fraction:
        If dataset provides only train/test, split test into val+test by this fraction.
    """
    dataset_id: str
    root_dir: str

    img_shape: Tuple[int, int, int]
    num_classes: int

    train_data_fname: str = "train_data.npy"
    train_labels_fname: str = "train_labels.npy"
    test_data_fname: str = "test_data.npy"
    test_labels_fname: str = "test_labels.npy"

    val_fraction: float = 0.5

    def __post_init__(self) -> None:
        # basic safety checks
        if not self.dataset_id:
            raise ValueError("DatasetSpec.dataset_id must be a non-empty string.")
        if not self.root_dir:
            raise ValueError("DatasetSpec.root_dir must be a non-empty string.")
        if self.num_classes <= 0:
            raise ValueError("DatasetSpec.num_classes must be > 0.")
