# src/gencysynth/metrics/sanity/__init__.py
"""
Sanity metrics for quick end_to_end validation.

These checks are designed to be:
- fast (safe for smoke tests / <5_epoch runs)
- framework_agnostic (NumPy only)
- model/dataset_agnostic
- Rule A friendly (returns structured dicts; no file I/O)

Public API
----------
- basic_stats: basic descriptive statistics (global + optional per_class)
- check_images: shape/dtype/range/finite checks for image arrays
- check_labels: shape/type/range checks for label arrays
- check_pair:   consistency checks for (real, synth) pairs
"""

from .basic_stats import basic_stats
from .shape_checks import check_images, check_labels, check_pair

__all__ = [
    "basic_stats",
    "check_images",
    "check_labels",
    "check_pair",
]