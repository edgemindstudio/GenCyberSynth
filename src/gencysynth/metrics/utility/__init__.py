# src/gencysynth/utility/__init__.py
"""
Utility tools used across GenCyberSynth pipelines.

Intent
------
This package contains small, reusable helpers that are not model-family specific.
They are designed to support:
- smoke tests (fast end-to-end runs)
- utility/quality checks (e.g., train a small classifier to measure usefulness)
- Rule A artifacts behavior (predictable, scalable paths, structured outputs)

Public API
----------
- train_classifier: lightweight classifier training + checkpointing + summary
- metrics: evaluation metrics helpers (accuracy, balanced accuracy, F1, confusion)
"""

from .train_classifier import train_classifier, main as train_classifier_main
from .metrics import (
    confusion_matrix,
    accuracy,
    balanced_accuracy,
    f1_per_class,
    macro_f1,
    classification_report_dict,
)

__all__ = [
    "train_classifier",
    "train_classifier_main",
    "confusion_matrix",
    "accuracy",
    "balanced_accuracy",
    "f1_per_class",
    "macro_f1",
    "classification_report_dict",
]