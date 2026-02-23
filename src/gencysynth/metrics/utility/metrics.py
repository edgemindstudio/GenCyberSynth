# src/gencysynth/utility/metrics.py
"""
Minimal, dependency-light classification metrics.

Design goals
------------
- NumPy-only (no sklearn dependency)
- Deterministic outputs suitable for writing into run summaries/manifests
- Safe for small K (e.g., 9 malware classes) and smoke tests

All functions accept integer labels:
- y_true: shape (N,)
- y_pred: shape (N,)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


def _as_int_1d(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 1:
        y = y.reshape(-1)
    return y.astype(np.int64, copy=False)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Compute confusion matrix C where C[i, j] counts true=i predicted=j.
    """
    y_true = _as_int_1d(y_true)
    y_pred = _as_int_1d(y_pred)
    K = int(num_classes)

    cm = np.zeros((K, K), dtype=np.int64)
    mask = (y_true >= 0) & (y_true < K) & (y_pred >= 0) & (y_pred < K)
    yt = y_true[mask]
    yp = y_pred[mask]
    for t, p in zip(yt, yp):
        cm[int(t), int(p)] += 1
    return cm


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = _as_int_1d(y_true)
    y_pred = _as_int_1d(y_pred)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(y_true == y_pred))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """
    Mean recall across classes (ignores classes with zero support).
    """
    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.diag(cm) / np.sum(cm, axis=1)
    recall = recall[np.isfinite(recall)]
    if recall.size == 0:
        return float("nan")
    return float(np.mean(recall))


def f1_per_class(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Per-class F1 score. Returns float32 array of shape (K,) with NaN for undefined.
    """
    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes).astype(np.float64)
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2.0 * precision * recall / (precision + recall)

    # Leave undefined values as NaN (e.g., no true samples and no predicted samples for a class)
    return f1.astype(np.float32)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    f1 = f1_per_class(y_true, y_pred, num_classes=num_classes)
    f1 = f1[np.isfinite(f1)]
    if f1.size == 0:
        return float("nan")
    return float(np.mean(f1))


def classification_report_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    label_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compact dict report safe to store in run summaries.
    """
    K = int(num_classes)
    cm = confusion_matrix(y_true, y_pred, num_classes=K)
    f1 = f1_per_class(y_true, y_pred, num_classes=K)

    report = {
        "num_classes": K,
        "accuracy": accuracy(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred, num_classes=K),
        "macro_f1": macro_f1(y_true, y_pred, num_classes=K),
        "per_class_f1": {str(i): (None if not np.isfinite(f1[i]) else float(f1[i])) for i in range(K)},
        "confusion_matrix": cm.tolist(),
    }

    if label_names is not None and len(label_names) == K:
        report["label_names"] = list(label_names)

    return report
