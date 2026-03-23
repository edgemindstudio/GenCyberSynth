# src/gencysynth/metrics/calibration/ece.py
"""
Expected Calibration Error (ECE).

ECE overview
------------
For each sample:
  - confidence = max predicted probability
  - accuracy   = 1 if predicted class == true class else 0

Then bin samples by confidence and compute:
  ECE = sum_b (n_b / N) * |acc_b - conf_b|

This implementation supports:
- multi_class predicted probabilities (N, K)
- integer labels (N,) or one_hot labels (N, K)
- optional adaptive binning (equal_frequency bins) or fixed bins

Rule A
------
- Pure computation only (no I/O).
- Works with any dataset; does not assume specific label names or paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ..types import DatasetMeta, MetricResult, RunMeta


def _labels_to_int(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Normalize labels to integer class IDs.
    Accepts:
      - (N,) integer labels
      - (N,K) one_hot labels
    """
    y = np.asarray(y)
    if y.ndim == 1:
        return y.astype(np.int64, copy=False)
    if y.ndim == 2 and y.shape[1] == num_classes:
        return np.argmax(y, axis=1).astype(np.int64, copy=False)
    raise ValueError(f"Expected labels (N,) or (N,{num_classes}); got {y.shape}")


def _validate_probs(p: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Ensure probabilities are float32 and shape (N,K). We also defensively clip.
    """
    p = np.asarray(p, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] != num_classes:
        raise ValueError(f"Expected probs (N,{num_classes}); got {p.shape}")
    # Small numeric issues are common; clip and renormalize lightly
    p = np.clip(p, 0.0, 1.0)
    s = np.sum(p, axis=1, keepdims=True)
    # Avoid division by zero: if a row is all zeros, spread uniformly
    p = np.where(s > 0, p / s, np.full_like(p, 1.0 / float(num_classes)))
    return p


def _ece_fixed_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int) -> Tuple[float, Dict]:
    """
    ECE with fixed bin edges in [0,1].
    """
    n_bins = int(max(1, n_bins))
    edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)

    ece = 0.0
    bins = []
    N = float(conf.shape[0])

    for b in range(n_bins):
        lo, hi = float(edges[b]), float(edges[b + 1])
        # Include right edge for last bin
        if b == n_bins - 1:
            m = (conf >= lo) & (conf <= hi)
        else:
            m = (conf >= lo) & (conf < hi)

        nb = int(np.sum(m))
        if nb == 0:
            bins.append({"bin": b, "count": 0, "acc": None, "conf": None, "gap": None, "range": [lo, hi]})
            continue

        acc_b = float(np.mean(correct[m]))
        conf_b = float(np.mean(conf[m]))
        gap = abs(acc_b - conf_b)
        ece += (nb / N) * gap
        bins.append({"bin": b, "count": nb, "acc": acc_b, "conf": conf_b, "gap": float(gap), "range": [lo, hi]})

    details = {"scheme": "fixed", "n_bins": int(n_bins), "bins": bins}
    return float(ece), details


def _ece_adaptive_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int) -> Tuple[float, Dict]:
    """
    ECE with equal_frequency (quantile) bins.
    """
    n_bins = int(max(1, n_bins))
    N = conf.shape[0]
    order = np.argsort(conf)
    conf_s = conf[order]
    corr_s = correct[order]

    # Compute split indices for equal sized bins
    splits = np.linspace(0, N, n_bins + 1).astype(np.int64)

    ece = 0.0
    bins = []
    for b in range(n_bins):
        a, z = int(splits[b]), int(splits[b + 1])
        if z <= a:
            bins.append({"bin": b, "count": 0, "acc": None, "conf": None, "gap": None})
            continue

        cbin = conf_s[a:z]
        obin = corr_s[a:z]
        nb = int(z - a)

        acc_b = float(np.mean(obin))
        conf_b = float(np.mean(cbin))
        gap = abs(acc_b - conf_b)
        ece += (nb / float(N)) * gap

        lo = float(cbin[0])
        hi = float(cbin[-1])
        bins.append({"bin": b, "count": nb, "acc": acc_b, "conf": conf_b, "gap": float(gap), "range": [lo, hi]})

    details = {"scheme": "adaptive", "n_bins": int(n_bins), "bins": bins}
    return float(ece), details


@dataclass
class ECEMetric:
    """
    calibration.ece

    Expected inputs
    ---------------
    cfg must provide predicted probabilities under one of these keys:
      - cfg["calibration"]["probs"] (dict with "y_true" and "p_pred")
      - OR cfg["extras"]["probs"]   (same idea)
    We keep this flexible because teams wire predictions differently.

    Required fields
    ---------------
    - y_true: labels (N,) or (N,K)
    - p_pred: probs  (N,K)

    Options (cfg.metrics.options.calibration.ece):
      n_bins: 15
      scheme: "fixed" | "adaptive"
      split:  "real" | "synth" | "all"   # for logging/labeling only; computation uses provided y/p

    Notes
    -----
    - This metric is agnostic to where predictions come from; it only computes.
    """

    def __call__(
        self,
        *,
        x_real01: np.ndarray,
        y_real: Optional[np.ndarray],
        x_synth01: np.ndarray,
        y_synth: Optional[np.ndarray],
        dataset: DatasetMeta,
        run: RunMeta,
        cfg: Dict,
    ) -> MetricResult:
        name = "calibration.ece"
        opts = (((cfg.get("metrics") or {}).get("options") or {}).get(name) or {})

        n_bins = int(opts.get("n_bins", 15))
        scheme = str(opts.get("scheme", "fixed")).lower()

        # ---- Extract prediction payload (framework_provided) ----
        # We do not assume artifact paths; caller must inject these into cfg.
        payload = None
        if isinstance(cfg.get("calibration"), dict) and isinstance(cfg["calibration"].get("probs"), dict):
            payload = cfg["calibration"]["probs"]
        elif isinstance(cfg.get("extras"), dict) and isinstance(cfg["extras"].get("probs"), dict):
            payload = cfg["extras"]["probs"]

        if payload is None:
            return MetricResult(
                name=name,
                value=float("nan"),
                status="skipped",
                details={
                    "reason": "Missing predicted probabilities. Provide cfg['calibration']['probs'] or cfg['extras']['probs'] with keys: y_true, p_pred.",
                },
            )

        y_true = payload.get("y_true", None)
        p_pred = payload.get("p_pred", None)
        if y_true is None or p_pred is None:
            return MetricResult(
                name=name,
                value=float("nan"),
                status="skipped",
                details={"reason": "Prediction payload missing y_true or p_pred."},
            )

        K = int(dataset.num_classes)
        y_int = _labels_to_int(np.asarray(y_true), K)
        p = _validate_probs(np.asarray(p_pred), K)

        # ---- Compute confidences and correctness ----
        pred = np.argmax(p, axis=1)
        conf = np.max(p, axis=1)
        correct = (pred == y_int).astype(np.float32)

        # ---- ECE (binning) ----
        if scheme == "adaptive":
            ece, bin_details = _ece_adaptive_bins(conf, correct, n_bins=n_bins)
        else:
            ece, bin_details = _ece_fixed_bins(conf, correct, n_bins=n_bins)

        details = {
            "n": int(conf.shape[0]),
            "num_classes": int(K),
            "binning": bin_details,
            "summary": {
                "mean_confidence": float(np.mean(conf)) if conf.size else float("nan"),
                "accuracy": float(np.mean(correct)) if correct.size else float("nan"),
            },
        }

        return MetricResult(
            name=name,
            value=float(ece),
            status="ok",
            details=details,
        )