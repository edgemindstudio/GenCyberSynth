# src/gencysynth/metrics/calibration/brier.py
"""
Brier score (multi-class).

Definition
----------
For multi-class classification with probabilities p_i over K classes,
and one-hot true vector y_i:

  Brier = mean_i sum_k (p_i[k] - y_i[k])^2

Lower is better.

Rule A
------
- Pure computation.
- Prediction payload must be injected by the orchestrator (no artifact reads here).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ..types import DatasetMeta, MetricResult, RunMeta


def _labels_to_onehot(y: np.ndarray, num_classes: int) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == num_classes:
        return y.astype(np.float32, copy=False)
    if y.ndim == 1:
        out = np.zeros((y.shape[0], num_classes), dtype=np.float32)
        out[np.arange(y.shape[0]), y.astype(np.int64)] = 1.0
        return out
    raise ValueError(f"Expected labels (N,) or (N,{num_classes}); got {y.shape}")


def _validate_probs(p: np.ndarray, num_classes: int) -> np.ndarray:
    p = np.asarray(p, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] != num_classes:
        raise ValueError(f"Expected probs (N,{num_classes}); got {p.shape}")
    p = np.clip(p, 0.0, 1.0)
    s = np.sum(p, axis=1, keepdims=True)
    p = np.where(s > 0, p / s, np.full_like(p, 1.0 / float(num_classes)))
    return p


@dataclass
class BrierMetric:
    """
    calibration.brier

    Required input (injected by caller)
    -----------------------------------
    cfg['calibration']['probs'] or cfg['extras']['probs'] with:
      - y_true
      - p_pred

    Options (cfg.metrics.options.calibration.brier):
      reduce: "mean"  # reserved for future (e.g., per-class)
      per_class: false
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
        name = "calibration.brier"
        opts = (((cfg.get("metrics") or {}).get("options") or {}).get(name) or {})

        per_class = bool(opts.get("per_class", False))

        # Extract probs payload (framework must provide; this metric won't load from disk)
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
        y1h = _labels_to_onehot(np.asarray(y_true), K)
        p = _validate_probs(np.asarray(p_pred), K)

        # Brier per example then mean
        per_ex = np.sum((p - y1h) ** 2, axis=1).astype(np.float64)
        score = float(np.mean(per_ex)) if per_ex.size else float("nan")

        details = {
            "n": int(per_ex.size),
            "num_classes": int(K),
            "mean": float(score),
        }

        # Optional per-class brier (useful for imbalanced datasets)
        if per_class:
            y_int = np.argmax(y1h, axis=1)
            byc = {}
            for k in range(K):
                mk = (y_int == k)
                nk = int(np.sum(mk))
                if nk == 0:
                    byc[str(k)] = {"status": "skipped", "n": 0}
                    continue
                byc[str(k)] = {"n": nk, "mean": float(np.mean(per_ex[mk]))}
            details["per_class"] = byc

        return MetricResult(
            name=name,
            value=float(score),
            status="ok",
            details=details,
        )