# src/gencysynth/metrics/privacy/nn_distance.py
"""
Nearest-neighbor distance (privacy proxy) in feature space.

Motivation
----------
A simple privacy risk proxy: if synthetic samples are extremely close to real samples
in a learned feature space, there may be memorization or training data leakage.

This metric computes, for each synthetic feature vector:
  d_i = min_j || f_synth[i] - f_real[j] ||_2

Then summarizes:
  - mean, median
  - p01, p05, p10, p25, p75, p90, p95, p99
  - min (closest neighbor distance)

Rule A
------
- Pure computation only.
- No file I/O; feature arrays must be injected by the orchestrator (or caller).
- Works across datasets and models as long as you pass feature matrices.

Expected inputs (injected)
--------------------------
The caller must provide feature matrices in cfg under either:
  cfg["privacy"]["features"] = {"real": (Nr,D), "synth": (Ns,D)}
or:
  cfg["extras"]["features"]  = {"real": (Nr,D), "synth": (Ns,D)}

Options (cfg.metrics.options.privacy.nn_distance)
-------------------------------------------------
- metric: "l2" (only L2 supported currently)
- batch_size: 2048     # batch synthetic rows to control memory
- real_block: 8192     # block real rows to control memory
- percentiles: [1,5,10,25,50,75,90,95,99]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..types import DatasetMeta, MetricResult, RunMeta


def _as_2d_float(x: np.ndarray, name: str) -> np.ndarray:
    """Coerce to float32 2D array (N,D) with basic validation."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D (N,D); got shape {x.shape}")
    if x.shape[0] <= 0 or x.shape[1] <= 0:
        raise ValueError(f"{name} must be non-empty; got shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains non-finite values (NaN/Inf).")
    return x


def _min_l2_distance_per_row(
    synth: np.ndarray,
    real: np.ndarray,
    *,
    synth_batch: int = 2048,
    real_block: int = 8192,
) -> np.ndarray:
    """
    Compute per-synth minimal L2 distance to any real vector.

    Memory-safe implementation:
      - iterate over synth in batches
      - iterate over real in blocks
      - keep the running min distance per synth row

    Returns
    -------
    dmin : (Ns,) float32 distances
    """
    Ns, D = synth.shape
    Nr, Dr = real.shape
    if Dr != D:
        raise ValueError(f"Feature dims mismatch: synth D={D}, real D={Dr}")

    synth_batch = int(max(1, synth_batch))
    real_block = int(max(1, real_block))

    dmin_all = np.empty((Ns,), dtype=np.float32)

    # Precompute squared norms for real blocks if desired. Here we do block-wise.
    for i0 in range(0, Ns, synth_batch):
        i1 = min(Ns, i0 + synth_batch)
        S = synth[i0:i1]  # (B,D)
        # Start with +inf
        dmin = np.full((S.shape[0],), np.inf, dtype=np.float32)

        # For each real block, compute squared distances efficiently:
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
        S_norm = np.sum(S * S, axis=1, keepdims=True).astype(np.float32)  # (B,1)

        for j0 in range(0, Nr, real_block):
            j1 = min(Nr, j0 + real_block)
            R = real[j0:j1]  # (R,D)
            R_norm = np.sum(R * R, axis=1, keepdims=True).astype(np.float32).T  # (1,R)

            # (B,R) squared distances
            # Using float32 matmul; safe + fast enough.
            dots = S @ R.T  # (B,R)
            d2 = S_norm + R_norm - 2.0 * dots
            # numerical floor (tiny negatives due to float error)
            d2 = np.maximum(d2, 0.0, dtype=np.float32)

            # update min
            dmin = np.minimum(dmin, np.min(d2, axis=1).astype(np.float32))

        dmin_all[i0:i1] = np.sqrt(dmin, dtype=np.float32)

    return dmin_all


def _summarize_distances(d: np.ndarray, percentiles: Sequence[int]) -> Dict:
    """Produce a stable summary dict for logging and schema-friendly results."""
    d = np.asarray(d, dtype=np.float32)
    if d.size == 0:
        return {"n": 0}

    pct = [int(p) for p in percentiles]
    pct = [p for p in pct if 0 <= p <= 100]
    pct = sorted(set(pct))

    out = {
        "n": int(d.size),
        "min": float(np.min(d)),
        "mean": float(np.mean(d)),
        "median": float(np.median(d)),
        "max": float(np.max(d)),
    }
    if pct:
        vals = np.percentile(d, pct).astype(np.float32)
        out["percentiles"] = {str(p): float(v) for p, v in zip(pct, vals)}
    return out


@dataclass
class NNDistMetric:
    """
    privacy.nn_distance

    Required injected payload
    -------------------------
    - real:  (Nr,D) float feature matrix
    - synth: (Ns,D) float feature matrix

    This metric does not define how features are computed; it only consumes them.
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
        name = "privacy.nn_distance"
        opts = (((cfg.get("metrics") or {}).get("options") or {}).get(name) or {})

        metric = str(opts.get("metric", "l2")).lower()
        if metric != "l2":
            return MetricResult(
                name=name,
                value=float("nan"),
                status="skipped",
                details={"reason": f"Unsupported metric '{metric}'. Only 'l2' is supported."},
            )

        synth_batch = int(opts.get("batch_size", 2048))
        real_block = int(opts.get("real_block", 8192))
        percentiles = opts.get("percentiles", [1, 5, 10, 25, 50, 75, 90, 95, 99])

        # ---- Extract injected features (Rule A: we don't load from disk) ----
        payload = None
        if isinstance(cfg.get("privacy"), dict) and isinstance(cfg["privacy"].get("features"), dict):
            payload = cfg["privacy"]["features"]
        elif isinstance(cfg.get("extras"), dict) and isinstance(cfg["extras"].get("features"), dict):
            payload = cfg["extras"]["features"]

        if payload is None:
            return MetricResult(
                name=name,
                value=float("nan"),
                status="skipped",
                details={
                    "reason": (
                        "Missing feature payload. Provide cfg['privacy']['features'] "
                        "or cfg['extras']['features'] with keys {'real','synth'}."
                    )
                },
            )

        if "real" not in payload or "synth" not in payload:
            return MetricResult(
                name=name,
                value=float("nan"),
                status="skipped",
                details={"reason": "Feature payload must contain keys 'real' and 'synth'."},
            )

        try:
            real_f = _as_2d_float(payload["real"], "real_features")
            synth_f = _as_2d_float(payload["synth"], "synth_features")
        except Exception as e:
            return MetricResult(
                name=name,
                value=float("nan"),
                status="error",
                details={"reason": f"Invalid feature arrays: {e}"},
            )

        # ---- Compute nearest-neighbor distances: synth -> real ----
        dmin = _min_l2_distance_per_row(
            synth=synth_f,
            real=real_f,
            synth_batch=synth_batch,
            real_block=real_block,
        )

        summary = _summarize_distances(dmin, percentiles=percentiles)

        # Primary scalar: mean NN distance (lower => closer/more risk)
        value = float(summary.get("mean", float("nan")))

        details = {
            "direction": "synth_to_real",
            "metric": "l2",
            "synth_batch": int(synth_batch),
            "real_block": int(real_block),
            "summary": summary,
        }

        return MetricResult(
            name=name,
            value=value,
            status="ok",
            details=details,
        )