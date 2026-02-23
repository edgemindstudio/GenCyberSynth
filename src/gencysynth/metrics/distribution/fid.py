# src/gencysynth/metrics/distribution/fid.py
"""
Fréchet Inception Distance (FID) metric plugin.

Important note
--------------
Classic FID uses Inception activations. For your current repo goal (end-to-end
sanity + scalable structure), this provides a *feature-agnostic FID*:

- Default feature space is lightweight pixel features (see mmd._default_features)
- Later you can replace the feature function with Inception features without
  changing the metric API/artifact layout.

Math
----
FID = ||mu_r - mu_s||^2 + Tr(Sigma_r + Sigma_s - 2 * sqrtm(Sigma_r * Sigma_s))

We implement sqrtm using an eigen-based method with numerical safeguards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ..types import DatasetMeta, MetricResult, RunMeta
from .mmd import _default_features


def _cov(x: np.ndarray) -> np.ndarray:
    """
    Covariance with rows as samples (N,D). Returns (D,D).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < 2:
        d = x.shape[1]
        return np.eye(d, dtype=np.float64)
    return np.cov(x, rowvar=False)


def _sqrtm_psd(A: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Compute matrix square root for a PSD matrix using eigen decomposition.
    Adds small diagonal jitter for numerical stability.

    Returns sqrt(A) in float64.
    """
    A = np.asarray(A, dtype=np.float64)
    A = (A + A.T) / 2.0
    # jitter to avoid negative eigenvalues from numerical error
    A = A + np.eye(A.shape[0], dtype=np.float64) * float(eps)

    w, V = np.linalg.eigh(A)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T


def frechet_distance(mu1: np.ndarray, cov1: np.ndarray, mu2: np.ndarray, cov2: np.ndarray) -> float:
    """
    FID computation with eigen sqrtm. Handles small numerical issues.
    """
    mu1 = np.asarray(mu1, dtype=np.float64).reshape(-1)
    mu2 = np.asarray(mu2, dtype=np.float64).reshape(-1)
    cov1 = np.asarray(cov1, dtype=np.float64)
    cov2 = np.asarray(cov2, dtype=np.float64)

    diff = mu1 - mu2
    # sqrtm of product
    cov_prod = cov1 @ cov2
    cov_sqrt = _sqrtm_psd(cov_prod)

    fid = float(diff @ diff + np.trace(cov1 + cov2 - 2.0 * cov_sqrt))
    # guard against tiny negative values due to numeric error
    return float(max(fid, 0.0))


@dataclass
class FIDMetric:
    """
    distribution.fid

    Options at cfg.metrics.options.distribution.fid:
      max_samples: 4096
      feature_dim: 256
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
        name = "distribution.fid"
        opts = (((cfg.get("metrics") or {}).get("options") or {}).get(name) or {})

        max_samples = int(opts.get("max_samples", 4096))
        feat_dim = int(opts.get("feature_dim", 256))

        rng = np.random.default_rng(int(run.seed))
        xr = np.asarray(x_real01, dtype=np.float32)
        xs = np.asarray(x_synth01, dtype=np.float32)

        if xr.shape[0] > max_samples:
            xr = xr[rng.choice(xr.shape[0], size=max_samples, replace=False)]
        if xs.shape[0] > max_samples:
            xs = xs[rng.choice(xs.shape[0], size=max_samples, replace=False)]

        fr = _default_features(xr, out_dim=feat_dim)  # (N,D)
        fs = _default_features(xs, out_dim=feat_dim)  # (M,D)

        mu_r = np.mean(fr, axis=0)
        mu_s = np.mean(fs, axis=0)
        cov_r = _cov(fr)
        cov_s = _cov(fs)

        fid = frechet_distance(mu_r, cov_r, mu_s, cov_s)

        return MetricResult(
            name=name,
            value=float(fid),
            details={
                "feature_dim": feat_dim,
                "max_samples": max_samples,
                "n_real_used": int(fr.shape[0]),
                "n_synth_used": int(fs.shape[0]),
                "note": "FID computed on lightweight pixel features (smoke-test).",
            },
            status="ok",
        )