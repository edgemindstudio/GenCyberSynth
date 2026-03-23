# src/gencysynth/metrics/distribution/mmd.py
"""
Maximum Mean Discrepancy (MMD) utilities + an MMD metric plugin.

Why MMD matters
---------------
MMD measures how different two distributions are using kernel embeddings.
Here we provide:
- a small, dependency_minimal implementation (NumPy only)
- kernels: RBF and polynomial
- unbiased MMD^2 estimator (common for KID and for generic MMD checks)

Rule A
------
This file never writes artifacts directly. It returns MetricResult only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ..types import DatasetMeta, MetricResult, RunMeta


def _pairwise_sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute squared Euclidean distances between all pairs in x and y.
    x: (N,D), y: (M,D) -> (N,M)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x2 = np.sum(x * x, axis=1, keepdims=True)  # (N,1)
    y2 = np.sum(y * y, axis=1, keepdims=True).T  # (1,M)
    return np.maximum(x2 + y2 - 2.0 * (x @ y.T), 0.0)


def rbf_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
    """
    RBF kernel k(x,y)=exp(-||x_y||^2/(2*sigma^2)).
    """
    d2 = _pairwise_sq_dists(x, y)
    s2 = float(sigma) ** 2
    return np.exp(-d2 / (2.0 * s2 + 1e_12))


def poly_kernel(x: np.ndarray, y: np.ndarray, degree: int = 3, gamma: float = 1.0, coef0: float = 1.0) -> np.ndarray:
    """
    Polynomial kernel k(x,y)=(gamma * x·y + coef0)^degree.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return (float(gamma) * (x @ y.T) + float(coef0)) ** int(degree)


def mmd2_unbiased(Kxx: np.ndarray, Kyy: np.ndarray, Kxy: np.ndarray) -> float:
    """
    Unbiased estimate of MMD^2.

    Kxx: (N,N), Kyy:(M,M), Kxy:(N,M)
    """
    Kxx = np.asarray(Kxx, dtype=np.float64)
    Kyy = np.asarray(Kyy, dtype=np.float64)
    Kxy = np.asarray(Kxy, dtype=np.float64)

    n = Kxx.shape[0]
    m = Kyy.shape[0]
    if n < 2 or m < 2:
        return float("nan")

    # Remove diagonal terms for unbiased estimator
    sum_xx = (np.sum(Kxx) - np.trace(Kxx)) / (n * (n - 1))
    sum_yy = (np.sum(Kyy) - np.trace(Kyy)) / (m * (m - 1))
    sum_xy = np.sum(Kxy) / (n * m)
    return float(sum_xx + sum_yy - 2.0 * sum_xy)


def median_heuristic_sigma(x: np.ndarray, max_pairs: int = 2000, rng: Optional[np.random.Generator] = None) -> float:
    """
    RBF sigma via median heuristic on a sample of pairwise distances.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n < 2:
        return 1.0

    rng = rng or np.random.default_rng(0)
    idx = rng.choice(n, size=min(n, max_pairs), replace=False)
    xs = x[idx]

    d2 = _pairwise_sq_dists(xs, xs)
    # take upper triangle without diagonal
    triu = d2[np.triu_indices(d2.shape[0], k=1)]
    med = np.median(triu) if triu.size else 1.0
    sigma = float(np.sqrt(max(med, 1e_12)))
    return sigma


def _default_features(x01: np.ndarray, out_dim: int = 256) -> np.ndarray:
    """
    Default feature map used for distribution metrics when no deep features exist.

    Strategy:
    - flatten pixels
    - optionally subsample dimensions for speed/stability

    This is a *sanity* feature space (good for smoke tests), not a replacement for Inception features.
    """
    x = np.asarray(x01, dtype=np.float32)
    n = x.shape[0]
    flat = x.reshape(n, -1).astype(np.float32, copy=False)

    d = flat.shape[1]
    if out_dim is None or out_dim <= 0 or out_dim >= d:
        return flat.astype(np.float64)

    # Deterministic stride_based subsample (no randomness; stable across runs)
    step = max(1, d // int(out_dim))
    feat = flat[:, ::step][:, : int(out_dim)]
    return feat.astype(np.float64)


@dataclass
class MMDMetric:
    """
    Generic MMD^2 metric plugin.

    Registry name (recommended): distribution.mmd_rbf
    Config (cfg.metrics.options.distribution.mmd_rbf):
      max_samples: 4096
      feature_dim: 256
      kernel:
        - for rbf: sigma="median" or numeric
        - for poly: degree, gamma, coef0
    """
    kernel: str = "rbf"  # {"rbf","poly"}

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
        name = "distribution.mmd_rbf" if self.kernel == "rbf" else "distribution.mmd_poly"

        # Read metric_specific options (Rule A: from cfg)
        opts = (((cfg.get("metrics") or {}).get("options") or {}).get(name) or {})
        max_samples = int(opts.get("max_samples", 4096))
        feat_dim = int(opts.get("feature_dim", 256))

        # Subsample for speed (deterministic by seed)
        rng = np.random.default_rng(int(run.seed))
        xr = np.asarray(x_real01, dtype=np.float32)
        xs = np.asarray(x_synth01, dtype=np.float32)

        if xr.shape[0] > max_samples:
            xr = xr[rng.choice(xr.shape[0], size=max_samples, replace=False)]
        if xs.shape[0] > max_samples:
            xs = xs[rng.choice(xs.shape[0], size=max_samples, replace=False)]

        # Feature mapping
        fr = _default_features(xr, out_dim=feat_dim)
        fs = _default_features(xs, out_dim=feat_dim)

        if self.kernel == "rbf":
            sigma_opt = opts.get("sigma", "median")
            if isinstance(sigma_opt, str) and sigma_opt.lower() == "median":
                sigma = median_heuristic_sigma(np.vstack([fr, fs]), rng=rng)
            else:
                sigma = float(sigma_opt)
            Kxx = rbf_kernel(fr, fr, sigma=sigma)
            Kyy = rbf_kernel(fs, fs, sigma=sigma)
            Kxy = rbf_kernel(fr, fs, sigma=sigma)

            val = mmd2_unbiased(Kxx, Kyy, Kxy)
            return MetricResult(
                name=name,
                value=float(val),
                details={
                    "kernel": "rbf",
                    "sigma": float(sigma),
                    "max_samples": max_samples,
                    "feature_dim": feat_dim,
                    "n_real_used": int(fr.shape[0]),
                    "n_synth_used": int(fs.shape[0]),
                },
                status="ok",
            )

        # Polynomial kernel variant (used by KID; also exposed here if you want it)
        degree = int(opts.get("degree", 3))
        gamma = float(opts.get("gamma", 1.0))
        coef0 = float(opts.get("coef0", 1.0))
        Kxx = poly_kernel(fr, fr, degree=degree, gamma=gamma, coef0=coef0)
        Kyy = poly_kernel(fs, fs, degree=degree, gamma=gamma, coef0=coef0)
        Kxy = poly_kernel(fr, fs, degree=degree, gamma=gamma, coef0=coef0)

        val = mmd2_unbiased(Kxx, Kyy, Kxy)
        return MetricResult(
            name="distribution.mmd_poly",
            value=float(val),
            details={
                "kernel": "poly",
                "degree": degree,
                "gamma": gamma,
                "coef0": coef0,
                "max_samples": max_samples,
                "feature_dim": feat_dim,
                "n_real_used": int(fr.shape[0]),
                "n_synth_used": int(fs.shape[0]),
            },
            status="ok",
        )