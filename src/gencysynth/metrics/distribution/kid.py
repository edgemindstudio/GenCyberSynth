# src/gencysynth/metrics/distribution/kid.py
"""
Kernel Inception Distance (KID) metric plugin.

Important note
--------------
Classic KID uses Inception features + polynomial kernel MMD^2.
In this repo's "smoke test / scalable structure" stage, we implement KID on a
lightweight feature space (pixel features by default), which is:
- deterministic
- dependency-minimal
- useful to validate end-to-end pipeline correctness

When you add Inception features later, swap the feature function inside mmd.py
or add a "feature extractor" module that feeds (N,D) features here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ..types import DatasetMeta, MetricResult, RunMeta
from .mmd import _default_features, mmd2_unbiased, poly_kernel


@dataclass
class KIDMetric:
    """
    distribution.kid

    Options at cfg.metrics.options.distribution.kid:
      max_samples: 4096
      feature_dim: 256
      degree: 3
      gamma: 1.0
      coef0: 1.0
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
        name = "distribution.kid"
        opts = (((cfg.get("metrics") or {}).get("options") or {}).get(name) or {})

        max_samples = int(opts.get("max_samples", 4096))
        feat_dim = int(opts.get("feature_dim", 256))
        degree = int(opts.get("degree", 3))
        gamma = float(opts.get("gamma", 1.0))
        coef0 = float(opts.get("coef0", 1.0))

        rng = np.random.default_rng(int(run.seed))
        xr = np.asarray(x_real01, dtype=np.float32)
        xs = np.asarray(x_synth01, dtype=np.float32)

        if xr.shape[0] > max_samples:
            xr = xr[rng.choice(xr.shape[0], size=max_samples, replace=False)]
        if xs.shape[0] > max_samples:
            xs = xs[rng.choice(xs.shape[0], size=max_samples, replace=False)]

        fr = _default_features(xr, out_dim=feat_dim)
        fs = _default_features(xs, out_dim=feat_dim)

        Kxx = poly_kernel(fr, fr, degree=degree, gamma=gamma, coef0=coef0)
        Kyy = poly_kernel(fs, fs, degree=degree, gamma=gamma, coef0=coef0)
        Kxy = poly_kernel(fr, fs, degree=degree, gamma=gamma, coef0=coef0)

        kid = mmd2_unbiased(Kxx, Kyy, Kxy)

        return MetricResult(
            name=name,
            value=float(kid),
            details={
                "kernel": "poly",
                "degree": degree,
                "gamma": gamma,
                "coef0": coef0,
                "max_samples": max_samples,
                "feature_dim": feat_dim,
                "n_real_used": int(fr.shape[0]),
                "n_synth_used": int(fs.shape[0]),
                "note": "KID computed on lightweight pixel features (smoke-test).",
            },
            status="ok",
        )