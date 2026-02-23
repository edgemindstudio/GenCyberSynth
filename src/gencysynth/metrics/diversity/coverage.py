# src/gencysynth/metrics/diversity/coverage.py
"""
Coverage proxy metric for synthetic diversity.

Goal
----
Coverage should answer: "How much of the real distribution is represented
by synthetic samples?"

A full coverage metric can be complex. For end-to-end repo sanity and scalable
evaluation, we implement a strong *proxy*:

- Extract lightweight features (pixel subsample, deterministic)
- Compute nearest-neighbor distance from each real sample -> closest synthetic
- Report quantiles + mean distance (lower means better coverage)
- Optionally compute per-class coverage if labels exist

Rule A
------
- No artifact I/O here.
- Deterministic subsampling based on run.seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ..types import DatasetMeta, MetricResult, RunMeta
from ..features import pixel_features


def _nn_min_dists(real_feat: np.ndarray, synth_feat: np.ndarray, chunk: int = 512) -> np.ndarray:
    """
    Compute min Euclidean distance from each real vector to any synth vector.
    Uses chunking to avoid huge memory use:
      real: (N,D), synth:(M,D) -> (N,)
    """
    real_feat = np.asarray(real_feat, dtype=np.float64)
    synth_feat = np.asarray(synth_feat, dtype=np.float64)

    N = real_feat.shape[0]
    out = np.empty((N,), dtype=np.float64)

    # Precompute synth norms for fast distance computations
    s2 = np.sum(synth_feat * synth_feat, axis=1, keepdims=True).T  # (1,M)

    for i in range(0, N, chunk):
        r = real_feat[i : i + chunk]
        r2 = np.sum(r * r, axis=1, keepdims=True)  # (b,1)
        # squared distances: ||r||^2 + ||s||^2 - 2 r@s^T
        d2 = np.maximum(r2 + s2 - 2.0 * (r @ synth_feat.T), 0.0)
        out[i : i + chunk] = np.sqrt(np.min(d2, axis=1))
    return out


def _summarize_dists(d: np.ndarray) -> Dict[str, float]:
    d = np.asarray(d, dtype=np.float64)
    if d.size == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "p95": float("nan")}
    return {
        "mean": float(np.mean(d)),
        "p50": float(np.quantile(d, 0.50)),
        "p90": float(np.quantile(d, 0.90)),
        "p95": float(np.quantile(d, 0.95)),
    }


@dataclass
class CoverageMetric:
    """
    diversity.coverage

    Options (cfg.metrics.options.diversity.coverage):
      max_real: 5000
      max_synth: 5000
      feature_dim: 256
      chunk: 512
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
        name = "diversity.coverage"
        opts = (((cfg.get("metrics") or {}).get("options") or {}).get(name) or {})

        max_real = int(opts.get("max_real", 5000))
        max_synth = int(opts.get("max_synth", 5000))
        feat_dim = int(opts.get("feature_dim", 256))
        chunk = int(opts.get("chunk", 512))
        per_class = bool(opts.get("per_class", False))

        rng = np.random.default_rng(int(run.seed))

        xr = np.asarray(x_real01, dtype=np.float32)
        xs = np.asarray(x_synth01, dtype=np.float32)

        yr = None if y_real is None else np.asarray(y_real)
        ys = None if y_synth is None else np.asarray(y_synth)

        # Deterministic subsampling for scalability
        if xr.shape[0] > max_real:
            idx = rng.choice(xr.shape[0], size=max_real, replace=False)
            xr = xr[idx]
            yr = yr[idx] if yr is not None else None

        if xs.shape[0] > max_synth:
            idx = rng.choice(xs.shape[0], size=max_synth, replace=False)
            xs = xs[idx]
            ys = ys[idx] if ys is not None else None

        # Feature extraction (shared helper in metrics.features)
        fr = pixel_features(xr, out_dim=feat_dim)
        fs = pixel_features(xs, out_dim=feat_dim)

        d = _nn_min_dists(fr, fs, chunk=chunk)
        summary = _summarize_dists(d)

        details = {
            "max_real": max_real,
            "max_synth": max_synth,
            "feature_dim": feat_dim,
            "chunk": chunk,
            "n_real_used": int(fr.shape[0]),
            "n_synth_used": int(fs.shape[0]),
            "distance": summary,
            "note": "Coverage proxy via NN distances on lightweight pixel features (smoke-test).",
        }

        # Optional per-class coverage proxy
        if per_class and (yr is not None) and (ys is not None):
            K = int(dataset.num_classes)
            per = {}
            for k in range(K):
                r_idx = (yr == k)
                s_idx = (ys == k)
                nr = int(np.sum(r_idx))
                ns = int(np.sum(s_idx))
                if nr == 0 or ns == 0:
                    per[str(k)] = {"status": "skipped", "n_real": nr, "n_synth": ns}
                    continue
                frk = fr[r_idx]
                fsk = fs[s_idx]
                dk = _nn_min_dists(frk, fsk, chunk=chunk)
                per[str(k)] = {
                    "n_real": nr,
                    "n_synth": ns,
                    "distance": _summarize_dists(dk),
                }
            details["per_class"] = per

        # Primary scalar: mean NN distance (lower is better)
        return MetricResult(
            name=name,
            value=float(summary["mean"]),
            details=details,
            status="ok",
        )