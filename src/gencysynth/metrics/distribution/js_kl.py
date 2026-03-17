# src/gencysynth/metrics/distribution/js_kl.py
"""
JS / KL divergence metrics over pixel_intensity distributions.

What this is for
----------------
These histogram_based divergences are:
- very fast
- dependency_minimal
- great for smoke tests and regression checks

We compute:
- KL(real || synth)
- KL(synth || real)
- JS(real, synth)

Config
------
cfg.metrics.options.distribution.js_kl_hist:
  bins: 32
  eps: 1e_8
  per_class: false   # if true and labels exist, compute per_class divergences too
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ..types import DatasetMeta, MetricResult, RunMeta
from ..features import pixel_histogram


def _to_prob(counts: np.ndarray, eps: float) -> np.ndarray:
    c = np.asarray(counts, dtype=np.float64)
    c = c + float(eps)
    return c / float(np.sum(c))


def kl_div(p: np.ndarray, q: np.ndarray, eps: float) -> float:
    """
    KL(p || q) with epsilon smoothing.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return float(np.sum(p * np.log(p / q)))


def js_div(p: np.ndarray, q: np.ndarray, eps: float) -> float:
    """
    JS(p, q) = 0.5*KL(p||m) + 0.5*KL(q||m), m=(p+q)/2
    """
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m, eps) + 0.5 * kl_div(q, m, eps)


@dataclass
class JSKLMetrics:
    """
    distribution.js_kl_hist
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
        name = "distribution.js_kl_hist"
        opts = (((cfg.get("metrics") or {}).get("options") or {}).get(name) or {})

        bins = int(opts.get("bins", 32))
        eps = float(opts.get("eps", 1e_8))
        per_class = bool(opts.get("per_class", False))

        hR = pixel_histogram(x_real01, bins=bins)
        hS = pixel_histogram(x_synth01, bins=bins)

        pR = _to_prob(np.asarray(hR["counts"], dtype=np.float64), eps=eps)
        pS = _to_prob(np.asarray(hS["counts"], dtype=np.float64), eps=eps)

        out = {
            "bins": bins,
            "eps": eps,
            "kl_real_synth": kl_div(pR, pS, eps=eps),
            "kl_synth_real": kl_div(pS, pR, eps=eps),
            "js": js_div(pR, pS, eps=eps),
            "hist_real": hR,
            "hist_synth": hS,
        }

        # Optional per_class histogram divergences if labels exist
        if per_class and (y_real is not None) and (y_synth is not None):
            K = int(dataset.num_classes)
            per = {}
            for k in range(K):
                r_idx = (y_real == k)
                s_idx = (y_synth == k)
                if int(np.sum(r_idx)) == 0 or int(np.sum(s_idx)) == 0:
                    per[str(k)] = {"status": "skipped", "reason": "missing real or synth samples"}
                    continue
                hkR = pixel_histogram(x_real01[r_idx], bins=bins)
                hkS = pixel_histogram(x_synth01[s_idx], bins=bins)
                pkR = _to_prob(np.asarray(hkR["counts"], dtype=np.float64), eps=eps)
                pkS = _to_prob(np.asarray(hkS["counts"], dtype=np.float64), eps=eps)
                per[str(k)] = {
                    "kl_real_synth": kl_div(pkR, pkS, eps=eps),
                    "kl_synth_real": kl_div(pkS, pkR, eps=eps),
                    "js": js_div(pkR, pkS, eps=eps),
                    "n_real": int(np.sum(r_idx)),
                    "n_synth": int(np.sum(s_idx)),
                }
            out["per_class"] = per

        # Put primary scalar in `value` so it’s easy to track in dashboards
        return MetricResult(
            name=name,
            value=float(out["js"]),
            details=out,
            status="ok",
        )