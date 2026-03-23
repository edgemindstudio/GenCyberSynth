# src/gencysynth/reporting/plots/diversity/nn_distance.py
"""
Nearest_neighbor distance plots (diversity/privacy diagnostic).

Goal
----
Visualize distributions of nearest_neighbor distances that were computed and saved
by the metrics layer.

Typical precomputed artifacts (examples; exact names may differ):
- nn_dist_real_to_real.npy
- nn_dist_synth_to_real.npy
- nn_dist_synth_to_synth.npy

This module does NOT compute distances.
It only loads precomputed arrays if they exist and plots histograms/summary.

Rule A
------
Reads:
  - ctx.run_dir (optional .npy artifacts saved under the run directory)
  - ctx.eval_summary (optional summary stats)
Writes:
  - <out_dir>/nn_distance.<ext>
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from ..config import PlotConfig
from ..core.context import PlotContext


def _cfg_get(d: Dict[str, Any], dotted: str, default=None):
    cur: Any = d
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _find_npy(run_dir: Path, rel_candidates: List[str]) -> Optional[Path]:
    """
    Find the first existing .npy file among candidate relative paths.
    """
    for rel in rel_candidates:
        p = run_dir / rel
        if p.exists() and p.is_file():
            return p
    return None


def _load_npy(p: Path) -> Optional[np.ndarray]:
    try:
        arr = np.load(p)
        arr = np.asarray(arr).reshape(-1)
        # keep only finite values
        arr = arr[np.isfinite(arr)]
        return arr.astype(np.float64, copy=False)
    except Exception:
        return None


def plot_nn_distance(ctx: PlotContext, out_dir: Path, cfg: PlotConfig) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dir = ctx.run_dir
    es = ctx.eval_summary or {}

    # Candidates are intentionally flexible; metrics writer can standardize later.
    # We look under common locations within a run directory.
    candidates = {
        "real→real": [
            "metrics/nn_distance/nn_dist_real_to_real.npy",
            "metrics/privacy/nn_dist_real_to_real.npy",
            "metrics/diversity/nn_dist_real_to_real.npy",
        ],
        "synth→real": [
            "metrics/nn_distance/nn_dist_synth_to_real.npy",
            "metrics/privacy/nn_dist_synth_to_real.npy",
            "metrics/diversity/nn_dist_synth_to_real.npy",
        ],
        "synth→synth": [
            "metrics/nn_distance/nn_dist_synth_to_synth.npy",
            "metrics/privacy/nn_dist_synth_to_synth.npy",
            "metrics/diversity/nn_dist_synth_to_synth.npy",
        ],
    }

    series: List[Tuple[str, np.ndarray]] = []
    for label, rels in candidates.items():
        p = _find_npy(run_dir, rels)
        if p is None:
            continue
        arr = _load_npy(p)
        if arr is None or arr.size == 0:
            continue
        series.append((label, arr))

    # If no arrays exist, we can still attempt to plot summary stats if present.
    # But we avoid fabricating distributions.
    if not series:
        # Optional: plot a tiny summary box if eval_summary has nn_distance stats
        stats = _cfg_get(es, "metrics.nn_distance", None) or _cfg_get(es, "privacy.nn_distance", None)
        if not isinstance(stats, dict) or not stats:
            return []

        # Best_effort: show min/median/mean/max if present
        keys = ["min", "p50", "median", "mean", "p95", "max"]
        items = [(k, stats.get(k, None)) for k in keys if isinstance(stats.get(k, None), (int, float))]
        if not items:
            return []

        fig = plt.figure(figsize=(8, 4.5))
        ax = fig.add_subplot(1, 1, 1)
        ax.axis("off")
        txt = "NN distance summary (from eval_summary)\n\n" + "\n".join([f"{k}: {float(v):.6g}" for k, v in items])
        ax.text(0.02, 0.98, txt, va="top", ha="left")

        written: List[Path] = []
        for ext in cfg.formats:
            p = out_dir / f"nn_distance.{ext}"
            if p.exists() and not cfg.overwrite:
                continue
            fig.savefig(p)
            written.append(p)
        plt.close(fig)
        return written

    # Plot histograms for the available series
    fig = plt.figure(figsize=(10, 5.5))
    ax = fig.add_subplot(1, 1, 1)

    # Use identical binning across series for comparability
    all_vals = np.concatenate([a for _, a in series], axis=0)
    if all_vals.size == 0:
        return []
    lo, hi = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return []

    bins = 50
    for label, arr in series:
        ax.hist(arr, bins=bins, histtype="step", density=True, label=label)

    ax.set_title("Nearest_Neighbor Distance Distributions (precomputed)")
    ax.set_xlabel("distance")
    ax.set_ylabel("density")
    ax.legend(loc="best")

    written: List[Path] = []
    for ext in cfg.formats:
        p = out_dir / f"nn_distance.{ext}"
        if p.exists() and not cfg.overwrite:
            continue
        fig.savefig(p)
        written.append(p)

    plt.close(fig)
    return written


__all__ = ["plot_nn_distance"]
