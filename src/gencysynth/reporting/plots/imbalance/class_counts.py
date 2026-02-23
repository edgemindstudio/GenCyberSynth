# src/gencysynth/reporting/plots/imbalance/class_counts.py
"""
Class-count plots.

Goal
----
Visualize class counts for:
- real data
- synthetic data
- mixed/combined data (if available)

Where the counts come from
--------------------------
We do best-effort extraction from:
- run_manifest (preferred for dataset info / counts)
- eval_summary  (often includes per_class_counts in evaluation output)
- run_events    (sometimes logs counts or sample plans)

Rule A
------
Reads:
  - ctx.run_manifest, ctx.eval_summary, ctx.run_events
Writes:
  - <out_dir>/class_counts.<ext>
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


def _to_counts_map(x: Any) -> Optional[Dict[str, int]]:
    """
    Normalize different representations into {class_id(str): count(int)}.

    Accepted shapes:
    - {"0": 100, "1": 50, ...}
    - {0: 100, 1: 50, ...}
    - [{"class": 0, "count": 100}, ...]
    - [{"label": 0, "n": 100}, ...]
    """
    if x is None:
        return None

    if isinstance(x, dict):
        out: Dict[str, int] = {}
        for k, v in x.items():
            try:
                out[str(k)] = int(v)
            except Exception:
                continue
        return out if out else None

    if isinstance(x, list):
        out2: Dict[str, int] = {}
        for item in x:
            if not isinstance(item, dict):
                continue
            cls = item.get("class", item.get("label", item.get("k", None)))
            cnt = item.get("count", item.get("n", item.get("num", None)))
            if cls is None or cnt is None:
                continue
            try:
                out2[str(int(cls))] = int(cnt)
            except Exception:
                continue
        return out2 if out2 else None

    return None


def _extract_counts(ctx: PlotContext) -> Tuple[Optional[Dict[str, int]], Optional[Dict[str, int]], Optional[Dict[str, int]]]:
    """
    Try to extract (real_counts, synth_counts, mixed_counts).

    We avoid hard requirements on schema; instead we search multiple likely keys.
    """
    rm = ctx.run_manifest or {}
    es = ctx.eval_summary or {}

    # --- Real counts ---
    real = _to_counts_map(
        _cfg_get(rm, "data.class_counts")
        or _cfg_get(rm, "dataset.class_counts")
        or _cfg_get(rm, "dataset.real.class_counts")
        or _cfg_get(es, "real.per_class_counts")
        or _cfg_get(es, "data.real.per_class_counts")
        or _cfg_get(es, "dataset.real.per_class_counts")
    )

    # --- Synthetic counts ---
    synth = _to_counts_map(
        _cfg_get(rm, "synthetic.per_class_counts")
        or _cfg_get(rm, "synth.per_class_counts")
        or _cfg_get(es, "synthetic.per_class_counts")
        or _cfg_get(es, "synth.per_class_counts")
        or _cfg_get(es, "generated.per_class_counts")
    )

    # --- Mixed counts ---
    mixed = _to_counts_map(
        _cfg_get(rm, "mixed.per_class_counts")
        or _cfg_get(es, "mixed.per_class_counts")
        or _cfg_get(es, "combined.per_class_counts")
        or _cfg_get(es, "train_mix.per_class_counts")
    )

    return real, synth, mixed


def plot_class_counts(ctx: PlotContext, out_dir: Path, cfg: PlotConfig) -> List[Path]:
    """
    Plot class counts if any are available.

    Returns
    -------
    List of paths written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    real, synth, mixed = _extract_counts(ctx)
    if real is None and synth is None and mixed is None:
        return []

    # Determine class universe
    keys = set()
    for m in (real, synth, mixed):
        if m:
            keys |= set(m.keys())
    classes = sorted(keys, key=lambda s: int(s) if str(s).isdigit() else s)

    def vec(m: Optional[Dict[str, int]]) -> np.ndarray:
        if not m:
            return np.zeros((len(classes),), dtype=np.float64)
        return np.asarray([float(m.get(c, 0)) for c in classes], dtype=np.float64)

    y_real = vec(real)
    y_synth = vec(synth)
    y_mixed = vec(mixed)

    # Bar chart with offsets (no custom colors)
    x = np.arange(len(classes), dtype=np.float64)
    width = 0.28

    fig = plt.figure(figsize=(max(8.0, 0.55 * len(classes)), 5.5))
    ax = fig.add_subplot(1, 1, 1)

    # Plot only the series that exist
    n_series = sum(m is not None for m in (real, synth, mixed))
    offsets = []
    if n_series == 1:
        offsets = [0.0]
    elif n_series == 2:
        offsets = [-width / 2, width / 2]
    else:
        offsets = [-width, 0.0, width]

    series = []
    labels = []
    if real is not None:
        series.append(y_real)
        labels.append("real")
    if synth is not None:
        series.append(y_synth)
        labels.append("synthetic")
    if mixed is not None:
        series.append(y_mixed)
        labels.append("mixed")

    for i, (vals, lab) in enumerate(zip(series, labels)):
        ax.bar(x + offsets[i], vals, width=width, label=lab)

    ax.set_title("Class Counts")
    ax.set_xlabel("class")
    ax.set_ylabel("count")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.legend(loc="best")

    written: List[Path] = []
    for ext in cfg.formats:
        p = out_dir / f"class_counts.{ext}"
        if p.exists() and not cfg.overwrite:
            continue
        fig.savefig(p)
        written.append(p)

    plt.close(fig)
    return written


__all__ = ["plot_class_counts"]
