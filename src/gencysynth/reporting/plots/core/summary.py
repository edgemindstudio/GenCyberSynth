# src/gencysynth/reporting/plots/core/summary.py
"""
Summary dashboard plots from eval_summary.

We do not assume a single eval_summary layout.
Instead, we look for:
- a dict of scalar metrics under common keys
- anything numeric at the top level

Rule A
------
Reads:
  - ctx.eval_summary
Writes:
  - <run_dir>/reporting/plots/core/summary_dashboard.png (and other formats)

Important
---------
This module does NOT compute metrics.
It only visualizes metrics that were already computed and saved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt

from ..config import PlotConfig
from .context import PlotContext


def _flatten_numeric_metrics(d: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract a flat dict of numeric metrics.

    Common patterns supported:
    - d["metrics"] = {"fid": 12.3, ...}
    - d["summary"] = {"kid": 0.01, ...}
    - top-level numeric values
    """
    out: Dict[str, float] = {}

    # Common containers
    for container_key in ("metrics", "summary", "scores", "results"):
        v = d.get(container_key, None)
        if isinstance(v, dict):
            for k, val in v.items():
                if isinstance(val, (int, float)):
                    out[f"{container_key}.{k}"] = float(val)

    # Top-level numeric values
    for k, val in d.items():
        if isinstance(val, (int, float)):
            out[str(k)] = float(val)

    return out


def make_summary_dashboard(ctx: PlotContext, out_dir: Path, cfg: PlotConfig) -> List[Path]:
    """
    Create a compact dashboard of key eval metrics.

    If eval_summary is missing or has no numeric metrics, returns [].
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = _flatten_numeric_metrics(ctx.eval_summary)
    if not metrics:
        return []

    # Keep it compact: pick up to N metrics, prioritizing well-known ones
    priority = []
    for k in metrics.keys():
        lk = k.lower()
        if "fid" in lk or "kid" in lk or "mmd" in lk or "js" in lk or "kl" in lk or "ece" in lk or "brier" in lk:
            priority.append(k)
        elif "accuracy" in lk or "acc" in lk or "f1" in lk or "balanced" in lk:
            priority.append(k)
    priority = list(dict.fromkeys(priority))  # stable unique

    picked = priority[:12]
    if len(picked) < 12:
        # Fill with remaining metrics
        for k in sorted(metrics.keys()):
            if k not in picked:
                picked.append(k)
            if len(picked) >= 12:
                break

    labels = picked
    values = [metrics[k] for k in picked]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)

    # Horizontal bar chart is readable for metric names.
    ax.barh(range(len(labels)), values)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Eval Summary Metrics (from eval_summary)")
    ax.set_xlabel("value")

    written: List[Path] = []
    for ext in cfg.formats:
        p = out_dir / f"summary_dashboard.{ext}"
        if p.exists() and not cfg.overwrite:
            continue
        fig.savefig(p)
        written.append(p)

    plt.close(fig)
    return written


__all__ = ["make_summary_dashboard"]
