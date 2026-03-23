# src/gencysynth/reporting/plots/diversity/duplicates.py
"""
Duplicates plots.

Goal
----
Visualize duplicate / near_duplicate rates produced by the metrics layer.

We expect the *metrics* code to have already computed something like:
- duplicates_rate overall
- duplicates_rate_per_class
- (optionally) exact vs near duplicates breakdown

Because schemas evolve, we do best_effort extraction from eval_summary.

Rule A
------
Reads:
  - ctx.eval_summary (preferred)
Writes:
  - <out_dir>/duplicates.<ext>
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


def _to_float_map(x: Any) -> Optional[Dict[str, float]]:
    """
    Normalize different representations into {class_id(str): value(float)}.
    """
    if x is None:
        return None
    if isinstance(x, dict):
        out: Dict[str, float] = {}
        for k, v in x.items():
            if isinstance(v, (int, float)):
                out[str(k)] = float(v)
        return out if out else None
    if isinstance(x, list):
        out2: Dict[str, float] = {}
        for item in x:
            if not isinstance(item, dict):
                continue
            cls = item.get("class", item.get("label", item.get("k", None)))
            val = item.get("value", item.get("rate", item.get("dup_rate", None)))
            if cls is None or val is None:
                continue
            if isinstance(val, (int, float)):
                out2[str(int(cls))] = float(val)
        return out2 if out2 else None
    return None


def _extract_duplicates(eval_summary: Dict[str, Any]) -> Tuple[Optional[float], Optional[Dict[str, float]]]:
    """
    Try to extract:
      - overall duplicate rate (float)
      - per_class duplicate rate map (dict)

    We accept multiple likely paths to keep reporting stable as schemas evolve.
    """
    es = eval_summary or {}

    overall = (
        _cfg_get(es, "metrics.duplicates.rate")
        or _cfg_get(es, "metrics.duplicates.duplicate_rate")
        or _cfg_get(es, "duplicates.rate")
        or _cfg_get(es, "duplicates.duplicate_rate")
        or _cfg_get(es, "diversity.duplicates.rate")
        or _cfg_get(es, "diversity.duplicates.duplicate_rate")
    )
    if not isinstance(overall, (int, float)):
        overall = None
    else:
        overall = float(overall)

    per_class_raw = (
        _cfg_get(es, "metrics.duplicates.per_class_rate")
        or _cfg_get(es, "metrics.duplicates.per_class")
        or _cfg_get(es, "duplicates.per_class_rate")
        or _cfg_get(es, "duplicates.per_class")
        or _cfg_get(es, "diversity.duplicates.per_class_rate")
        or _cfg_get(es, "diversity.duplicates.per_class")
    )
    per_class = _to_float_map(per_class_raw)

    return overall, per_class


def plot_duplicates(ctx: PlotContext, out_dir: Path, cfg: PlotConfig) -> List[Path]:
    """
    Plot duplicate rates (overall + per_class if present).

    Returns
    -------
    List of output files written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    es = ctx.eval_summary or {}
    if not es:
        return []

    overall, per_class = _extract_duplicates(es)
    if overall is None and per_class is None:
        return []

    fig = plt.figure(figsize=(10, 5.5))
    ax = fig.add_subplot(1, 1, 1)

    # Per_class bars if available
    if per_class:
        classes = sorted(per_class.keys(), key=lambda s: int(s) if str(s).isdigit() else s)
        vals = np.asarray([per_class[c] for c in classes], dtype=np.float64)
        x = np.arange(len(classes))
        ax.bar(x, vals)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_xlabel("class")
        ax.set_ylabel("duplicate rate")
        ax.set_title("Duplicate Rate per Class")

        # Overlay overall line if present
        if overall is not None:
            ax.axhline(overall, linestyle="--", linewidth=1.2)
            ax.text(
                0.02,
                overall,
                f" overall={overall:.4f}",
                va="bottom",
                ha="left",
                transform=ax.get_yaxis_transform(),
            )
    else:
        # Overall_only view
        ax.bar([0], [overall if overall is not None else 0.0])
        ax.set_xticks([0])
        ax.set_xticklabels(["overall"])
        ax.set_ylabel("duplicate rate")
        ax.set_title("Duplicate Rate (Overall)")

    # Safe bounds (rates usually in [0,1])
    ax.set_ylim(0.0, 1.0 if (overall is None or overall <= 1.0) else max(1.0, overall * 1.1))

    written: List[Path] = []
    for ext in cfg.formats:
        p = out_dir / f"duplicates.{ext}"
        if p.exists() and not cfg.overwrite:
            continue
        fig.savefig(p)
        written.append(p)

    plt.close(fig)
    return written


__all__ = ["plot_duplicates"]
