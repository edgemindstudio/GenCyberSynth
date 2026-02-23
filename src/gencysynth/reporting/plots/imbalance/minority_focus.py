# src/gencysynth/reporting/plots/imbalance/minority_focus.py
"""
Minority-focus plots.

Goal
----
When per-class metrics exist, focus on minority / hard classes:
- show "worst-k" classes by F1 (or another chosen metric)
- optionally compare real vs synthetic vs mixed if eval_summary includes multiple splits

Data source
-----------
Best-effort extraction from eval_summary for a dict like:
  per_class:
    f1: {"0": 0.9, "1": 0.4, ...}
or:
  per_class_metrics:
    {"0": {"f1": ...}, "1": {"f1": ...}}

This module is intentionally tolerant because per-class layouts tend to evolve.

Rule A
------
Reads:
  - ctx.eval_summary
Writes:
  - <out_dir>/minority_focus.<ext>
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


def _extract_per_class_metric(eval_summary: Dict[str, Any], metric_name: str) -> Optional[Dict[str, float]]:
    """
    Best-effort extraction of per-class metric map: {class_id: value}.

    Supported patterns:
    - eval_summary["per_class"][metric_name] = {"0": 0.1, ...}
    - eval_summary["per_class_metrics"][<class>][metric_name] = ...
    - eval_summary["metrics"]["per_class"][metric_name] = ...
    """
    # 1) per_class.<metric>
    m = _cfg_get(eval_summary, f"per_class.{metric_name}", None)
    if isinstance(m, dict):
        out = {}
        for k, v in m.items():
            if isinstance(v, (int, float)):
                out[str(k)] = float(v)
        return out if out else None

    # 2) metrics.per_class.<metric>
    m2 = _cfg_get(eval_summary, f"metrics.per_class.{metric_name}", None)
    if isinstance(m2, dict):
        out2 = {}
        for k, v in m2.items():
            if isinstance(v, (int, float)):
                out2[str(k)] = float(v)
        return out2 if out2 else None

    # 3) per_class_metrics.<class>.<metric>
    pcm = _cfg_get(eval_summary, "per_class_metrics", None)
    if isinstance(pcm, dict):
        out3: Dict[str, float] = {}
        for cls, md in pcm.items():
            if not isinstance(md, dict):
                continue
            v = md.get(metric_name, None)
            if isinstance(v, (int, float)):
                out3[str(cls)] = float(v)
        return out3 if out3 else None

    return None


def plot_minority_focus(ctx: PlotContext, out_dir: Path, cfg: PlotConfig, *, metric: str = "f1", worst_k: int = 5) -> List[Path]:
    """
    Plot the worst-k classes by per-class metric (default: F1).

    If per-class metric data is absent, returns [].
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    es = ctx.eval_summary or {}
    per_cls = _extract_per_class_metric(es, metric_name=metric)
    if not per_cls:
        # Try common alternates
        per_cls = _extract_per_class_metric(es, metric_name=f"{metric}_score") or _extract_per_class_metric(es, metric_name=f"{metric}_class")
    if not per_cls:
        return []

    # Sort ascending -> worst first
    items = sorted(per_cls.items(), key=lambda kv: float(kv[1]))
    items = items[: max(1, int(worst_k))]

    classes = [k for k, _ in items]
    values = [float(v) for _, v in items]

    fig = plt.figure(figsize=(10, 5.5))
    ax = fig.add_subplot(1, 1, 1)

    ax.bar(range(len(classes)), values)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=0)
    ax.set_ylim(0.0, 1.0 if np.nanmax(values) <= 1.0 else np.nanmax(values) * 1.1)

    ax.set_title(f"Worst-{len(classes)} Classes by {metric.upper()} (from eval_summary)")
    ax.set_xlabel("class")
    ax.set_ylabel(metric)

    written: List[Path] = []
    for ext in cfg.formats:
        p = out_dir / f"minority_focus.{ext}"
        if p.exists() and not cfg.overwrite:
            continue
        fig.savefig(p)
        written.append(p)

    plt.close(fig)
    return written


__all__ = ["plot_minority_focus"]
