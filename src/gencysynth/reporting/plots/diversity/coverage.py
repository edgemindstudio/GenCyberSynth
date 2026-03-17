# src/gencysynth/reporting/plots/diversity/coverage.py
"""
Coverage plots.

Goal
----
Visualize "coverage" style metrics that summarize how well synthetic data spans
the real data manifold / support.

Typical examples (computed elsewhere):
- coverage (overall)
- coverage_per_class
- precision/recall for coverage_style metrics (not classifier precision/recall)

This module does NOT compute coverage. It only visualizes what exists in eval_summary.

Rule A
------
Reads:
  - ctx.eval_summary
Writes:
  - <out_dir>/coverage.<ext>
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

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
    if x is None:
        return None
    if isinstance(x, dict):
        out: Dict[str, float] = {}
        for k, v in x.items():
            if isinstance(v, (int, float)):
                out[str(k)] = float(v)
        return out if out else None
    return None


def _extract_coverage(es: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a small set of coverage_related fields (best_effort).
    Returns a dict with optional keys:
      - overall: float
      - per_class: {class: float}
      - precision: float
      - recall: float
    """
    overall = (
        _cfg_get(es, "metrics.coverage.value")
        or _cfg_get(es, "metrics.coverage.coverage")
        or _cfg_get(es, "coverage.value")
        or _cfg_get(es, "coverage.coverage")
        or _cfg_get(es, "diversity.coverage.value")
        or _cfg_get(es, "diversity.coverage.coverage")
    )
    overall = float(overall) if isinstance(overall, (int, float)) else None

    per_class = (
        _cfg_get(es, "metrics.coverage.per_class")
        or _cfg_get(es, "metrics.coverage.per_class_coverage")
        or _cfg_get(es, "coverage.per_class")
        or _cfg_get(es, "coverage.per_class_coverage")
        or _cfg_get(es, "diversity.coverage.per_class")
    )
    per_class = _to_float_map(per_class)

    # Optional coverage_style precision/recall
    prec = (
        _cfg_get(es, "metrics.coverage.precision")
        or _cfg_get(es, "coverage.precision")
        or _cfg_get(es, "diversity.coverage.precision")
    )
    rec = (
        _cfg_get(es, "metrics.coverage.recall")
        or _cfg_get(es, "coverage.recall")
        or _cfg_get(es, "diversity.coverage.recall")
    )
    prec = float(prec) if isinstance(prec, (int, float)) else None
    rec = float(rec) if isinstance(rec, (int, float)) else None

    return {"overall": overall, "per_class": per_class, "precision": prec, "recall": rec}


def plot_coverage(ctx: PlotContext, out_dir: Path, cfg: PlotConfig) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    es = ctx.eval_summary or {}
    if not es:
        return []

    data = _extract_coverage(es)
    overall = data["overall"]
    per_class = data["per_class"]
    prec = data["precision"]
    rec = data["recall"]

    if overall is None and per_class is None and prec is None and rec is None:
        return []

    fig = plt.figure(figsize=(10, 5.5))
    ax = fig.add_subplot(1, 1, 1)

    # Priority: per_class if available; otherwise show overall and optional precision/recall as markers.
    if per_class:
        classes = sorted(per_class.keys(), key=lambda s: int(s) if str(s).isdigit() else s)
        vals = np.asarray([per_class[c] for c in classes], dtype=np.float64)
        x = np.arange(len(classes))
        ax.bar(x, vals)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_xlabel("class")
        ax.set_ylabel("coverage")
        ax.set_title("Coverage per Class")

        if overall is not None:
            ax.axhline(overall, linestyle="--", linewidth=1.2)
            ax.text(0.02, overall, f" overall={overall:.3f}", va="bottom", ha="left", transform=ax.get_yaxis_transform())
    else:
        labels = []
        vals = []
        if overall is not None:
            labels.append("coverage")
            vals.append(overall)
        if prec is not None:
            labels.append("precision")
            vals.append(prec)
        if rec is not None:
            labels.append("recall")
            vals.append(rec)

        ax.bar(range(len(labels)), vals)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel("value")
        ax.set_title("Coverage Summary")

    ax.set_ylim(0.0, 1.0)

    written: List[Path] = []
    for ext in cfg.formats:
        p = out_dir / f"coverage.{ext}"
        if p.exists() and not cfg.overwrite:
            continue
        fig.savefig(p)
        written.append(p)

    plt.close(fig)
    return written


__all__ = ["plot_coverage"]
