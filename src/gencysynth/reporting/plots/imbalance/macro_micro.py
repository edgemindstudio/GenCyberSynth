# src/gencysynth/reporting/plots/imbalance/macro_micro.py
"""
Macro vs Micro plots.

Goal
----
Show macro and micro versions of common classification metrics (if available),
for example:
- f1_macro vs f1_micro
- precision_macro vs precision_micro
- recall_macro vs recall_micro
- balanced_accuracy (often closer to macro behavior)

Data source
-----------
We do best-effort extraction from eval_summary:
- eval_summary.metrics.*
- eval_summary.summary.*
- eval_summary.results.*

Rule A
------
Reads:
  - ctx.eval_summary
Writes:
  - <out_dir>/macro_micro.<ext>
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def _find_metric(es: Dict[str, Any], names: List[str]) -> Optional[float]:
    """
    Find a metric by trying multiple possible key paths.
    """
    for name in names:
        # Try within common containers and top-level
        for prefix in ("metrics.", "summary.", "results.", ""):
            v = _cfg_get(es, prefix + name, None) if prefix else es.get(name, None)
            if isinstance(v, (int, float)):
                return float(v)
    return None


def plot_macro_micro(ctx: PlotContext, out_dir: Path, cfg: PlotConfig) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    es = ctx.eval_summary or {}
    if not es:
        return []

    # Standard metric pairs to look for (flexible naming)
    pairs = [
        ("F1",      ["f1_macro", "macro_f1", "f1/macro"],      ["f1_micro", "micro_f1", "f1/micro"]),
        ("Precision", ["precision_macro", "macro_precision"],  ["precision_micro", "micro_precision"]),
        ("Recall",  ["recall_macro", "macro_recall"],          ["recall_micro", "micro_recall"]),
    ]

    # Additional singletons (macro-ish)
    singles = [
        ("Balanced Acc", ["balanced_accuracy", "balanced_acc", "bal_acc"]),
        ("Accuracy",     ["accuracy", "acc"]),
    ]

    labels: List[str] = []
    macro_vals: List[float] = []
    micro_vals: List[float] = []

    for title, macro_keys, micro_keys in pairs:
        m = _find_metric(es, macro_keys)
        u = _find_metric(es, micro_keys)
        if m is None and u is None:
            continue
        labels.append(title)
        macro_vals.append(float("nan") if m is None else float(m))
        micro_vals.append(float("nan") if u is None else float(u))

    # If nothing found, do nothing.
    if not labels and not singles:
        return []

    # Prepare figure. We use grouped bars: macro and micro.
    fig = plt.figure(figsize=(10, 5.5))
    ax = fig.add_subplot(1, 1, 1)

    if labels:
        x = list(range(len(labels)))
        w = 0.35
        ax.bar([i - w / 2 for i in x], macro_vals, width=w, label="macro")
        ax.bar([i + w / 2 for i in x], micro_vals, width=w, label="micro")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)

    # Overlay singletons as horizontal markers when available.
    # (Avoids fabricating a "macro/micro" when it's a single metric.)
    y0 = []
    ylab = []
    for title, keys in singles:
        v = _find_metric(es, keys)
        if v is not None:
            y0.append(float(v))
            ylab.append(title)

    if y0:
        # Plot as horizontal lines with labels.
        for v, t in zip(y0, ylab):
            ax.axhline(v, linestyle="--", linewidth=1.2)
            ax.text(0.02, v, f" {t}={v:.3f}", va="bottom", ha="left", transform=ax.get_yaxis_transform())

    ax.set_title("Macro vs Micro Metrics (from eval_summary)")
    ax.set_ylabel("value")
    ax.legend(loc="best")

    written: List[Path] = []
    for ext in cfg.formats:
        p = out_dir / f"macro_micro.{ext}"
        if p.exists() and not cfg.overwrite:
            continue
        fig.savefig(p)
        written.append(p)

    plt.close(fig)
    return written


__all__ = ["plot_macro_micro"]
