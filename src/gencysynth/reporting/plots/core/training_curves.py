# src/gencysynth/reporting/plots/core/training_curves.py
"""
Training curves from run_events.

Expected input
--------------
run_events.json is expected to contain a list of events OR a dict with an "events" list.
Each event should have:
  - "step" or "epoch" or "global_step" (x_axis)
  - "scalars" dict or flat keys that include loss/metrics

Because event formats vary across trainers, we implement best_effort parsing.

Rule A
------
Reads:
  - ctx.run_events
Writes:
  - <run_dir>/reporting/plots/core/training_curves.png (and other formats)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from ..config import PlotConfig
from .context import PlotContext


def _events_list(run_events: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(run_events, list):
        return [e for e in run_events if isinstance(e, dict)]
    if isinstance(run_events, dict):
        ev = run_events.get("events", None)
        if isinstance(ev, list):
            return [e for e in ev if isinstance(e, dict)]
    return []


def _get_step(e: Dict[str, Any]) -> Optional[float]:
    for k in ("step", "epoch", "global_step", "iter", "iteration"):
        if k in e:
            try:
                return float(e[k])
            except Exception:
                pass
    return None


def _scalar_dict(e: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract scalar measurements from an event.

    Supports:
      - e["scalars"] = {"loss": 1.2, ...}
      - flat keys like e["train_loss"], e["val_loss"], ...
    """
    out: Dict[str, float] = {}

    scalars = e.get("scalars", None)
    if isinstance(scalars, dict):
        for k, v in scalars.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                continue

    # Flat fallbacks (common keys)
    for k in list(e.keys()):
        if k in ("step", "epoch", "global_step", "time", "timestamp", "wall_time", "scalars"):
            continue
        v = e.get(k)
        if isinstance(v, (int, float)):
            out[str(k)] = float(v)

    return out


def _collect_series(events: List[Dict[str, Any]]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Build series mapping: name -> (x, y)

    We align on step where possible; if some events lack the scalar, we skip that point.
    """
    xs: List[float] = []
    scalar_rows: List[Dict[str, float]] = []

    for e in events:
        step = _get_step(e)
        if step is None:
            continue
        scalars = _scalar_dict(e)
        if not scalars:
            continue
        xs.append(step)
        scalar_rows.append(scalars)

    if not xs:
        return {}

    # Determine which scalar keys appear often enough.
    keys = set()
    for row in scalar_rows:
        keys |= set(row.keys())

    series: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    x_arr = np.asarray(xs, dtype=np.float64)

    for k in sorted(keys):
        ys: List[float] = []
        xk: List[float] = []
        for x, row in zip(xs, scalar_rows):
            if k in row:
                xk.append(x)
                ys.append(row[k])
        if len(ys) >= 2:  # curves need at least 2 points to be useful
            series[k] = (np.asarray(xk, dtype=np.float64), np.asarray(ys, dtype=np.float64))

    return series


def make_training_curves(ctx: PlotContext, out_dir: Path, cfg: PlotConfig) -> List[Path]:
    """
    Write a small set of training_curve plots from run_events.

    Output files
    ------------
    - core/training_curves.<ext>
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    events = _events_list(ctx.run_events)
    if not events:
        return []

    series = _collect_series(events)
    if not series:
        return []

    # Prefer "loss_like" keys first, then everything else.
    preferred = []
    for k in series.keys():
        lk = k.lower()
        if "loss" in lk or "fid" in lk or "kid" in lk or "acc" in lk or "f1" in lk:
            preferred.append(k)
    preferred = preferred[:8]  # keep plot readable
    other = [k for k in series.keys() if k not in preferred][:8]
    plot_keys = preferred + other
    if not plot_keys:
        return []

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)

    for k in plot_keys:
        x, y = series[k]
        ax.plot(x, y, label=k)

    ax.set_title("Training Curves (from run_events)")
    ax.set_xlabel("step/epoch")
    ax.set_ylabel("value")
    ax.legend(loc="best", ncol=1)

    written: List[Path] = []
    for ext in cfg.formats:
        p = out_dir / f"training_curves.{ext}"
        if p.exists() and not cfg.overwrite:
            continue
        fig.savefig(p)
        written.append(p)

    plt.close(fig)
    return written


__all__ = ["make_training_curves"]
