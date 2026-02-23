# src/gencysynth/reporting/plots/core/__init__.py
"""
Core plot group.

This package produces "run-level" plots that are generally useful for any model:
- High-level metric summary dashboards (from eval_summary)
- Training curves (from run_events)
- Optional tables rendered to PNG (from eval_summary)

Rule A
------
Reads:
  - run artifacts under <run_dir>/...
Writes:
  - plots under <run_dir>/reporting/plots/core/...
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ..config import PlotConfig
from .context import PlotContext, load_plot_context
from .style import apply_plot_style
from .summary import make_summary_dashboard
from .training_curves import make_training_curves
from .tables import render_metric_tables


def make_plots(run_dir: Path, out_dir: Path, cfg: PlotConfig, raw_cfg: Dict[str, Any]) -> List[Path]:
    """
    Group entrypoint used by reporting.plots.api.

    Parameters
    ----------
    run_dir:
        Run artifact directory:
          <artifacts_root>/<dataset_id>/runs/<run_id>
    out_dir:
        Output directory for this plot group:
          <run_dir>/reporting/plots/core
    cfg:
        PlotConfig (formats, dpi, overwrite, toggles)
    raw_cfg:
        Full orchestrator config dict (unused here except for future extensions)

    Returns
    -------
    List[Path] of files written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Apply a consistent Matplotlib style for all plots in this group.
    apply_plot_style(dpi=cfg.dpi)

    # Load run artifacts once into a PlotContext.
    ctx = load_plot_context(run_dir)

    written: List[Path] = []

    # 1) Run-level summary dashboard (eval_summary -> visual summary)
    written.extend(make_summary_dashboard(ctx, out_dir, cfg))

    # 2) Training curves (run_events -> curves)
    written.extend(make_training_curves(ctx, out_dir, cfg))

    # 3) Optional tables rendered to images (eval_summary -> table)
    # This is safe and helpful for quick browsing in artifacts.
    written.extend(render_metric_tables(ctx, out_dir, cfg))

    return written


__all__ = ["make_plots",
           "PlotContext",
           "load_plot_context"]
