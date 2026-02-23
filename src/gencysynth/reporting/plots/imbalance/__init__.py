# src/gencysynth/reporting/plots/imbalance/__init__.py
"""
Imbalance plot group.

This package contains plots that help diagnose class imbalance effects, such as:
- class counts for REAL / SYNTH / MIXED
- macro vs micro metric comparisons (e.g., F1, precision, recall)
- minority-class focused views (e.g., worst-k classes)

Rule A
------
Reads:
  - run artifacts via PlotContext (run_manifest, eval_summary, run_events)
Writes:
  - <run_dir>/reporting/plots/imbalance/...

This package does NOT compute metrics. It only visualizes artifacts that already exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ..config import PlotConfig
from ..core.context import PlotContext, load_plot_context
from ..core.style import apply_plot_style

from .class_counts import plot_class_counts
from .macro_micro import plot_macro_micro
from .minority_focus import plot_minority_focus


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
          <run_dir>/reporting/plots/imbalance
    cfg:
        PlotConfig (formats, dpi, overwrite, toggles)
    raw_cfg:
        Full orchestrator config dict (reserved for future extensions)

    Returns
    -------
    List of output files written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_plot_style(dpi=cfg.dpi)

    ctx = load_plot_context(run_dir)
    written: List[Path] = []

    written.extend(plot_class_counts(ctx, out_dir, cfg))
    written.extend(plot_macro_micro(ctx, out_dir, cfg))
    written.extend(plot_minority_focus(ctx, out_dir, cfg))

    return written


__all__ = ["make_plots"]
