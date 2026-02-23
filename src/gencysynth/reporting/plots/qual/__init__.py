# src/gencysynth/reporting/plots/qual/__init__.py
"""
Qualitative plot group.

This package produces *visual* artifacts to help humans quickly inspect:
- sample quality
- mode collapse / diversity issues
- obvious leakage (near-copies) and artifacts

Rule A
------
Reads:
  - Run artifacts only (no global state), typically:
      <run_dir>/synthetic/png/<class>/<seed>/*.png
    Optionally:
      <run_dir>/real/png/<class>/... (if real image rendering is persisted)
    Optionally:
      <run_dir>/synthetic/manifest*.json (if your orchestrator writes it)
Writes:
  - <run_dir>/reporting/plots/qual/...

This package does NOT compute metrics; it only renders images.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ..config import PlotConfig
from ..core.context import PlotContext, load_plot_context
from ..core.style import apply_plot_style

from .grids import plot_grids
from .side_by_side import plot_side_by_side
from .galleries import write_galleries


def make_plots(run_dir: Path, out_dir: Path, cfg: PlotConfig, raw_cfg: Dict[str, Any]) -> List[Path]:
    """
    Qual plot-group entrypoint used by reporting.plots.api.

    Parameters
    ----------
    run_dir:
        Run artifact directory:
          <artifacts_root>/<dataset_id>/runs/<run_id>
    out_dir:
        Output directory for this plot group:
          <run_dir>/reporting/plots/qual
    cfg:
        PlotConfig (formats, dpi, overwrite, toggles)
    raw_cfg:
        Full orchestrator config dict (reserved for future extensions)

    Returns
    -------
    List of files written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_plot_style(dpi=cfg.dpi)

    ctx: PlotContext = load_plot_context(run_dir)
    written: List[Path] = []

    written.extend(plot_grids(ctx, out_dir, cfg))
    written.extend(plot_side_by_side(ctx, out_dir, cfg))
    written.extend(write_galleries(ctx, out_dir, cfg))

    return written


__all__ = ["make_plots"]
