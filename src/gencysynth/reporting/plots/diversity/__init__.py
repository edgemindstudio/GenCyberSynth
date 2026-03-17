# src/gencysynth/reporting/plots/diversity/__init__.py
"""
Diversity plot group.

This package contains plots that diagnose *diversity* of generated data, such as:
- duplicates / near_duplicates rates
- coverage_style summaries (how well synthetic spans real space)
- nearest_neighbor distance summaries (useful for diversity + privacy diagnostics)

Rule A
------
Reads:
  - run artifacts via PlotContext (run_manifest, eval_summary, run_events)
  - optional precomputed arrays saved under the run directory (e.g., *.npy)
Writes:
  - <run_dir>/reporting/plots/diversity/...

This package does NOT compute metrics. It only visualizes outputs produced by metrics code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ..config import PlotConfig
from ..core.context import PlotContext, load_plot_context
from ..core.style import apply_plot_style

from .duplicates import plot_duplicates
from .coverage import plot_coverage
from .nn_distance import plot_nn_distance


def make_plots(run_dir: Path, out_dir: Path, cfg: PlotConfig, raw_cfg: Dict[str, Any]) -> List[Path]:
    """
    Diversity plot_group entrypoint used by reporting.plots.api.

    Parameters
    ----------
    run_dir:
        Run artifact directory:
          <artifacts_root>/<dataset_id>/runs/<run_id>
    out_dir:
        Output directory for this plot group:
          <run_dir>/reporting/plots/diversity
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

    ctx: PlotContext = load_plot_context(run_dir)
    written: List[Path] = []

    written.extend(plot_duplicates(ctx, out_dir, cfg))
    written.extend(plot_coverage(ctx, out_dir, cfg))
    written.extend(plot_nn_distance(ctx, out_dir, cfg))

    return written


__all__ = ["make_plots"]
