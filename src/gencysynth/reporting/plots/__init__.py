# src/gencysynth/reporting/plots/__init__.py
"""
gencysynth.reporting.plots

Run-level plotting orchestration.

This package is intentionally thin:
- It provides stable entrypoints (plot_all, plot_groups, plot_group).
- It standardizes how plotting code reads from and writes to artifacts.
- It does NOT compute metrics. It only visualizes results already written
  by training/evaluation into run artifacts (Rule A).

Directory convention (Rule A)
-----------------------------
All plots produced for a run MUST be written under the run directory:

  <artifacts_root>/<dataset_id>/runs/<run_id>/reporting/plots/<group>/

Where:
- dataset_id is derived from the dataset fingerprint/registry (via run manifest).
- run_id is the run identifier (see run_manifest).
- group is one of: core, imbalance, diversity, qual, ...

The orchestrator resolves:
- run_dir: the run artifact directory for a specific run
- out_dir: <run_dir>/reporting/plots

Public API
----------
- plot_all(...)
- plot_groups(...)
- plot_group(...)
"""

from .api import plot_all, plot_groups, plot_group
from .config import PlotConfig, load_plot_config

__all__ = [
    "PlotConfig",
    "load_plot_config",
    "plot_all",
    "plot_groups",
    "plot_group",
]