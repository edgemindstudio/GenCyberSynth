# src/gencysynth/reporting/plots/api.py
"""
Run_level plotting API.

This is the orchestrator layer that:
1) Resolves run_dir (artifact location) in a dataset_agnostic manner.
2) Loads plot configuration (defaults.yaml + overrides).
3) Delegates per_group plotting to group modules (core/, imbalance/, etc.).
4) Writes plots under run_dir/reporting/plots/<group>/...

Rule A constraints
------------------
- READ: only from run artifacts (e.g., run_manifest.json, eval_summary.json, run_events.json).
- WRITE: only under run_dir/reporting/plots/...
- Do not compute metrics; visualize what was already computed.

Integration
-----------
This file is designed to be called by your orchestrator / CLI, e.g.:

  from gencysynth.reporting.plots import plot_all
  plot_all(cfg, run_dir=".../artifacts/<dataset_id>/runs/<run_id>")

If run_dir is omitted, it will attempt a best_effort resolution using cfg.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import json

from .config import PlotConfig, load_plot_config


# -----------------------------
# Artifact reading utilities
# -----------------------------
def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf_8") as f:
        return json.load(f)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _cfg_get(cfg: Dict[str, Any], dotted: str, default=None):
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


# -----------------------------
# Run directory resolution
# -----------------------------
def resolve_run_dir(cfg: Dict[str, Any], run_dir: Optional[str | Path] = None) -> Path:
    """
    Resolve the run directory containing artifacts for a single run.

    Priority:
    1) explicit run_dir argument
    2) cfg["paths"]["run_dir"] (if your orchestrator sets it)
    3) cfg["run"]["dir"] or cfg["run_dir"]
    4) fallback: <paths.artifacts>/<dataset_id>/runs/<run_id> if both are present
       (requires cfg contains dataset_id + run_id)

    This function is intentionally conservative: it avoids guessing too much.
    """
    if run_dir is not None:
        return Path(run_dir)

    # Common orchestrator patterns
    p = _cfg_get(cfg, "paths.run_dir", None) or _cfg_get(cfg, "run.dir", None) or cfg.get("run_dir", None)
    if p:
        return Path(p)

    # Fallback: artifacts root + dataset_id + run_id
    artifacts_root = Path(_cfg_get(cfg, "paths.artifacts", "artifacts"))
    dataset_id = _cfg_get(cfg, "data.dataset_id", None) or cfg.get("dataset_id", None)
    run_id = _cfg_get(cfg, "run.run_id", None) or cfg.get("run_id", None)

    if dataset_id and run_id:
        return artifacts_root / str(dataset_id) / "runs" / str(run_id)

    raise ValueError(
        "Could not resolve run_dir. Provide run_dir explicitly or set one of: "
        "cfg['paths']['run_dir'], cfg['run']['dir'], cfg['run_dir'], or "
        "(paths.artifacts + dataset_id + run_id)."
    )


def resolve_plots_root(run_dir: Path) -> Path:
    """
    Plots root directory for a run (Rule A contract).
    """
    return run_dir / "reporting" / "plots"


# -----------------------------
# Group dispatch (pluggable)
# -----------------------------
def _import_group_plotter(group: str):
    """
    Import the group's plotter module lazily.

    Each group module should expose a function:
      make_plots(run_dir: Path, out_dir: Path, cfg: PlotConfig, raw_cfg: dict) -> List[Path]
    """
    if group == "core":
        from .core import make_plots as fn  # type: ignore
        return fn
    if group == "imbalance":
        from .imbalance import make_plots as fn  # type: ignore
        return fn
    if group == "diversity":
        from .diversity import make_plots as fn  # type: ignore
        return fn
    if group == "qual":
        from .qual import make_plots as fn  # type: ignore
        return fn

    # Allow future groups without changing this file:
    # reporting/plots/<group>/__init__.py must define make_plots.
    try:
        mod = __import__(f"gencysynth.reporting.plots.{group}", fromlist=["make_plots"])
        return getattr(mod, "make_plots")
    except Exception as e:
        raise ValueError(f"Unknown plot group '{group}'. No plotter found. Error: {e}") from e


# -----------------------------
# Public API
# -----------------------------
def plot_group(
    cfg: Dict[str, Any],
    group: str,
    *,
    run_dir: Optional[str | Path] = None,
    plot_cfg: Optional[PlotConfig] = None,
) -> List[Path]:
    """
    Generate plots for a single group.

    Writes plots under:
      <run_dir>/reporting/plots/<group>/

    Returns a list of file paths written.
    """
    run_dir_p = resolve_run_dir(cfg, run_dir)
    plots_root = resolve_plots_root(run_dir_p)
    out_dir = plots_root / group
    _ensure_dir(out_dir)

    # Load plot config from cfg overrides if not provided.
    if plot_cfg is None:
        # Convention: cfg["reporting"]["plots"] holds plot settings.
        plot_cfg = load_plot_config(_cfg_get(cfg, "reporting.plots", {}) or {})

    # Respect enabled_groups
    if not plot_cfg.enabled_groups.get(group, False):
        return []

    plotter = _import_group_plotter(group)
    written: List[Path] = plotter(run_dir_p, out_dir, plot_cfg, cfg)  # type: ignore[misc]
    return written


def plot_groups(
    cfg: Dict[str, Any],
    groups: Iterable[str],
    *,
    run_dir: Optional[str | Path] = None,
    plot_cfg: Optional[PlotConfig] = None,
) -> List[Path]:
    """
    Generate plots for multiple groups.
    """
    all_written: List[Path] = []
    for g in groups:
        all_written.extend(plot_group(cfg, g, run_dir=run_dir, plot_cfg=plot_cfg))
    return all_written


def plot_all(
    cfg: Dict[str, Any],
    *,
    run_dir: Optional[str | Path] = None,
    plot_cfg: Optional[PlotConfig] = None,
) -> List[Path]:
    """
    Generate plots for all enabled groups.

    The set of groups is taken from plot_cfg.enabled_groups.
    """
    if plot_cfg is None:
        plot_cfg = load_plot_config(_cfg_get(cfg, "reporting.plots", {}) or {})

    groups = [g for g, enabled in plot_cfg.enabled_groups.items() if enabled]
    return plot_groups(cfg, groups, run_dir=run_dir, plot_cfg=plot_cfg)


def write_plot_run_meta(
    cfg: Dict[str, Any],
    *,
    run_dir: Optional[str | Path] = None,
    plot_cfg: Optional[PlotConfig] = None,
) -> Path:
    """
    Optional helper: write a small metadata record describing the plotting config used.

    This is useful for reproducibility and debugging:
      <run_dir>/reporting/plots/plot_run_meta.json
    """
    run_dir_p = resolve_run_dir(cfg, run_dir)
    plots_root = resolve_plots_root(run_dir_p)
    _ensure_dir(plots_root)

    if plot_cfg is None:
        plot_cfg = load_plot_config(_cfg_get(cfg, "reporting.plots", {}) or {})

    out = plots_root / "plot_run_meta.json"
    payload = {
        "plot_cfg": asdict(plot_cfg),
        "run_dir": str(run_dir_p),
        "plots_root": str(plots_root),
    }
    with open(out, "w", encoding="utf_8") as f:
        json.dump(payload, f, indent=2)
    return out
