# src/gencysynth/reporting/plots/config.py
"""
Plot configuration.

This module defines:
- PlotConfig: a typed config with safe defaults
- load_plot_config: merges defaults.yaml with user config dict

Rule A note
-----------
Plot code should not guess arbitrary locations. It should resolve *run_dir*
and write output under run_dir/reporting/plots/...

Config is only used to:
- enable/disable plot groups
- control lightweight plot behavior (dpi, formats, max images, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


# -----------------------------
# Data model
# -----------------------------
@dataclass
class PlotConfig:
    """
    Plot settings for run-level reporting.

    Keep this intentionally small:
    - plotting should be deterministic and driven by artifact content.
    - avoid model-specific logic here.
    """

    # Enabled plot groups (subfolders under reporting/plots/)
    enabled_groups: Dict[str, bool] = field(default_factory=lambda: {
        "core": True,
        "imbalance": True,
        "diversity": True,
        "qual": True,
    })

    # Output behavior
    formats: Iterable[str] = ("png",)  # png is the default artifact-friendly format
    dpi: int = 200

    # Qualitative plots often generate many images.
    qual_max_images: int = 64
    qual_grid_cols: int = 8

    # If True, overwrite existing plot files for the same run_id.
    overwrite: bool = True

    # Optional: allow suppressing expensive plots.
    # (Plot modules should honor this flag where relevant.)
    fast_mode: bool = False


# -----------------------------
# Utilities
# -----------------------------
def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep-merge src into dst (dicts only), returning dst.

    This is used to merge user config overrides into defaults.
    """
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_defaults_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_plot_config(
    cfg: Optional[Dict[str, Any]] = None,
    *,
    defaults_path: Optional[Path] = None,
) -> PlotConfig:
    """
    Build PlotConfig from defaults.yaml + cfg overrides.

    Expected user override location (typical):
      cfg["reporting"]["plots"] = {...}

    But this loader is flexible: you can pass exactly the plot subtree dict.
    """
    cfg = cfg or {}

    # Locate defaults.yaml relative to this file if not provided.
    if defaults_path is None:
        defaults_path = Path(__file__).with_name("defaults.yaml")

    defaults = load_defaults_yaml(defaults_path)

    # Merge defaults + user overrides
    merged: Dict[str, Any] = {}
    _deep_update(merged, defaults)
    _deep_update(merged, cfg)

    # Map merged dict -> PlotConfig fields (with guardrails)
    enabled_groups = dict(merged.get("enabled_groups", {}))
    formats = tuple(merged.get("formats", ("png",)))
    dpi = int(merged.get("dpi", 200))
    qual_max_images = int(merged.get("qual_max_images", 64))
    qual_grid_cols = int(merged.get("qual_grid_cols", 8))
    overwrite = bool(merged.get("overwrite", True))
    fast_mode = bool(merged.get("fast_mode", False))

    return PlotConfig(
        enabled_groups=enabled_groups or PlotConfig().enabled_groups,
        formats=formats,
        dpi=dpi,
        qual_max_images=qual_max_images,
        qual_grid_cols=qual_grid_cols,
        overwrite=overwrite,
        fast_mode=fast_mode,
    )
