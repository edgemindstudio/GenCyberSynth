# src/gencysynth/reporting/plots/core/style.py
"""
Matplotlib styling for consistent, readable plots across the repo.

Key constraint: NO hardcoded colors.
------------------------------------
We avoid setting an explicit color palette. Matplotlib defaults are fine and
keep plots consistent without embedding "brand" colors into scientific figures.

Rule A
------
This module does not read/write artifacts. It only sets rcParams.
"""

from __future__ import annotations

from typing import Optional

import matplotlib as mpl


def apply_plot_style(*, dpi: int = 200, fontsize: int = 10) -> None:
    """
    Apply repo-wide plotting style.

    Notes
    -----
    - We do NOT set specific colors.
    - We do set typography + grid + layout friendliness.
    """
    mpl.rcParams.update({
        # Output quality
        "figure.dpi": dpi,
        "savefig.dpi": dpi,

        # Typography
        "font.size": fontsize,
        "axes.titlesize": fontsize + 2,
        "axes.labelsize": fontsize,
        "legend.fontsize": fontsize - 1,

        # Layout
        "figure.autolayout": True,

        # Grid + axis aesthetics
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,

        # Lines
        "lines.linewidth": 1.6,

        # Safer defaults for saving
        "savefig.bbox": "tight",
    })


__all__ = ["apply_plot_style"]
