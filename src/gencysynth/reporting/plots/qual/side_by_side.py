# src/gencysynth/reporting/plots/qual/side_by_side.py
"""
Side_by_side comparisons (REAL vs SYNTH).

Goal
----
Create compact panels that place:
  - real samples (if available)
  - synthetic samples
next to each other for a subset of classes.

This helps quickly spot:
- unrealistic artifacts
- distribution mismatch
- missing modes / collapse

Rule A
------
Reads:
  - ctx.run_dir/real/png/... (optional)
  - ctx.run_dir/synthetic/png/... (preferred)
Writes:
  - <out_dir>/side_by_side_<class>.{ext}  (or a combined figure)

Important
---------
This module only visualizes existing artifacts; it doesn't compute or render real data.
If real PNGs are not available in the run dir, we degrade gracefully and only show synth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from ..config import PlotConfig
from ..core.context import PlotContext


@dataclass(frozen=True)
class SxSSpec:
    """Side_by_side figure spec."""
    per_class: int = 8
    seed_limit: int = 1
    max_classes: int = 6  # limit for quick test runs


def _list_pngs(p: Path) -> List[Path]:
    if not p.exists():
        return []
    return sorted([x for x in p.glob("*.png") if x.is_file()])


def _safe_open_rgb(p: Path) -> Image.Image:
    im = Image.open(p)
    if im.mode == "L":
        return im.convert("RGB")
    if im.mode != "RGB":
        return im.convert("RGB")
    return im


def _discover_synth_root(run_dir: Path) -> Path:
    return run_dir / "synthetic" / "png"


def _discover_real_root(run_dir: Path) -> Path:
    """
    Optional canonical location for real images if your pipeline persists them:
      <run_dir>/real/png/<class>/<...>.png
    """
    return run_dir / "real" / "png"


def _seed_dirs(class_dir: Path) -> List[Path]:
    if not class_dir.exists():
        return []
    dirs = [p for p in class_dir.iterdir() if p.is_dir()]
    # numeric sort if possible
    def _key(p: Path):
        return (0, int(p.name)) if p.name.isdigit() else (1, p.name)
    return sorted(dirs, key=_key)


def _collect_synth(class_dir: Path, *, seed_limit: int, per_class: int) -> List[Path]:
    seeds = _seed_dirs(class_dir)
    if not seeds:
        return []
    chosen = seeds[-max(1, int(seed_limit)) :]
    out: List[Path] = []
    for sd in chosen:
        out.extend(_list_pngs(sd))
        if len(out) >= per_class:
            break
    return out[:per_class]


def _collect_real(class_dir: Path, *, per_class: int) -> List[Path]:
    # Real images are not expected to be seeded; just pick first N
    return _list_pngs(class_dir)[:per_class]


def _plot_row(axs, imgs: List[Path], title: str) -> None:
    """
    Plot a list of images into a row of axes.
    """
    for i, ax in enumerate(axs):
        ax.set_axis_off()
        if i >= len(imgs):
            continue
        try:
            im = _safe_open_rgb(imgs[i])
            ax.imshow(np.asarray(im))
        except Exception:
            continue
    axs[0].set_title(title, fontsize=10, loc="left")


def plot_side_by_side(ctx: PlotContext, out_dir: Path, cfg: PlotConfig) -> List[Path]:
    run_dir = ctx.run_dir
    synth_root = _discover_synth_root(run_dir)
    if not synth_root.exists():
        return []

    real_root = _discover_real_root(run_dir)
    spec = SxSSpec()

    # Discover classes from synth root
    class_ids = sorted([p.name for p in synth_root.iterdir() if p.is_dir()],
                       key=lambda s: int(s) if s.isdigit() else s)
    if not class_ids:
        return []

    class_ids = class_ids[: spec.max_classes]

    written: List[Path] = []
    for cid in class_ids:
        synth_cls_dir = synth_root / cid
        synth_imgs = _collect_synth(synth_cls_dir, seed_limit=spec.seed_limit, per_class=spec.per_class)
        if not synth_imgs:
            continue

        real_imgs: List[Path] = []
        real_cls_dir = real_root / cid
        if real_cls_dir.exists():
            real_imgs = _collect_real(real_cls_dir, per_class=spec.per_class)

        # Layout: 2 rows if real exists, else 1 row
        ncols = max(1, min(spec.per_class, len(synth_imgs)))
        nrows = 2 if real_imgs else 1

        fig = plt.figure(figsize=(1.6 * ncols, 2.2 * nrows))
        gs = fig.add_gridspec(nrows=nrows, ncols=ncols)

        # Top row: REAL (optional)
        if real_imgs:
            axs_real = [fig.add_subplot(gs[0, j]) for j in range(ncols)]
            _plot_row(axs_real, real_imgs[:ncols], title=f"class {cid} — REAL")

        # Bottom row (or only row): SYNTH
        row_idx = 1 if real_imgs else 0
        axs_synth = [fig.add_subplot(gs[row_idx, j]) for j in range(ncols)]
        _plot_row(axs_synth, synth_imgs[:ncols], title=f"class {cid} — SYNTH")

        fig.suptitle(f"Side_by_Side Qualitative Comparison (class {cid})", fontsize=12)

        for ext in cfg.formats:
            p = out_dir / f"side_by_side_class_{cid}.{ext}"
            if p.exists() and not cfg.overwrite:
                continue
            fig.savefig(p, dpi=cfg.dpi, bbox_inches="tight")
            written.append(p)

        plt.close(fig)

    return written


__all__ = ["plot_side_by_side"]
