# src/gencysynth/reporting/plots/qual/grids.py
"""
Qualitative grids.

Goal
----
Render compact image grids (montages) from existing PNG artifacts, typically:
  <run_dir>/synthetic/png/<class>/<seed>/*.png

We intentionally avoid assuming image shape; we load PNGs and tile them.

Rule A
------
Reads:
  - ctx.run_dir / synthetic/png/...
Writes:
  - <out_dir>/grid_<scope>.<ext>

Notes
-----
- We keep this lightweight and robust: if inputs are missing, we skip plots.
- We do not compute images; we only render existing ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from ..config import PlotConfig
from ..core.context import PlotContext


@dataclass(frozen=True)
class GridSpec:
    """Grid layout specification."""
    rows: int
    cols: int
    max_per_class: int
    seed_limit: int  # max number of seeds to include (most recent N)


def _list_pngs(dir_path: Path) -> List[Path]:
    if not dir_path.exists():
        return []
    return sorted([p for p in dir_path.glob("*.png") if p.is_file()])


def _safe_open_rgb(p: Path) -> Image.Image:
    """
    Open image and normalize to RGB for consistent tiling.
    """
    im = Image.open(p)
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    if im.mode == "L":
        im = im.convert("RGB")
    return im


def _make_montage(images: Sequence[Image.Image], rows: int, cols: int, pad: int = 2) -> Optional[Image.Image]:
    """
    Create a simple montage image (PIL) with padding.
    Returns None if images is empty.
    """
    if not images:
        return None

    # Use first image size as canonical tile size; resize others to match.
    w0, h0 = images[0].size
    tiles = [im.resize((w0, h0), resample=Image.BILINEAR) for im in images[: rows * cols]]

    out_w = cols * w0 + (cols - 1) * pad
    out_h = rows * h0 + (rows - 1) * pad
    canvas = Image.new("RGB", (out_w, out_h), color=(255, 255, 255))

    for idx, im in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        x = c * (w0 + pad)
        y = r * (h0 + pad)
        canvas.paste(im, (x, y))
    return canvas


def _default_grid_spec() -> GridSpec:
    """
    Defaults tuned for quick-run inspection.
    - For each class, we take up to max_per_class images.
    - We assemble across classes into a single montage.
    """
    return GridSpec(rows=6, cols=9, max_per_class=3, seed_limit=1)


def _discover_synth_png_root(run_dir: Path) -> Path:
    """
    Canonical location for PNG outputs (as used by unified synth):
      <run_dir>/synthetic/png/<class>/<seed>/*.png

    If you later change the canonical path, update it here.
    """
    return run_dir / "synthetic" / "png"


def _discover_available_seeds(class_dir: Path) -> List[str]:
    """
    Under <class_dir>/ there may be seed folders: <seed>/...
    Return seed folder names sorted numerically when possible.
    """
    if not class_dir.exists():
        return []
    seeds = [p.name for p in class_dir.iterdir() if p.is_dir()]
    # numeric sort when possible
    def _key(s: str):
        return (0, int(s)) if s.isdigit() else (1, s)
    return sorted(seeds, key=_key)


def _collect_images_for_class(
    synth_png_root: Path,
    class_id: str,
    *,
    seed_limit: int,
    max_images: int,
) -> List[Path]:
    """
    Collect up to `max_images` png paths for a given class, preferring the most recent seeds.

    Layout:
      synth_png_root/<class_id>/<seed>/<files>.png
    """
    class_dir = synth_png_root / class_id
    seeds = _discover_available_seeds(class_dir)
    if not seeds:
        return []

    # Prefer the last seeds (most recent numerically or lexicographically).
    chosen = seeds[-max(1, int(seed_limit)) :]
    paths: List[Path] = []
    for seed in chosen:
        seed_dir = class_dir / seed
        pngs = _list_pngs(seed_dir)
        paths.extend(pngs)
        if len(paths) >= max_images:
            break
    return paths[:max_images]


def plot_grids(ctx: PlotContext, out_dir: Path, cfg: PlotConfig) -> List[Path]:
    """
    Produce one or more montage grids from synthetic PNG outputs.

    Returns
    -------
    List of output files written.
    """
    run_dir = ctx.run_dir
    synth_root = _discover_synth_png_root(run_dir)
    if not synth_root.exists():
        return []

    spec = _default_grid_spec()

    # Discover classes as subfolders under synthetic/png/
    class_ids = sorted([p.name for p in synth_root.iterdir() if p.is_dir()],
                       key=lambda s: int(s) if s.isdigit() else s)
    if not class_ids:
        return []

    # Collect images across classes
    tiles: List[Image.Image] = []
    labels: List[str] = []
    for cid in class_ids:
        paths = _collect_images_for_class(
            synth_root, cid, seed_limit=spec.seed_limit, max_images=spec.max_per_class
        )
        for p in paths:
            try:
                tiles.append(_safe_open_rgb(p))
                labels.append(f"class {cid}")
            except Exception:
                continue

    if not tiles:
        return []

    # Determine grid size automatically if defaults too small/large
    n = len(tiles)
    rows = spec.rows
    cols = spec.cols
    if rows * cols < n:
        cols = min(max(cols, 6), 12)
        rows = int(math.ceil(n / cols))

    montage = _make_montage(tiles, rows=rows, cols=cols)
    if montage is None:
        return []

    # Save montage with matplotlib so it respects global style + consistent formats.
    fig = plt.figure(figsize=(cols * 1.2, rows * 1.2))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(np.asarray(montage))
    ax.set_axis_off()
    ax.set_title("Synthetic Samples Grid (qual)", fontsize=12)

    written: List[Path] = []
    for ext in cfg.formats:
        p = out_dir / f"grid_synth.{ext}"
        if p.exists() and not cfg.overwrite:
            continue
        fig.savefig(p, dpi=cfg.dpi, bbox_inches="tight")
        written.append(p)

    plt.close(fig)
    return written


__all__ = ["plot_grids"]
