# src/gencysynth/reporting/plots/qual/galleries.py
"""
Galleries: write "browseable" qualitative outputs.

Goal
----
Produce lightweight gallery-style outputs for manual browsing:
- Copy (or symlink) a small subset of synthetic PNGs into a clean directory tree
- Optionally create an index text file with counts and hints

We avoid HTML to keep dependencies minimal. You can add HTML later if desired.

Rule A
------
Reads:
  - ctx.run_dir/synthetic/png/<class>/<seed>/*.png
Writes:
  - <out_dir>/gallery/<class>/<seed>/...  (copied subset)
  - <out_dir>/gallery/INDEX.txt
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import shutil

from ..config import PlotConfig
from ..core.context import PlotContext


@dataclass(frozen=True)
class GallerySpec:
    """
    Gallery policy:
    - copy at most max_per_seed images per seed folder
    - include at most seed_limit seeds per class (prefer most recent)
    - include at most max_classes classes for quick-run sanity tests
    """
    max_per_seed: int = 50
    seed_limit: int = 1
    max_classes: int = 20


def _discover_synth_root(run_dir: Path) -> Path:
    return run_dir / "synthetic" / "png"


def _seed_dirs(class_dir: Path) -> List[Path]:
    if not class_dir.exists():
        return []
    dirs = [p for p in class_dir.iterdir() if p.is_dir()]
    def _key(p: Path):
        return (0, int(p.name)) if p.name.isdigit() else (1, p.name)
    return sorted(dirs, key=_key)


def _list_pngs(seed_dir: Path) -> List[Path]:
    if not seed_dir.exists():
        return []
    return sorted([p for p in seed_dir.glob("*.png") if p.is_file()])


def write_galleries(ctx: PlotContext, out_dir: Path, cfg: PlotConfig) -> List[Path]:
    """
    Create a gallery subtree under <out_dir>/gallery/...

    Returns
    -------
    List of files written.
    """
    run_dir = ctx.run_dir
    synth_root = _discover_synth_root(run_dir)
    if not synth_root.exists():
        return []

    spec = GallerySpec()
    gallery_root = out_dir / "gallery"
    gallery_root.mkdir(parents=True, exist_ok=True)

    # Discover classes
    class_ids = sorted([p.name for p in synth_root.iterdir() if p.is_dir()],
                       key=lambda s: int(s) if s.isdigit() else s)
    if not class_ids:
        return []

    class_ids = class_ids[: spec.max_classes]

    # Copy a subset into gallery
    copied: List[Path] = []
    index_lines: List[str] = []
    index_lines.append("GenCyberSynth — Qual Gallery")
    index_lines.append(f"run_dir: {run_dir}")
    index_lines.append("")
    index_lines.append("Layout:")
    index_lines.append("  gallery/<class>/<seed>/*.png")
    index_lines.append("")

    for cid in class_ids:
        class_dir = synth_root / cid
        seeds = _seed_dirs(class_dir)
        if not seeds:
            continue

        chosen = seeds[-max(1, int(spec.seed_limit)) :]
        for sd in chosen:
            pngs = _list_pngs(sd)[: spec.max_per_seed]
            if not pngs:
                continue

            target_dir = gallery_root / cid / sd.name
            target_dir.mkdir(parents=True, exist_ok=True)

            for p in pngs:
                dst = target_dir / p.name
                if dst.exists() and not cfg.overwrite:
                    continue
                try:
                    shutil.copy2(p, dst)
                    copied.append(dst)
                except Exception:
                    continue

            index_lines.append(f"class {cid} / seed {sd.name}: copied {len(pngs)} images")

    # Write index
    index_path = gallery_root / "INDEX.txt"
    if (not index_path.exists()) or cfg.overwrite:
        index_path.write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    written: List[Path] = []
    written.extend(copied)
    written.append(index_path)
    return written


__all__ = ["write_galleries"]
