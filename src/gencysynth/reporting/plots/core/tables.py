# src/gencysynth/reporting/plots/core/tables.py
"""
Render evaluation metric tables to images (optional but very handy).

This is a visualization_only layer:
- It reads eval_summary
- It renders a small table image (PNG) under reporting/plots/core/

Rule A
------
Reads:
  - ctx.eval_summary
Writes:
  - <run_dir>/reporting/plots/core/metric_table.png (and other formats)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt

from ..config import PlotConfig
from .context import PlotContext


def _extract_table_rows(eval_summary: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Convert eval_summary into a simple 2_column table:
      metric_name | value

    We intentionally keep it generic:
    - flatten numeric values from common containers and top_level
    """
    rows: List[Tuple[str, str]] = []

    # Containers first (stable order)
    for container_key in ("metrics", "summary", "scores", "results"):
        v = eval_summary.get(container_key, None)
        if isinstance(v, dict):
            for k, val in v.items():
                if isinstance(val, (int, float, str)):
                    rows.append((f"{container_key}.{k}", str(val)))

    # Then top_level numeric/string values
    for k, val in eval_summary.items():
        if isinstance(val, (int, float, str)):
            rows.append((str(k), str(val)))

    # Deduplicate while preserving order
    seen = set()
    uniq: List[Tuple[str, str]] = []
    for a, b in rows:
        if a in seen:
            continue
        seen.add(a)
        uniq.append((a, b))
    return uniq


def render_metric_tables(ctx: PlotContext, out_dir: Path, cfg: PlotConfig) -> List[Path]:
    """
    Render a simple metric table image.

    Returns [] if eval_summary isn't present or yields no rows.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _extract_table_rows(ctx.eval_summary)
    if not rows:
        return []

    # Keep the table readable: cap row count.
    rows = rows[:40]

    fig = plt.figure(figsize=(10, 0.35 * (len(rows) + 2)))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")

    cell_text = [[a, b] for (a, b) in rows]
    table = ax.table(
        cellText=cell_text,
        colLabels=["metric", "value"],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.2)

    written: List[Path] = []
    for ext in cfg.formats:
        p = out_dir / f"metric_table.{ext}"
        if p.exists() and not cfg.overwrite:
            continue
        fig.savefig(p)
        written.append(p)

    plt.close(fig)
    return written


__all__ = ["render_metric_tables"]
