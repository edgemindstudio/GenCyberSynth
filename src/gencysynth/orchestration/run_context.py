# src/gencysynth/orchestration/run_context.py
"""
GenCyberSynth — RunContext (where to write)
=========================================

A RunContext is the resolved, filesystem_aware counterpart to RunSpec.

RunSpec answers: "what to run?"
RunContext answers: "where does this run read/write artifacts/logs/eval?"

Rule A
------
Everything is keyed by (dataset_id, model_tag, run_id)

So outputs never collide:

  artifacts/
    runs/<dataset_id>/<model_tag>/<run_id>/...
    eval/<dataset_id>/<model_tag>/<run_id>/...
    logs/<dataset_id>/<model_tag>/<run_id>/...

This module centralizes the directory creation + convenience accessors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from gencysynth.models.base_types import RunContext as _RunContext  # canonical type
from gencysynth.utils.paths import ensure_dir


@dataclass(frozen=True)
class RunDirs:
    """Convenience wrapper around run/log/eval directories."""
    run_dir: Path
    logs_dir: Path
    eval_dir: Path


def ensure_dirs(ctx: _RunContext) -> RunDirs:
    """
    Ensure run/log/eval directories exist for this context.
    Returns a simple RunDirs structure for convenience.
    """
    if ctx.run_dir is None or ctx.logs_dir is None or ctx.eval_dir is None:
        raise ValueError("RunContext is missing one or more directories (run_dir/logs_dir/eval_dir).")

    ensure_dir(ctx.run_dir)
    ensure_dir(ctx.logs_dir)
    ensure_dir(ctx.eval_dir)

    return RunDirs(run_dir=ctx.run_dir, logs_dir=ctx.logs_dir, eval_dir=ctx.eval_dir)


__all__ = ["RunDirs", "ensure_dirs"]
