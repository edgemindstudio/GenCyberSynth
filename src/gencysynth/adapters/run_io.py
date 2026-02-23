# src/gencysynth/adapters/run_io.py

"""
Canonical artifact path helpers for adapters.

This is intentionally a THIN wrapper over gencysynth.utils.paths
so adapter code reads cleanly and stays consistent.

Rule A
------
Paths are always resolved using (dataset_id, model_tag, run_id), so multiple
datasets and multiple variants never collide.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from gencysynth.utils.paths import (
    ensure_dir,
    resolve_eval_paths as _resolve_eval_paths,
    resolve_logs_paths as _resolve_logs_paths,
    resolve_run_paths as _resolve_run_paths,
    EvalPaths,
    LogsPaths,
    RunPaths,
)


PathLike = Union[str, Path]


def resolve_run_paths(*, artifacts_root: PathLike, dataset_id: str, model_tag: str, run_id: str) -> RunPaths:
    return _resolve_run_paths(
        artifacts_root=artifacts_root,
        dataset_id=dataset_id,
        model_tag=model_tag,
        run_id=run_id,
    )


def resolve_eval_paths(*, artifacts_root: PathLike, dataset_id: str, model_tag: str, run_id: str) -> EvalPaths:
    return _resolve_eval_paths(
        artifacts_root=artifacts_root,
        dataset_id=dataset_id,
        model_tag=model_tag,
        run_id=run_id,
    )


def resolve_logs_paths(*, artifacts_root: PathLike, dataset_id: str, model_tag: str, run_id: str) -> LogsPaths:
    return _resolve_logs_paths(
        artifacts_root=artifacts_root,
        dataset_id=dataset_id,
        model_tag=model_tag,
        run_id=run_id,
    )


def ensure_run_dirs(run_paths: RunPaths, logs_paths: LogsPaths, eval_paths: EvalPaths | None = None) -> None:
    """
    Create standard directories for a run.

    Notes
    -----
    - We only create directories; files are created by writers.
    - Keeping this centralized prevents subtle layout drift across model families.
    """
    ensure_dir(run_paths.root_dir)
    ensure_dir(run_paths.samples_dir)
    ensure_dir(run_paths.checkpoints_dir)

    ensure_dir(logs_paths.root_dir)

    if eval_paths is not None:
        ensure_dir(eval_paths.root_dir)
