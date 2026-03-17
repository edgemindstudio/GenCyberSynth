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
from dataclasses import dataclass

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


@dataclass(frozen=True)
class RunIO:
    """
    Convenience wrapper holding canonical Rule_A paths for a single run.

    This keeps adapter code clean:
      io = RunIO(artifacts_root, dataset_id, model_tag, run_id)
      io.ensure_dirs()
      io.run_paths.manifest_path ...
    """
    artifacts_root: PathLike
    dataset_id: str
    model_tag: str
    run_id: str

    @property
    def run_paths(self) -> RunPaths:
        return resolve_run_paths(
            artifacts_root=self.artifacts_root,
            dataset_id=self.dataset_id,
            model_tag=self.model_tag,
            run_id=self.run_id,
        )

    @property
    def logs_paths(self) -> LogsPaths:
        return resolve_logs_paths(
            artifacts_root=self.artifacts_root,
            dataset_id=self.dataset_id,
            model_tag=self.model_tag,
            run_id=self.run_id,
        )

    @property
    def eval_paths(self) -> EvalPaths:
        return resolve_eval_paths(
            artifacts_root=self.artifacts_root,
            dataset_id=self.dataset_id,
            model_tag=self.model_tag,
            run_id=self.run_id,
        )

    def ensure_dirs(self, *, include_eval: bool = True) -> None:
        """Create standard directories for this run."""
        ensure_run_dirs(
            self.run_paths,
            self.logs_paths,
            self.eval_paths if include_eval else None,
        )

    @classmethod
    def from_run_ctx(cls, run_ctx) -> "RunIO":
        """
        Best_effort construction from a run_ctx object.

        Expected attributes on run_ctx (common patterns):
          - artifacts_root OR artifacts
          - dataset_id
          - model_tag
          - run_id
        """
        # Try common attribute names
        artifacts_root = getattr(run_ctx, "artifacts_root", None) or getattr(run_ctx, "artifacts", None)
        dataset_id = getattr(run_ctx, "dataset_id", None)
        model_tag = getattr(run_ctx, "model_tag", None)
        run_id = getattr(run_ctx, "run_id", None)

        missing = [k for k, v in {
            "artifacts_root": artifacts_root,
            "dataset_id": dataset_id,
            "model_tag": model_tag,
            "run_id": run_id,
        }.items() if not v]

        if missing:
            raise ValueError(f"RunIO.from_run_ctx missing fields on run_ctx: {missing}")

        return cls(
            artifacts_root=artifacts_root,
            dataset_id=str(dataset_id),
            model_tag=str(model_tag),
            run_id=str(run_id),
        )

__all__ = ["RunIO",
           "resolve_run_paths",
           "resolve_eval_paths",
           "resolve_logs_paths",
           "ensure_run_dirs"]