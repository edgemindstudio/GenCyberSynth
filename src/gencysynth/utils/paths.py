# src/gencysynth/utils/paths.py
"""
GenCyberSynth — Path policy helpers (dataset-scalable)
======================================================

This module is the single source of truth for *where things live* on disk.

Scalability rule (Rule A)
-------------------------
Everything is keyed by (dataset_id, model_tag, run_id) so artifacts never collide:

  artifacts/
    runs/<dataset_id>/<model_tag>/<run_id>/...
    eval/<dataset_id>/<model_tag>/<run_id>/...
    logs/<dataset_id>/<model_tag>/<run_id>/...
    datasets/<dataset_id>/fingerprint.json      (dataset metadata)
    reports/<dataset_id>/<suite_id>/...         (suite-level reports)

Key responsibility
------------------
- Provide filesystem-safe slugging for keys used in paths.
- Provide small dataclasses that describe standard output locations.
- Make it easy to keep code consistent across models, orchestration, eval, and reporting.

Important
---------
You typically do NOT manually create these directories.
Your code should call ensure_dir(...) before writing, or call orchestration/context.py
with create_dirs=True.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union


# =============================================================================
# Core helpers
# =============================================================================
def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure a directory exists (parents=True) and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_slug(s: str, *, max_len: int = 120) -> str:
    """
    Make a filesystem-friendly slug.

    Keeps: alnum, '-', '_', '.', '/'
    Replaces everything else with '_'

    Notes
    -----
    - We keep '/' so model_tag can remain hierarchical (e.g., "gan/dcgan").
    - This is intentionally shared across all path resolvers.
    """
    if not isinstance(s, str):
        return "unknown"
    s = s.strip().replace(" ", "_")
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", ".", "/"):
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out).strip("_")
    return slug[:max_len] if slug else "unknown"


# =============================================================================
# 1) Evaluation outputs
# =============================================================================
@dataclass(frozen=True)
class EvalPaths:
    """
    Standard evaluation output locations.

    Layout:
      artifacts/eval/<dataset_id>/<model_tag>/<run_id>/
        - summary.jsonl   (append-only log, one record per evaluation call)
        - summary.txt     (human-readable console block)
        - latest.json     (optional; used by eval/runner.py)
    """
    root_dir: Path
    jsonl_path: Path
    console_path: Path
    latest_path: Path


def resolve_eval_paths(
    *,
    artifacts_root: Union[str, Path],
    dataset_id: str,
    model_tag: str,
    run_id: str,
) -> EvalPaths:
    """Resolve evaluation output paths under the dataset-scalable convention."""
    ar = Path(artifacts_root)
    ds = safe_slug(dataset_id)
    mt = safe_slug(model_tag)
    rid = safe_slug(run_id)

    root = ar / "eval" / ds / mt / rid
    return EvalPaths(
        root_dir=root,
        jsonl_path=root / "summary.jsonl",
        console_path=root / "summary.txt",
        latest_path=root / "latest.json",
    )


# =============================================================================
# 2) Run outputs (training + sampling)
# =============================================================================
@dataclass(frozen=True)
class RunPaths:
    """
    Standard per-run output locations.

    Layout:
      artifacts/runs/<dataset_id>/<model_tag>/<run_id>/
        - manifest.json           (required; written by sampling/orchestration)
        - provenance.json         (written by orchestration/provenance.py)
        - run_meta.json           (harvester snapshot: identity + results)
        - samples/                (recommended: generated images / arrays)
        - checkpoints/            (optional: ckpts, weights)
    """
    root_dir: Path
    samples_dir: Path
    checkpoints_dir: Path
    manifest_path: Path
    provenance_path: Path
    run_meta_path: Path


def resolve_run_paths(
    *,
    artifacts_root: Union[str, Path],
    dataset_id: str,
    model_tag: str,
    run_id: str,
) -> RunPaths:
    """Resolve run output paths under the dataset-scalable convention."""
    ar = Path(artifacts_root)
    ds = safe_slug(dataset_id)
    mt = safe_slug(model_tag)
    rid = safe_slug(run_id)

    root = ar / "runs" / ds / mt / rid
    return RunPaths(
        root_dir=root,
        samples_dir=root / "samples",
        checkpoints_dir=root / "checkpoints",
        manifest_path=root / "manifest.json",
        provenance_path=root / "provenance.json",
        run_meta_path=root / "run_meta.json",
    )


# =============================================================================
# 3) Run logs (text + structured JSONL)
# =============================================================================
@dataclass(frozen=True)
class LogsPaths:
    """
    Standard run-scoped log locations.

    Layout:
      artifacts/logs/<dataset_id>/<model_tag>/<run_id>/
        - run.log
        - events.jsonl
    """
    root_dir: Path
    log_path: Path
    events_path: Path


def resolve_logs_paths(
    *,
    artifacts_root: Union[str, Path],
    dataset_id: str,
    model_tag: str,
    run_id: str,
) -> LogsPaths:
    """Resolve per-run log output paths under the dataset-scalable convention."""
    ar = Path(artifacts_root)
    ds = safe_slug(dataset_id)
    mt = safe_slug(model_tag)
    rid = safe_slug(run_id)

    root = ar / "logs" / ds / mt / rid
    return LogsPaths(
        root_dir=root,
        log_path=root / "run.log",
        events_path=root / "events.jsonl",
    )


__all__ = [
    "ensure_dir",
    "safe_slug",
    "EvalPaths",
    "resolve_eval_paths",
    "RunPaths",
    "resolve_run_paths",
    "LogsPaths",
    "resolve_logs_paths",
]
