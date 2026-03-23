# src/gencysynth/data/fingerprint_writer.py
"""
gencysynth.data.fingerprint_writer
==================================

Thin convenience layer to:
  1) compute a dataset fingerprint (quartet_style or arbitrary files)
  2) write it to the standardized, dataset_scalable artifacts location

This keeps orchestration/CLI code clean:
    fp_path = write_dataset_fingerprint(cfg, dataset_meta)

Where:
- cfg is your loaded config dict (expects cfg["paths"]["artifacts"] optionally)
- dataset_meta contains dataset_id + dataset_root (+ optional file names)

This module intentionally depends only on:
- gencysynth.data.fingerprint (fingerprint computation)
- gencysynth.utils.io (safe JSON writing)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

from gencysynth.data.fingerprint import (
    default_fingerprint_output_path,
    fingerprint_dataset_files,
    fingerprint_npy_quartet,
)
from gencysynth.utils.io import write_json


__all__ = [
    "write_dataset_fingerprint",
    "write_dataset_fingerprint_quartet",
    "write_dataset_fingerprint_files",
]


def _cfg_get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    """Small dotted_path getter (kept local so eval/runner doesn't have to import it)."""
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def write_dataset_fingerprint(
    cfg: Dict[str, Any],
    dataset_meta: Dict[str, Any],
    *,
    compute_full_hash: bool = True,
    compute_quick_hash: bool = True,
    out_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    One_call helper used by loaders/orchestration to persist dataset fingerprint.

    Parameters
    ----------
    cfg:
        Loaded config dict. We read:
          - cfg["paths"]["artifacts"] (default "artifacts")
    dataset_meta:
        A dict with at least:
          - dataset_id: stable identifier, e.g. "USTC_TFC2016_40x40_gray"
          - dataset_root: directory where data files live

        Optional quartet overrides:
          - train_data, train_labels, test_data, test_labels (filenames)
        Optional explicit file list:
          - files: list[str|Path] of files to fingerprint (relative to dataset_root or absolute)

        Rule:
          - If dataset_meta["files"] is provided => fingerprint those
          - Else => assume quartet_style .npy dataset
    compute_full_hash:
        Full sha256 for each file (strong audit; slower on huge files).
    compute_quick_hash:
        Quick hash for large files (helpful sanity signature).
    out_path:
        If provided, writes exactly here. Otherwise uses:
          {artifacts_root}/data/{dataset_id}/dataset_fingerprint.json

    Returns
    -------
    Path
        Path to the written fingerprint JSON.
    """
    dataset_id = str(dataset_meta.get("dataset_id") or "unknown_dataset")
    dataset_root = dataset_meta.get("dataset_root")
    if not dataset_root:
        raise ValueError("dataset_meta must include 'dataset_root'.")

    artifacts_root = _cfg_get(cfg, "paths.artifacts", "artifacts")

    # Decide output location
    out = Path(out_path) if out_path is not None else default_fingerprint_output_path(
        artifacts_root=artifacts_root,
        dataset_id=dataset_id,
    )

    # Decide fingerprint mode
    files = dataset_meta.get("files", None)
    if isinstance(files, (list, tuple)) and len(files) > 0:
        # Arbitrary_layout datasets
        fp = fingerprint_dataset_files(
            dataset_id=dataset_id,
            dataset_root=dataset_root,
            files=files,  # can be absolute or relative
            compute_full_hash=compute_full_hash,
            compute_quick_hash=compute_quick_hash,
        )
    else:
        # Quartet_style datasets (default GenCyberSynth format)
        fp = fingerprint_npy_quartet(
            dataset_id=dataset_id,
            dataset_root=dataset_root,
            train_data=str(dataset_meta.get("train_data") or "train_data.npy"),
            train_labels=str(dataset_meta.get("train_labels") or "train_labels.npy"),
            test_data=str(dataset_meta.get("test_data") or "test_data.npy"),
            test_labels=str(dataset_meta.get("test_labels") or "test_labels.npy"),
            compute_full_hash=compute_full_hash,
            compute_quick_hash=compute_quick_hash,
        )

    # Write fingerprint JSON (atomic)
    write_json(out, fp, indent=2, sort_keys=True, atomic=True)
    return out


def write_dataset_fingerprint_quartet(
    cfg: Dict[str, Any],
    *,
    dataset_id: str,
    dataset_root: Union[str, Path],
    train_data: str = "train_data.npy",
    train_labels: str = "train_labels.npy",
    test_data: str = "test_data.npy",
    test_labels: str = "test_labels.npy",
    compute_full_hash: bool = True,
    compute_quick_hash: bool = True,
    out_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Explicit quartet_only wrapper (useful when you don't want dataset_meta dicts).
    """
    meta = {
        "dataset_id": dataset_id,
        "dataset_root": str(dataset_root),
        "train_data": train_data,
        "train_labels": train_labels,
        "test_data": test_data,
        "test_labels": test_labels,
    }
    return write_dataset_fingerprint(
        cfg,
        meta,
        compute_full_hash=compute_full_hash,
        compute_quick_hash=compute_quick_hash,
        out_path=out_path,
    )


def write_dataset_fingerprint_files(
    cfg: Dict[str, Any],
    *,
    dataset_id: str,
    dataset_root: Union[str, Path],
    files: Iterable[Union[str, Path]],
    compute_full_hash: bool = True,
    compute_quick_hash: bool = True,
    out_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Explicit arbitrary_files wrapper.
    """
    meta = {
        "dataset_id": dataset_id,
        "dataset_root": str(dataset_root),
        "files": list(files),
    }
    return write_dataset_fingerprint(
        cfg,
        meta,
        compute_full_hash=compute_full_hash,
        compute_quick_hash=compute_quick_hash,
        out_path=out_path,
    )
