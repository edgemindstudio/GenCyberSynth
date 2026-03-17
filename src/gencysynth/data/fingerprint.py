# src/gencysynth/data/fingerprint.py
"""
gencysynth.data.fingerprint
===========================

Dataset fingerprinting utilities for GenCyberSynth.

Why fingerprint datasets?
-------------------------
When GenCyberSynth scales to multiple datasets (and multiple storage locations),
we need an audit_proof way to say:

- exactly which files were used
- basic file metadata (size, modified time)
- strong content identifiers (sha256)
- lightweight array metadata (shape/dtype) when possible

This supports:
- reproducibility on HPC
- reviewer/auditor trust
- consistent artifacts layout per dataset

Output format
-------------
The functions here return a Python dict that can be JSON_serialized.
It is designed to align with your schema:
  gencysynth/schemas/dataset_fingerprint.schema.json

No heavy deps:
-------------
- Uses hashlib + pathlib + os only.
- Optionally reads NumPy headers to capture shape/dtype without loading full arrays.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

__all__ = [
    "FileFingerprint",
    "fingerprint_file",
    "fingerprint_dataset_files",
    "fingerprint_npy_quartet",
    "default_fingerprint_output_path",
]


# -----------------------------------------------------------------------------
# Small data container (returned as dict via .asdict()).
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class FileFingerprint:
    """
    Fingerprint for a single file.

    Attributes
    ----------
    path:
        Path as string (stored as provided; typically absolute or dataset_root_relative).
    exists:
        Whether the file exists at time of fingerprinting.
    size_bytes:
        File size in bytes, if exists.
    mtime_epoch:
        Modification time (epoch seconds), if exists.
    sha256:
        Full_file sha256 (strong content ID), if exists.
    quick_sha256:
        Optional short hash based on first+last blocks (for very large files).
        This is NOT a replacement for sha256, but can be used as a quick check.
    npy:
        Optional dict with NumPy header info (dtype, shape, fortran_order) if file is .npy
        and numpy is available.
    """
    path: str
    exists: bool
    size_bytes: Optional[int]
    mtime_epoch: Optional[float]
    sha256: Optional[str]
    quick_sha256: Optional[str]
    npy: Optional[Dict[str, Any]]


# -----------------------------------------------------------------------------
# Hashing helpers
# -----------------------------------------------------------------------------
def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """
    Compute full sha256 of a file by streaming it.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(int(chunk_size))
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _quick_sha256_file(path: Path, *, block_size: int = 1024 * 1024) -> str:
    """
    Compute a "quick" sha256 based on first + last blocks.

    Useful when:
      - files are huge and full hashing is expensive
      - you still want a cheap sanity signature

    NOTE: This is not cryptographically equivalent to hashing the full file.
    """
    size = path.stat().st_size
    h = hashlib.sha256()

    with path.open("rb") as f:
        head = f.read(int(block_size))
        h.update(head)

        if size > block_size:
            f.seek(max(0, size - block_size))
            tail = f.read(int(block_size))
            h.update(tail)

    # Include size so different_length files don't collide as easily
    h.update(str(size).encode("utf_8"))
    return h.hexdigest()


def _try_read_npy_header(path: Path) -> Optional[Dict[str, Any]]:
    """
    Best_effort read of NumPy .npy header to capture dtype/shape without loading.

    Returns None if:
      - numpy isn't available
      - file isn't a valid .npy
    """
    if path.suffix.lower() != ".npy":
        return None

    try:
        import numpy as np  # optional dependency
        # This loads header and memmaps the data without reading everything
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        return {
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "ndim": int(arr.ndim),
        }
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Single_file fingerprint
# -----------------------------------------------------------------------------
def fingerprint_file(
    file_path: Union[str, Path],
    *,
    root: Optional[Union[str, Path]] = None,
    compute_full_hash: bool = True,
    compute_quick_hash: bool = True,
    quick_hash_threshold_bytes: int = 512 * 1024 * 1024,  # 512MB
) -> Dict[str, Any]:
    """
    Fingerprint a single file.

    Parameters
    ----------
    file_path:
        Path to file.
    root:
        If provided, the returned "path" is stored relative to this root (when possible).
        This makes fingerprints portable across machines.
    compute_full_hash:
        If True, compute full sha256 (strongest).
    compute_quick_hash:
        If True, compute quick_sha256 (first+last blocks) for huge files.
    quick_hash_threshold_bytes:
        Only compute quick hash when size >= threshold (unless full hash disabled).

    Returns
    -------
    dict
        JSON_serializable dict containing file fingerprint info.
    """
    p = Path(file_path)
    root_p = Path(root) if root is not None else None

    # Store relative path when root is provided and applicable
    stored_path: str
    try:
        stored_path = str(p.relative_to(root_p)) if root_p is not None else str(p)
    except Exception:
        stored_path = str(p)

    if not p.exists():
        ff = FileFingerprint(
            path=stored_path,
            exists=False,
            size_bytes=None,
            mtime_epoch=None,
            sha256=None,
            quick_sha256=None,
            npy=None,
        )
        return ff.__dict__

    st = p.stat()
    size = int(st.st_size)
    mtime = float(st.st_mtime)

    sha256 = None
    quick_sha256 = None

    # Full hash (strongest, but can be slow on huge datasets)
    if compute_full_hash:
        sha256 = _sha256_file(p)

    # Quick hash (optional, usually only for very large files)
    if compute_quick_hash and (not compute_full_hash or size >= int(quick_hash_threshold_bytes)):
        quick_sha256 = _quick_sha256_file(p)

    npy_meta = _try_read_npy_header(p)

    ff = FileFingerprint(
        path=stored_path,
        exists=True,
        size_bytes=size,
        mtime_epoch=mtime,
        sha256=sha256,
        quick_sha256=quick_sha256,
        npy=npy_meta,
    )
    return ff.__dict__


# -----------------------------------------------------------------------------
# Dataset_level fingerprints
# -----------------------------------------------------------------------------
def fingerprint_dataset_files(
    *,
    dataset_id: str,
    dataset_root: Union[str, Path],
    files: Iterable[Union[str, Path]],
    compute_full_hash: bool = True,
    compute_quick_hash: bool = True,
    created_at_epoch: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Fingerprint an arbitrary set of dataset files.

    This is the most general entrypoint and works for any dataset layout.

    Returns:
      {
        "dataset_id": ...,
        "dataset_root": ...,
        "created_at_epoch": ...,
        "files": [ FileFingerprintDict, ... ],
      }
    """
    root = Path(dataset_root)
    created_at_epoch = float(created_at_epoch) if created_at_epoch is not None else time.time()

    fps: List[Dict[str, Any]] = []
    for f in files:
        fp = fingerprint_file(
            f,
            root=root,
            compute_full_hash=compute_full_hash,
            compute_quick_hash=compute_quick_hash,
        )
        fps.append(fp)

    # Sort to keep output stable for diffs and audits
    fps = sorted(fps, key=lambda d: str(d.get("path", "")))

    return {
        "dataset_id": str(dataset_id),
        "dataset_root": str(root),
        "created_at_epoch": created_at_epoch,
        "files": fps,
    }


def fingerprint_npy_quartet(
    *,
    dataset_id: str,
    dataset_root: Union[str, Path],
    train_data: str = "train_data.npy",
    train_labels: str = "train_labels.npy",
    test_data: str = "test_data.npy",
    test_labels: str = "test_labels.npy",
    compute_full_hash: bool = True,
    compute_quick_hash: bool = True,
) -> Dict[str, Any]:
    """
    Convenience helper for the standard GenCyberSynth classification dataset quartet.

    This matches the most common format (USTC_TFC2016_style):
      train_data.npy / train_labels.npy / test_data.npy / test_labels.npy
    """
    root = Path(dataset_root)
    files = [
        root / train_data,
        root / train_labels,
        root / test_data,
        root / test_labels,
    ]
    return fingerprint_dataset_files(
        dataset_id=dataset_id,
        dataset_root=root,
        files=files,
        compute_full_hash=compute_full_hash,
        compute_quick_hash=compute_quick_hash,
    )


# -----------------------------------------------------------------------------
# Artifacts path helper (dataset_scalable)
# -----------------------------------------------------------------------------
def default_fingerprint_output_path(
    *,
    artifacts_root: Union[str, Path] = "artifacts",
    dataset_id: str,
) -> Path:
    """
    Default location for dataset fingerprint record in your scalable artifacts layout:

      {artifacts_root}/data/{dataset_id}/dataset_fingerprint.json

    Why:
    - keeps dataset_wide artifacts together
    - separates them cleanly from per_run eval outputs
    """
    ar = Path(artifacts_root)
    ds = str(dataset_id).strip() or "unknown_dataset"
    out_dir = ar / "data" / ds
    return out_dir / "dataset_fingerprint.json"
