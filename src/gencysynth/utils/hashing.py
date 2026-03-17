# src/gencysynth/utils/hashing.py
"""
GenCyberSynth — Hashing utilities (Rule A friendly)

Purpose
-------
This module centralizes hashing primitives used across the repository for:
- dataset fingerprinting
- run manifest stability
- artifact integrity checks
- reproducibility reports

Design constraints
------------------
- Deterministic across machines: avoid non_deterministic ordering.
- Stable serialization: use canonical JSON for dicts/lists.
- Friendly to large directories: allow excludes and file extensions filters.

Notes
-----
You already have a light hashing implementation in `utils/reproducibility.py`.
This module is a more general “public” surface intended to be imported widely.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# -----------------------------
# Core hashing primitives
# -----------------------------
def sha256_bytes(data: bytes) -> str:
    """SHA256 hex digest of raw bytes."""
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def sha256_text(text: str, *, encoding: str = "utf_8") -> str:
    """SHA256 hex digest of a text string."""
    return sha256_bytes(text.encode(encoding))


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """
    SHA256 hex digest of a file, streamed in chunks.
    Safe for large files.
    """
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


# -----------------------------
# Canonical JSON hashing
# -----------------------------
def canonical_json_dumps(obj: Any) -> str:
    """
    Canonical JSON serialization (stable):
    - sorted keys
    - no extra whitespace
    - UTF_8 safe
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(obj: Any) -> str:
    """SHA256 of canonical JSON representation of an object."""
    return sha256_text(canonical_json_dumps(obj))


# -----------------------------
# Directory hashing (deterministic walk)
# -----------------------------
@dataclass(frozen=True)
class DirHashOptions:
    """
    Options for directory hashing.
    """
    exclude_dirs: Tuple[str, ...] = (".git", "__pycache__", "artifacts", "logs")
    exclude_suffixes: Tuple[str, ...] = (".pyc", ".pyo", ".DS_Store")
    include_extensions: Optional[Tuple[str, ...]] = None  # e.g. (".py",".yaml",".json")
    follow_symlinks: bool = False


def sha256_dir(
    root: Path,
    *,
    opts: DirHashOptions = DirHashOptions(),
) -> str:
    """
    Deterministically hash directory contents.

    The resulting digest depends on:
    - file relative paths
    - file bytes (sha256 per file)
    - stable ordering across platforms
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    entries: List[Tuple[str, str]] = []

    for dirpath, dirnames, filenames in os.walk(root, followlinks=opts.follow_symlinks):
        # Sort and prune directories deterministically
        dirnames.sort()
        dirnames[:] = [
            d for d in dirnames
            if d not in opts.exclude_dirs
        ]

        filenames.sort()
        for fn in filenames:
            if any(fn.endswith(suf) for suf in opts.exclude_suffixes):
                continue
            if opts.include_extensions is not None:
                if not any(fn.endswith(ext) for ext in opts.include_extensions):
                    continue

            fpath = Path(dirpath) / fn
            rel = fpath.relative_to(root).as_posix()

            # Skip if the file is not a regular file (just in case)
            if not fpath.is_file():
                continue

            entries.append((rel, sha256_file(fpath)))

    # Combine into a stable manifest and hash it
    manifest = [{"path": p, "sha256": h} for (p, h) in entries]
    return sha256_json(manifest)