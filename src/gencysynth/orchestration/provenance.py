# src/gencysynth/orchestration/provenance.py
"""
GenCyberSynth — Provenance capture (audit trail)
================================================

This module records the "who/what/when/where" of a run:
  - config paths and/or config content hash
  - git commit (best_effort)
  - host/user/pid
  - timestamps
  - environment hints (python version)

This is designed for HPC reproducibility and later forensic debugging.

Where it writes
---------------
Under the run directory:

  artifacts/runs/<dataset_id>/<model_tag>/<run_id>/
    provenance.json
"""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from gencysynth.utils.io import write_json
from gencysynth.utils.reproducibility import now_iso


def _git_head_commit(repo_root: Optional[Path] = None) -> Optional[str]:
    """Best_effort git commit hash (returns None if unavailable)."""
    try:
        cmd = ["git", "rev_parse", "HEAD"]
        out = subprocess.check_output(cmd, cwd=str(repo_root) if repo_root else None, stderr=subprocess.DEVNULL)
        s = out.decode("utf_8").strip()
        return s or None
    except Exception:
        return None


@dataclass(frozen=True)
class ProvenanceRecord:
    """Structured provenance record stored per run."""
    timestamp: str
    user: Optional[str]
    host: str
    pid: int
    python: str
    platform: str

    git_commit: Optional[str] = None
    config_path: Optional[str] = None
    overrides_path: Optional[str] = None
    notes: Optional[str] = None

    # Optional: run identity keys for quick grepping
    dataset_id: Optional[str] = None
    model_tag: Optional[str] = None
    run_id: Optional[str] = None
    seed: Optional[int] = None


def build_provenance(
    *,
    dataset_id: str,
    model_tag: str,
    run_id: str,
    seed: int,
    config_path: Optional[str] = None,
    overrides_path: Optional[str] = None,
    repo_root: Optional[Path] = None,
    notes: Optional[str] = None,
) -> ProvenanceRecord:
    """Build a provenance record (pure; does not write)."""
    return ProvenanceRecord(
        timestamp=now_iso(),
        user=os.getenv("USER") or os.getenv("USERNAME"),
        host=platform.node(),
        pid=os.getpid(),
        python=platform.python_version(),
        platform=platform.platform(),
        git_commit=_git_head_commit(repo_root),
        config_path=config_path,
        overrides_path=overrides_path,
        notes=notes,
        dataset_id=dataset_id,
        model_tag=model_tag,
        run_id=run_id,
        seed=int(seed),
    )


def write_provenance(run_dir: Path, prov: ProvenanceRecord) -> Path:
    """Write provenance.json under run_dir."""
    out = Path(run_dir) / "provenance.json"
    return write_json(out, asdict(prov), indent=2, sort_keys=True, atomic=True)


__all__ = ["ProvenanceRecord", "build_provenance", "write_provenance"]
