# src/gencysynth/orchestration/manifest.py
"""
GenCyberSynth — Synthetic Run Manifest
=====================================

A synthetic generation run should produce a manifest describing exactly what was
generated. This makes evaluation and aggregation robust and auditable.

Manifest location (Rule A)
--------------------------
  artifacts/runs/<dataset_id>/<model_tag>/<run_id>/manifest.json

Manifest format (high-level)
----------------------------
{
  "schema_version": "run_manifest_v2",
  "timestamp": "...",
  "dataset": {"id": "..."},
  "model": {"tag": "..."},
  "run": {"run_id": "...", "seed": 42},
  "paths": [
     {"path": ".../img_000001.png", "label": 3, "split": "synth"},
     ...
  ],
  "per_class_counts": {"0": 1000, "1": 1000, ...},
  "notes": {...}
}

This module provides:
- building/updating the manifest while sampling
- writing it safely using gencysynth.utils.io.write_json
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from gencysynth.utils.io import write_json
from gencysynth.utils.reproducibility import now_iso


@dataclass
class ManifestSample:
    """One generated sample record."""
    path: str
    label: int
    split: str = "synth"
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunManifest:
    """
    In-memory manifest object.

    Notes:
    - per_class_counts should be strings for JSON friendliness (keys are class ids).
    """
    schema_version: str = "run_manifest_v2"
    timestamp: str = field(default_factory=now_iso)

    dataset: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)
    run: Dict[str, Any] = field(default_factory=dict)

    paths: List[ManifestSample] = field(default_factory=list)
    per_class_counts: Dict[str, int] = field(default_factory=dict)

    notes: Dict[str, Any] = field(default_factory=dict)

    def add(self, path: str, label: int, *, split: str = "synth", meta: Optional[Dict[str, Any]] = None) -> None:
        """Append one sample and update per-class counts."""
        self.paths.append(ManifestSample(path=str(path), label=int(label), split=str(split), meta=meta or {}))
        k = str(int(label))
        self.per_class_counts[k] = int(self.per_class_counts.get(k, 0)) + 1

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # dataclasses -> dict will serialize ManifestSample fine
        return d


def new_manifest(
    *,
    dataset_id: str,
    model_tag: str,
    run_id: str,
    seed: int,
    extra: Optional[Dict[str, Any]] = None,
) -> RunManifest:
    """Create a new RunManifest with required identity fields set."""
    m = RunManifest()
    m.dataset = {"id": str(dataset_id)}
    m.model = {"tag": str(model_tag)}
    m.run = {"run_id": str(run_id), "seed": int(seed)}
    if extra:
        m.notes.update(extra)
    return m


def write_manifest(path: Path, manifest: RunManifest) -> Path:
    """
    Write manifest to disk (atomic overwrite).
    Caller ensures parent directory exists.
    """
    return write_json(Path(path), manifest.to_dict(), indent=2, sort_keys=True, atomic=True)


__all__ = ["ManifestSample", "RunManifest", "new_manifest", "write_manifest"]
