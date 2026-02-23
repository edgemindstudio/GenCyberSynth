# src/gencysynth/reporting/plots/core/context.py
"""
PlotContext: a lightweight read-only view of run artifacts.

Why this exists
---------------
Plotting should be stable even when:
- You add new datasets
- You change where certain artifact files live
- You change the model family producing the run

So PlotContext:
- Resolves common artifact file paths under run_dir
- Loads JSON/YAML into dicts
- Provides small helpers for "best-effort" extraction

Rule A
------
This module MUST NOT write artifacts.
It only reads run artifacts and returns structured data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json

try:
    import yaml  # optional dependency; repo already uses yaml elsewhere
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists() or yaml is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


@dataclass(frozen=True)
class PlotContext:
    """
    Read-only view of a run.

    Attributes
    ----------
    run_dir:
        Root artifact directory for the run.
    run_manifest_path:
        Path to run_manifest file if found (json/yaml).
    eval_summary_path:
        Path to eval_summary file if found.
    run_events_path:
        Path to run_events file if found.

    run_manifest:
        Parsed dict (may be empty).
    eval_summary:
        Parsed dict (may be empty).
    run_events:
        Parsed dict (may be empty).
    """
    run_dir: Path
    run_manifest_path: Optional[Path]
    eval_summary_path: Optional[Path]
    run_events_path: Optional[Path]

    run_manifest: Dict[str, Any]
    eval_summary: Dict[str, Any]
    run_events: Dict[str, Any]

    def dataset_id(self) -> Optional[str]:
        """
        Best-effort dataset_id extraction (depends on your run_manifest contract).
        """
        # Common patterns
        for key in ("dataset_id", "data.dataset_id", "dataset.id"):
            v = _cfg_get(self.run_manifest, key, None)
            if v is not None:
                return str(v)
        return None

    def run_id(self) -> Optional[str]:
        """
        Best-effort run_id extraction (depends on your run_manifest contract).
        """
        for key in ("run_id", "run.run_id", "id"):
            v = _cfg_get(self.run_manifest, key, None)
            if v is not None:
                return str(v)
        return None


def _cfg_get(cfg: Dict[str, Any], dotted: str, default=None):
    """
    Dot-path getter for nested dicts (plot code uses this frequently).
    """
    cur: Any = cfg
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def resolve_run_manifest_path(run_dir: Path) -> Optional[Path]:
    """
    Locate the run manifest under run_dir.

    We support a few historical placements to keep plotting robust as the repo evolves.
    """
    candidates = [
        run_dir / "run_manifest.json",
        run_dir / "run_manifest.yaml",
        run_dir / "run_manifest.yml",
        run_dir / "manifest.json",
        run_dir / "manifest.yaml",
        run_dir / "manifest.yml",
    ]
    return _first_existing(candidates)


def resolve_eval_summary_path(run_dir: Path) -> Optional[Path]:
    """
    Locate eval_summary under run_dir.

    Your repo already has eval_summary.schema.json, so the file is typically named
    eval_summary.json somewhere in the run folder.
    """
    candidates = [
        run_dir / "eval_summary.json",
        run_dir / "eval" / "eval_summary.json",
        run_dir / "metrics" / "eval_summary.json",
        run_dir / "eval" / "summary.json",
        run_dir / "metrics" / "summary.json",
    ]
    return _first_existing(candidates)


def resolve_run_events_path(run_dir: Path) -> Optional[Path]:
    """
    Locate run_events under run_dir.

    Your repo already has run_events.schema.json, so the file is typically named
    run_events.json (or similar) and stored at run_dir or under logs/.
    """
    candidates = [
        run_dir / "run_events.json",
        run_dir / "events.json",
        run_dir / "logs" / "run_events.json",
        run_dir / "logs" / "events.json",
    ]
    return _first_existing(candidates)


def load_plot_context(run_dir: Path) -> PlotContext:
    """
    Load PlotContext from run_dir.

    This is intentionally tolerant:
    - If an artifact file is missing, the parsed dict will be empty.
    - Plot functions should gracefully skip what they cannot plot.
    """
    run_dir = Path(run_dir)

    rm_path = resolve_run_manifest_path(run_dir)
    es_path = resolve_eval_summary_path(run_dir)
    re_path = resolve_run_events_path(run_dir)

    run_manifest: Dict[str, Any] = {}
    if rm_path is not None:
        run_manifest = _read_json(rm_path) if rm_path.suffix == ".json" else _read_yaml(rm_path)

    eval_summary: Dict[str, Any] = {}
    if es_path is not None:
        eval_summary = _read_json(es_path)

    run_events: Dict[str, Any] = {}
    if re_path is not None:
        run_events = _read_json(re_path)

    return PlotContext(
        run_dir=run_dir,
        run_manifest_path=rm_path,
        eval_summary_path=es_path,
        run_events_path=re_path,
        run_manifest=run_manifest,
        eval_summary=eval_summary,
        run_events=run_events,
    )


__all__ = [
    "PlotContext",
    "load_plot_context",
    "resolve_run_manifest_path",
    "resolve_eval_summary_path",
    "resolve_run_events_path",
]
