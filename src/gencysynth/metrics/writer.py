# src/gencysynth/metrics/writer.py
"""
Metrics artifact writer.

This is intentionally boring and predictable:
- One directory per (dataset_id, run_id)
- JSON summaries and optional JSONL event log
- Paths returned to the caller so orchestrators can reference them

Rule A layout
-------------
{paths.artifacts}/metrics/<dataset_id>/<run_id>/
  summary.json
  events.jsonl
  by_metric/<metric_name>.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .types import MetricResult


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class MetricsPaths:
    root: Path
    summary_json: Path
    events_jsonl: Path
    by_metric_dir: Path


def resolve_metrics_paths(*, artifacts_root: Path, dataset_id: str, run_id: str) -> MetricsPaths:
    root = artifacts_root / "metrics" / str(dataset_id) / str(run_id)
    by_metric = root / "by_metric"
    return MetricsPaths(
        root=root,
        summary_json=root / "summary.json",
        events_jsonl=root / "events.jsonl",
        by_metric_dir=by_metric,
    )


def write_metric_result(paths: MetricsPaths, result: MetricResult) -> Path:
    """
    Write a single metric result under by_metric/<name>.json.
    """
    ensure_dir(paths.by_metric_dir)
    out = paths.by_metric_dir / f"{result.name}.json"
    out.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf_8")
    return out


def write_summary(paths: MetricsPaths, payload: Dict[str, Any]) -> Path:
    """
    Write summary.json.
    """
    ensure_dir(paths.root)
    paths.summary_json.write_text(json.dumps(payload, indent=2), encoding="utf_8")
    return paths.summary_json


def append_event(paths: MetricsPaths, event: Dict[str, Any]) -> None:
    """
    Append a single event to events.jsonl.
    """
    ensure_dir(paths.root)
    with open(paths.events_jsonl, "a", encoding="utf_8") as f:
        f.write(json.dumps(event) + "\n")