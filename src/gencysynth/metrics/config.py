# src/gencysynth/metrics/config.py
"""
Metrics configuration and Rule A artifact normalization.

This module is the bridge between "global repo config" and the metrics package.

We keep the contract minimal:
- Read dataset identity from cfg["data"]["id"] (fallback: cfg["data"]["root"] / cfg["DATA_DIR"])
- Read artifacts root from cfg["paths"]["artifacts"] (fallback: "artifacts")
- Decide which metrics to run from cfg["metrics"]["enabled"] (fallback to defaults.yaml list)
- Provide per-metric options under cfg["metrics"]["options"][<metric_name>]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _cfg_get(cfg: Dict, dotted: str, default=None):
    cur = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def normalize_dataset_id(cfg: Dict) -> str:
    """
    Prefer stable cfg.data.id. Fallback to a filesystem-safe transform of cfg.data.root.
    """
    dataset_id = _cfg_get(cfg, "data.id", None)
    if dataset_id:
        return str(dataset_id)

    root = str(_cfg_get(cfg, "data.root", _cfg_get(cfg, "DATA_DIR", "data")))
    return root.replace("/", "_").replace("\\", "_").replace(" ", "_")


def normalize_run_id(cfg: Dict) -> str:
    """
    Prefer cfg.run.id, else cfg.run_id, else a conservative fallback "run".
    (The orchestrator should ideally set this.)
    """
    rid = _cfg_get(cfg, "run.id", None)
    if rid:
        return str(rid)
    rid = cfg.get("run_id", None)
    if rid:
        return str(rid)
    return "run"


def artifacts_root(cfg: Dict) -> Path:
    return Path(_cfg_get(cfg, "paths.artifacts", "artifacts"))


def enabled_metrics(cfg: Dict, defaults: List[str]) -> List[str]:
    """
    Decide which metrics to run. If cfg.metrics.enabled missing, use defaults.
    """
    enabled = _cfg_get(cfg, "metrics.enabled", None)
    if enabled is None:
        return list(defaults)
    if isinstance(enabled, str):
        return [enabled]
    if isinstance(enabled, (list, tuple)):
        return [str(x) for x in enabled]
    return list(defaults)


def metric_options(cfg: Dict, metric_name: str) -> Dict[str, Any]:
    """
    Per-metric knobs under cfg.metrics.options.<metric_name>.
    """
    opts = _cfg_get(cfg, f"metrics.options.{metric_name}", {}) or {}
    return dict(opts)