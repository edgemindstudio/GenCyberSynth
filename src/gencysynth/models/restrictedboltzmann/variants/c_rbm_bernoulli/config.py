# src/gencysynth/models/restrictedboltzmann/variants/c_rbm_bernoulli/config.py
"""
Rule A config normalizer for: restrictedboltzmann/variants/c_rbm_bernoulli

Responsibilities
----------------
- Load defaults.yaml (this variant's defaults)
- Merge user overrides (dict loaded by the unified CLI)
- Resolve Rule A artifact paths so pipelines/sample/train write to the
  correct run_scoped folders even when multiple datasets are used.

This module is intentionally small and dependency_light.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _deep_update(dst: Dict[str, Any], src: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge src into dst (dicts only)."""
    for k, v in src.items():
        if isinstance(v, Mapping) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)  # type: ignore[index]
        else:
            dst[k] = v
    return dst


def _cfg_get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _as_path(x: Any) -> Path:
    return x if isinstance(x, Path) else Path(str(x))


def _read_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf_8") as f:
        return yaml.safe_load(f) or {}


# -----------------------------------------------------------------------------
# Rule A artifact path resolver
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class RunScope:
    dataset_id: str
    variant_family: str
    variant_name: str
    run_id: str


def _resolve_scope(cfg: Mapping[str, Any]) -> RunScope:
    dataset_id = str(_cfg_get(cfg, "data.dataset_id", "dataset"))
    run_id = str(_cfg_get(cfg, "run.run_id", _cfg_get(cfg, "run_id", "run")))

    family = str(_cfg_get(cfg, "variant.family", "restrictedboltzmann"))
    name = str(_cfg_get(cfg, "variant.name", "c_rbm_bernoulli"))

    return RunScope(
        dataset_id=dataset_id,
        variant_family=family,
        variant_name=name,
        run_id=run_id,
    )


def _default_rule_a_artifacts(cfg: Mapping[str, Any]) -> Dict[str, str]:
    """
    Produce Rule A artifact directories.

    Layout:
      {paths.artifacts}/runs/<dataset_id>/<family>/<name>/<run_id>/
        - checkpoints/
        - summaries/
        - synthetic/
    """
    root = _as_path(_cfg_get(cfg, "paths.artifacts", "artifacts"))
    scope = _resolve_scope(cfg)
    base = root / "runs" / scope.dataset_id / scope.variant_family / scope.variant_name / scope.run_id

    return {
        "checkpoints": str(base / "checkpoints"),
        "summaries": str(base / "summaries"),
        "synthetic": str(base / "synthetic"),
    }


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def load_defaults() -> Dict[str, Any]:
    """
    Load this variant's defaults.yaml located next to this file.
    """
    here = Path(__file__).resolve().parent
    return _read_yaml(here / "defaults.yaml")


def normalize(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Merge defaults with user cfg and resolve artifact directories.

    Returns a dict suitable to pass directly into:
      - pipeline.RBMPipeline(cfg)
      - train/train.py adapter (if used)
      - sample.synth(cfg, ...)
    """
    merged: Dict[str, Any] = load_defaults()
    if cfg:
        _deep_update(merged, cfg)

    # Ensure top_level sections exist
    merged.setdefault("paths", {})
    merged.setdefault("data", {})
    merged.setdefault("variant", {})
    merged.setdefault("run", {})
    merged.setdefault("artifacts", {})

    # Expand/override artifacts with Rule A concrete paths unless the user
    # explicitly provided absolute paths (we still respect explicit values).
    rule_a = _default_rule_a_artifacts(merged)

    arts = merged.get("artifacts", {}) or {}
    for key in ("checkpoints", "summaries", "synthetic"):
        if key not in arts or arts[key] in (None, "", "auto"):
            arts[key] = rule_a[key]

    merged["artifacts"] = arts

    # Back_compat bridge:
    # Some of your older modules read cfg["ARTIFACTS"].
    merged.setdefault("ARTIFACTS", {})
    merged["ARTIFACTS"].setdefault("checkpoints", merged["artifacts"]["checkpoints"])
    merged["ARTIFACTS"].setdefault("summaries", merged["artifacts"]["summaries"])
    merged["ARTIFACTS"].setdefault("synthetic", merged["artifacts"]["synthetic"])

    return merged


__all__ = ["normalize",
           "load_defaults",
           "RunScope"]