# src/gencysynth/models/maskedautoflow/variants/maf_affine/config.py
"""
Rule A config glue for MaskedAutoFlow / maf_affine.

This module is intentionally small:
- Loads the variant's defaults.yaml
- Merges user overrides (from unified cfg dict)
- Exposes helpers to resolve *dataset-scoped* artifact directories

The goal is that train.py / sample.py / pipeline.py can:
  1) call load_defaults()
  2) call merge_cfg(defaults, user_cfg)
  3) call resolve_paths(merged_cfg) to get correct, scalable artifact roots

No model code and no heavy imports here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml


# ----------------------------- YAML helpers ----------------------------- #
def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"defaults.yaml not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge upd into base (dicts only).
    Returns a NEW dict; does not mutate the inputs.
    """
    out = dict(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _cfg_get(cfg: Dict[str, Any], dotted: str, default=None):
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


# ----------------------------- Defaults ----------------------------- #
def load_defaults() -> Dict[str, Any]:
    """
    Load defaults.yaml shipped next to this file.
    """
    here = Path(__file__).resolve().parent
    return _read_yaml(here / "defaults.yaml")


def merge_cfg(defaults: Dict[str, Any], user_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Merge unified user cfg dict into variant defaults.

    We accept unified cfgs that might store model settings under:
      - cfg['models'][family][variant] ... (new style), OR
      - top-level keys (legacy)

    This function keeps it simple:
      - We deep-merge the entire user_cfg onto defaults
      - Callers can pass a narrowed view if they want

    Recommended usage (typical):
      merged = merge_cfg(load_defaults(), cfg.get("models", {}).get("maskedautoflow", {}).get("maf_affine", {}))
    """
    return _deep_update(defaults, user_cfg or {})


# ----------------------------- Artifact path rules (Rule A) ----------------------------- #
@dataclass(frozen=True)
class ArtifactPaths:
    """
    Dataset-scoped artifact directories for this variant.

    Layout (Rule A)
    ---------------
    artifacts/
      <dataset_id>/
        maskedautoflow/
          maf_affine/
            checkpoints/
            summaries/
            synthetic/
    """
    root: Path
    checkpoints: Path
    summaries: Path
    synthetic: Path


def resolve_dataset_id(cfg: Dict[str, Any]) -> str:
    """
    Resolve a stable dataset identifier for artifact scoping.

    Priority order:
      1) cfg['data']['id']
      2) cfg['dataset']['id']
      3) basename of cfg['data']['root'] (or DATA_DIR)
      4) fallback: 'default_dataset'
    """
    ds_id = _cfg_get(cfg, "data.id", None) or _cfg_get(cfg, "dataset.id", None)
    if ds_id:
        return str(ds_id)

    data_root = _cfg_get(cfg, "data.root", None) or _cfg_get(cfg, "DATA_DIR", None)
    if data_root:
        return Path(str(data_root)).expanduser().resolve().name

    return "default_dataset"


def resolve_artifacts_root(cfg: Dict[str, Any]) -> Path:
    """
    Resolve the global artifacts root (not dataset-scoped yet).

    Priority:
      1) cfg['paths']['artifacts']
      2) cfg['paths.artifacts'] (if flattened)
      3) 'artifacts'
    """
    root = _cfg_get(cfg, "paths.artifacts", None)
    if root is None:
        # tolerate flattened key usage
        root = cfg.get("paths.artifacts", None)
    return Path(root or "artifacts")


def resolve_paths(cfg: Dict[str, Any]) -> ArtifactPaths:
    """
    Resolve dataset-scoped artifact paths for maf_affine.

    If the config explicitly provides overrides under:
      cfg['artifacts']['maskedautoflow']['maf_affine'][...]
    we will respect them, otherwise we use Rule A defaults.
    """
    artifacts_root = resolve_artifacts_root(cfg)
    dataset_id = resolve_dataset_id(cfg)

    # Default Rule A layout
    base = artifacts_root / dataset_id / "maskedautoflow" / "maf_affine"

    # Optional overrides
    ckpt = _cfg_get(cfg, "artifacts.maskedautoflow.maf_affine.checkpoints", None)
    sums = _cfg_get(cfg, "artifacts.maskedautoflow.maf_affine.summaries", None)
    synth = _cfg_get(cfg, "artifacts.maskedautoflow.maf_affine.synthetic", None)

    ckpt_dir = Path(ckpt) if ckpt else (base / "checkpoints")
    sums_dir = Path(sums) if sums else (base / "summaries")
    synth_dir = Path(synth) if synth else (base / "synthetic")

    return ArtifactPaths(
        root=base,
        checkpoints=ckpt_dir,
        summaries=sums_dir,
        synthetic=synth_dir,
    )


__all__ = [
    "ArtifactPaths",
    "load_defaults",
    "merge_cfg",
    "resolve_dataset_id",
    "resolve_artifacts_root",
    "resolve_paths",
]
