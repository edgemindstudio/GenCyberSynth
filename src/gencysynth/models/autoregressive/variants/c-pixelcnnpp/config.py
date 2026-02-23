# src/gencysynth/models/autoregressive/variants/c-pixelcnnpp/config.py

"""
GenCyberSynth — Autoregressive — c-pixelcnnpp — Config
======================================================

Rule A goals
------------
1) Dataset-aware artifacts:
   All outputs must live under:
     <paths.artifacts>/<dataset>/<family>/<variant>/{checkpoints,synthetic,summaries,tensorboard}

2) Config stability:
   Support both the new nested keys (model.*, train.*, synth.*, data.*, paths.*)
   and older flat keys (IMG_SHAPE, NUM_CLASSES, EPOCHS, etc.) so existing runs
   do not break.

This module is intentionally small and dependency-light. It provides helpers to:
- read defaults.yaml
- merge user config
- resolve standardized artifact directories
- expose a "normalized" flat dict for legacy code paths
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# =============================================================================
# Small helpers
# =============================================================================
def _cfg_get(cfg: Dict[str, Any], dotted: str, default=None):
    """Read nested config keys with dot-notation."""
    cur: Any = cfg
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge src into dst (mutates dst)."""
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _read_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# =============================================================================
# Artifact path resolution (Rule A)
# =============================================================================
@dataclass(frozen=True)
class ArtifactPaths:
    """
    Canonical artifact directories for this variant under Rule A.
    """
    root: Path
    dataset: str
    family: str
    variant: str

    checkpoints: Path
    synthetic: Path
    summaries: Path
    tensorboard: Path


def resolve_artifacts(cfg: Dict[str, Any]) -> ArtifactPaths:
    """
    Resolve dataset-aware artifact directories.

    Precedence
    ----------
    1) cfg["artifacts"]["checkpoints"/...], if provided (advanced override)
    2) Otherwise construct from:
         paths.artifacts + data.name + model.family + model.variant

    Returns
    -------
    ArtifactPaths with directories created by caller (we do not mkdir here).
    """
    artifacts_root = Path(_cfg_get(cfg, "paths.artifacts", "artifacts"))
    dataset_name = str(_cfg_get(cfg, "data.name", _cfg_get(cfg, "DATASET", "dataset")))
    family = str(_cfg_get(cfg, "model.family", "autoregressive"))
    variant = str(_cfg_get(cfg, "model.variant", "c-pixelcnnpp"))

    base = artifacts_root / dataset_name / family / variant

    # Optional explicit overrides (still should include dataset in the path)
    ckpt = Path(_cfg_get(cfg, "artifacts.checkpoints", base / "checkpoints"))
    synth = Path(_cfg_get(cfg, "artifacts.synthetic", base / "synthetic"))
    sums = Path(_cfg_get(cfg, "artifacts.summaries", base / "summaries"))
    tb = Path(_cfg_get(cfg, "artifacts.tensorboard", base / "tensorboard"))

    return ArtifactPaths(
        root=artifacts_root,
        dataset=dataset_name,
        family=family,
        variant=variant,
        checkpoints=ckpt,
        synthetic=synth,
        summaries=sums,
        tensorboard=tb,
    )


# =============================================================================
# Public API
# =============================================================================
def load_defaults() -> Dict[str, Any]:
    """
    Load defaults.yaml sitting next to this file.
    """
    here = Path(__file__).resolve().parent
    return _read_yaml(here / "defaults.yaml")


def load_config(user_cfg_path: Optional[str | Path] = None, user_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load defaults + merge user overrides from either:
      - a YAML file path
      - a dict payload

    Returns merged config (nested).
    """
    cfg = load_defaults()

    if user_cfg_path is not None:
        cfg_path = Path(user_cfg_path)
        _deep_update(cfg, _read_yaml(cfg_path))

    if user_cfg is not None:
        _deep_update(cfg, dict(user_cfg))

    return cfg


def normalize_legacy_keys(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Provide a flat, legacy-friendly view of the config.

    This helps older training/sampling code that expects keys like IMG_SHAPE,
    NUM_CLASSES, EPOCHS, etc. New code should prefer nested keys.
    """
    out = dict(cfg)  # shallow copy

    # Shapes / classes
    out["IMG_SHAPE"] = tuple(_cfg_get(cfg, "model.img_shape", _cfg_get(cfg, "IMG_SHAPE", (40, 40, 1))))
    out["NUM_CLASSES"] = int(_cfg_get(cfg, "model.num_classes", _cfg_get(cfg, "NUM_CLASSES", 9)))

    # Training
    out["SEED"] = int(_cfg_get(cfg, "train.seed", _cfg_get(cfg, "SEED", 42)))
    out["EPOCHS"] = int(_cfg_get(cfg, "train.epochs", _cfg_get(cfg, "EPOCHS", 200)))
    out["BATCH_SIZE"] = int(_cfg_get(cfg, "train.batch_size", _cfg_get(cfg, "BATCH_SIZE", 256)))
    out["LR"] = float(_cfg_get(cfg, "train.lr", _cfg_get(cfg, "LR", 2e-4)))
    out["BETA_1"] = float(_cfg_get(cfg, "train.beta_1", _cfg_get(cfg, "BETA_1", 0.5)))
    out["PATIENCE"] = int(_cfg_get(cfg, "train.patience", _cfg_get(cfg, "PATIENCE", 10)))
    out["SAVE_EVERY"] = int(_cfg_get(cfg, "train.save_every", _cfg_get(cfg, "SAVE_EVERY", 25)))
    out["FROM_LOGITS"] = bool(_cfg_get(cfg, "train.from_logits", _cfg_get(cfg, "FROM_LOGITS", False)))
    out["VAL_FRACTION"] = float(_cfg_get(cfg, "train.val_fraction", _cfg_get(cfg, "VAL_FRACTION", 0.5)))

    # Model knobs
    out["FILTERS"] = int(_cfg_get(cfg, "model.filters", _cfg_get(cfg, "FILTERS", 64)))
    out["MASKED_LAYERS"] = int(_cfg_get(cfg, "model.masked_layers", _cfg_get(cfg, "MASKED_LAYERS", 6)))
    out["LABEL_CHANNELS"] = int(_cfg_get(cfg, "model.label_channels", _cfg_get(cfg, "LABEL_CHANNELS", 1)))

    # Data root
    out["DATA_DIR"] = str(_cfg_get(cfg, "data.root", _cfg_get(cfg, "DATA_DIR", "data")))

    # Synthesis
    out["SAMPLES_PER_CLASS"] = int(_cfg_get(cfg, "synth.samples_per_class", _cfg_get(cfg, "SAMPLES_PER_CLASS", 1000)))

    # Artifact paths (Rule A canonical)
    ap = resolve_artifacts(cfg)
    out.setdefault("ARTIFACTS", {})
    out["ARTIFACTS"]["autoregressive_checkpoints"] = str(ap.checkpoints)
    out["ARTIFACTS"]["autoregressive_synthetic"] = str(ap.synthetic)
    out["ARTIFACTS"]["autoregressive_summaries"] = str(ap.summaries)
    out["ARTIFACTS"]["autoregressive_tensorboard"] = str(ap.tensorboard)

    return out


__all__ = [
    "ArtifactPaths",
    "resolve_artifacts",
    "load_defaults",
    "load_config",
    "normalize_legacy_keys",
]