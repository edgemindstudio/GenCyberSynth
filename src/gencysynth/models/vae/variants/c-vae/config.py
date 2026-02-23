# src/gencysynth/models/vae/variants/c-vae/config.py
"""
Rule A — cVAE (Conditional VAE) variant config utilities.

This module standardizes how the cVAE variant:
- interprets user config (YAML/dict),
- resolves dataset identity,
- and *derives artifact paths* in a scalable way.

Design goals
------------
1) Dataset-aware artifacts (so multiple datasets don't collide):
     {paths.artifacts}/{data.id}/models/vae/c-vae/...
2) Run-aware folders (optional) to keep multiple experiments:
     .../runs/{run.name}/...
3) Backward-compatible aliases:
   - IMG_SHAPE, NUM_CLASSES, LATENT_DIM, EPOCHS, BATCH_SIZE, LR, BETA_1, BETA_KL
   - data.root / DATA_DIR, num_classes, vae.latent_dim, etc.

This file does not train or sample — it only prepares normalized config.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import copy
import re


# -----------------------------------------------------------------------------
# Small dict utilities
# -----------------------------------------------------------------------------
def cfg_get(cfg: Dict[str, Any], dotted: str, default=None):
    """Safely fetch cfg['a']['b']['c'] via dotted string."""
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def cfg_setdefault_path(cfg: Dict[str, Any], dotted: str, value: str) -> None:
    """Set dotted path if missing, creating dicts as needed."""
    keys = dotted.split(".")
    cur = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur.setdefault(keys[-1], value)


def _slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "default"


# -----------------------------------------------------------------------------
# Rule A conventions (dataset-aware artifact routing)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class CVAEPaths:
    """
    Canonical artifact locations for Rule A.

    Layout (dataset-aware):
      {artifacts_root}/{data_id}/models/vae/c-vae/
        checkpoints/
        summaries/
        synthetic/
        tensorboard/
    """
    root: Path
    checkpoints: Path
    summaries: Path
    synthetic: Path
    tensorboard: Path


def resolve_data_id(cfg: Dict[str, Any]) -> str:
    """
    Determine a stable dataset identifier.

    Priority:
      1) data.id (explicit)
      2) dataset.id (fallback)
      3) basename of data.root or DATA_DIR
      4) "default_dataset"
    """
    data_id = cfg_get(cfg, "data.id", None) or cfg_get(cfg, "dataset.id", None)
    if isinstance(data_id, str) and data_id.strip():
        return _slugify(data_id)

    root = cfg_get(cfg, "data.root", None) or cfg.get("DATA_DIR", None)
    if isinstance(root, str) and root.strip():
        return _slugify(Path(root).name)

    return "default-dataset"


def resolve_run_name(cfg: Dict[str, Any]) -> Optional[str]:
    """
    Optional run grouping (recommended if you do many experiments).
    If missing, we do not create a runs/<name>/ layer.
    """
    run_name = cfg_get(cfg, "run.name", None) or cfg_get(cfg, "experiment.name", None)
    if isinstance(run_name, str) and run_name.strip():
        return _slugify(run_name)
    return None


def make_cvae_paths(cfg: Dict[str, Any]) -> CVAEPaths:
    """
    Build dataset-aware (and optionally run-aware) paths for this variant.
    """
    artifacts_root = Path(cfg_get(cfg, "paths.artifacts", "artifacts"))
    data_id = resolve_data_id(cfg)
    run_name = resolve_run_name(cfg)

    # Base: artifacts/<data_id>/models/vae/c-vae
    base = artifacts_root / data_id / "models" / "vae" / "c-vae"

    # Optional: .../runs/<run_name>
    if run_name:
        base = base / "runs" / run_name

    ckpt = base / "checkpoints"
    summ = base / "summaries"
    synth = base / "synthetic"
    tb = artifacts_root / data_id / "tensorboard" / "vae" / "c-vae"
    if run_name:
        tb = tb / run_name

    return CVAEPaths(root=base, checkpoints=ckpt, summaries=summ, synthetic=synth, tensorboard=tb)


# -----------------------------------------------------------------------------
# Normalization
# -----------------------------------------------------------------------------
def normalize_cfg(cfg_in: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize config dict for cVAE and populate cfg['ARTIFACTS'].

    Output keys guaranteed (if absent in input):
      - IMG_SHAPE (tuple)
      - NUM_CLASSES (int)
      - LATENT_DIM (int)
      - EPOCHS (int)
      - BATCH_SIZE (int)
      - LR, BETA_1, BETA_KL (float)
      - VAL_FRACTION (float, default 0.5)
      - SEED (int, default 42)
      - ARTIFACTS.checkpoints/summaries/synthetic/tensorboard (str)
      - data.id (derived if missing)
    """
    cfg = copy.deepcopy(cfg_in) if cfg_in is not None else {}

    # Basic defaults / alias support
    cfg.setdefault("SEED", 42)
    cfg.setdefault("VAL_FRACTION", cfg_get(cfg, "data.val_fraction", 0.5))

    # Shape & classes (support legacy keys)
    img_shape = cfg.get("IMG_SHAPE", cfg_get(cfg, "img.shape", (40, 40, 1)))
    cfg["IMG_SHAPE"] = tuple(img_shape)

    num_classes = cfg.get("NUM_CLASSES", cfg.get("num_classes", cfg_get(cfg, "data.num_classes", 9)))
    cfg["NUM_CLASSES"] = int(num_classes)

    # Model hparams
    cfg["LATENT_DIM"] = int(cfg.get("LATENT_DIM", cfg_get(cfg, "vae.latent_dim", 100)))
    cfg["EPOCHS"] = int(cfg.get("EPOCHS", cfg_get(cfg, "train.epochs", 200)))
    cfg["BATCH_SIZE"] = int(cfg.get("BATCH_SIZE", cfg_get(cfg, "train.batch_size", 256)))

    cfg["LR"] = float(cfg.get("LR", cfg_get(cfg, "vae.lr", 2e-4)))
    cfg["BETA_1"] = float(cfg.get("BETA_1", cfg_get(cfg, "vae.beta_1", 0.5)))
    cfg["BETA_KL"] = float(cfg.get("BETA_KL", cfg_get(cfg, "vae.beta_kl", 1.0)))

    # Ensure data.root exists if only DATA_DIR provided (for unified conventions)
    if cfg_get(cfg, "data.root", None) is None and isinstance(cfg.get("DATA_DIR", None), str):
        cfg_setdefault_path(cfg, "data.root", cfg["DATA_DIR"])

    # Ensure/derive data.id
    data_id = resolve_data_id(cfg)
    cfg_setdefault_path(cfg, "data.id", data_id)

    # Derive artifact paths (Rule A)
    paths = make_cvae_paths(cfg)

    cfg.setdefault("ARTIFACTS", {})
    A = cfg["ARTIFACTS"]

    # Canonical locations for this variant
    A.setdefault("checkpoints", str(paths.checkpoints))
    A.setdefault("summaries", str(paths.summaries))
    A.setdefault("synthetic", str(paths.synthetic))
    A.setdefault("tensorboard", str(paths.tensorboard))

    # Convenience: explicit decoder/encoder best paths (used by sample/synth)
    A.setdefault("encoder_best", str(paths.checkpoints / "E_best.weights.h5"))
    A.setdefault("decoder_best", str(paths.checkpoints / "D_best.weights.h5"))
    A.setdefault("encoder_last", str(paths.checkpoints / "E_last.weights.h5"))
    A.setdefault("decoder_last", str(paths.checkpoints / "D_last.weights.h5"))

    return cfg


def ensure_artifact_dirs(cfg: Dict[str, Any]) -> None:
    """
    Create artifact directories referenced by cfg['ARTIFACTS'] (safe on HPC).
    """
    arts = cfg.get("ARTIFACTS", {}) or {}
    for k in ("checkpoints", "summaries", "synthetic", "tensorboard"):
        p = arts.get(k, None)
        if p:
            Path(p).mkdir(parents=True, exist_ok=True)


__all__ = [
    "CVAEPaths",
    "cfg_get",
    "resolve_data_id",
    "resolve_run_name",
    "make_cvae_paths",
    "normalize_cfg",
    "ensure_artifact_dirs",
]
