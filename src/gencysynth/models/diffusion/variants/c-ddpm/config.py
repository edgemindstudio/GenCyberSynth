# src/gencysynth/models/diffusion/variants/c-ddpm/config.py
"""
GenCyberSynth — Diffusion family — c-DDPM variant — Config helpers
=================================================================

RULE A (Scalable artifact policy)
---------------------------------
This module is **config-only**:
  - It defines the canonical model tag and variant identity.
  - It provides small helpers to read config keys with sane fallbacks.
  - It does NOT write artifacts and does NOT resolve run directories.

Run path resolution is done by:
  gencysynth.orchestration.context.resolve_run_context

This keeps the variant portable across:
  - multiple datasets
  - multiple runs per dataset (HPC arrays)
  - multiple seeds

Variant identity
----------------
- family:   diffusion
- variant:  c-ddpm
- model_tag: diffusion/c-ddpm
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple


# -----------------------------------------------------------------------------
# Variant identity (must match folder path and registry tag)
# -----------------------------------------------------------------------------
FAMILY: str = "diffusion"
VARIANT: str = "c-ddpm"
MODEL_TAG: str = "diffusion/c-ddpm"


# -----------------------------------------------------------------------------
# Small config getters
# -----------------------------------------------------------------------------
def cfg_get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    """
    Read nested config values using dotted keys, e.g.:
        cfg_get(cfg, "paths.artifacts", "artifacts")
        cfg_get(cfg, "diffusion.timesteps", 1000)

    Returns `default` if the path does not exist.
    """
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def cfg_get_int(cfg: Mapping[str, Any], dotted: str, default: int) -> int:
    v = cfg_get(cfg, dotted, default)
    return int(v)


def cfg_get_float(cfg: Mapping[str, Any], dotted: str, default: float) -> float:
    v = cfg_get(cfg, dotted, default)
    return float(v)


# -----------------------------------------------------------------------------
# Canonical parameter resolution
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class CDdpmParams:
    """
    Canonical c-DDPM parameters resolved from config.

    Notes
    -----
    - These are model/training knobs (NOT run paths).
    - Train/sampling code should call resolve_params(cfg) once and then
      treat these as the single source of truth for this run.
    """

    # Data / shapes
    img_shape: Tuple[int, int, int]
    num_classes: int

    # Model architecture
    base_filters: int
    depth: int
    time_emb_dim: int

    # Training
    epochs: int
    batch_size: int
    lr: float
    beta_1: float
    patience: int
    log_every: int

    # Diffusion schedule
    timesteps: int
    schedule: str
    beta_start: float
    beta_end: float

    # Synthesis budget
    samples_per_class: int
    diffusion_steps_preview: int


def resolve_params(cfg: Mapping[str, Any]) -> CDdpmParams:
    """
    Resolve parameters using the repo-wide conventions.

    Priority strategy (matches your GAN code style)
    -----------------------------------------------
    1) New structured keys under:
         diffusion.*
         synth.*
    2) Legacy flat keys:
         IMG_SHAPE, NUM_CLASSES, EPOCHS, BATCH_SIZE, ...
    3) Built-in defaults (safe)
    """
    # --- shapes ---
    img_shape = tuple(cfg_get(cfg, "IMG_SHAPE", cfg_get(cfg, "img.shape", (40, 40, 1))))
    H, W, C = int(img_shape[0]), int(img_shape[1]), int(img_shape[2])

    num_classes = int(cfg_get(cfg, "NUM_CLASSES", cfg_get(cfg, "num_classes", 9)))

    # --- training ---
    epochs = int(cfg_get(cfg, "EPOCHS", cfg_get(cfg, "diffusion.epochs", 200)))
    batch_size = int(cfg_get(cfg, "BATCH_SIZE", cfg_get(cfg, "diffusion.batch_size", 256)))
    lr = float(cfg_get(cfg, "LR", cfg_get(cfg, "diffusion.lr", 2e-4)))
    beta_1 = float(cfg_get(cfg, "BETA_1", cfg_get(cfg, "diffusion.beta_1", 0.9)))
    patience = int(cfg_get(cfg, "PATIENCE", cfg_get(cfg, "diffusion.patience", 10)))
    log_every = int(cfg_get(cfg, "LOG_EVERY", cfg_get(cfg, "diffusion.log_every", 25)))

    # --- architecture ---
    base_filters = int(cfg_get(cfg, "BASE_FILTERS", cfg_get(cfg, "diffusion.base_filters", 64)))
    depth = int(cfg_get(cfg, "DEPTH", cfg_get(cfg, "diffusion.depth", 2)))
    time_emb_dim = int(cfg_get(cfg, "TIME_EMB_DIM", cfg_get(cfg, "diffusion.time_emb_dim", 128)))

    # --- schedule ---
    timesteps = int(cfg_get(cfg, "TIMESTEPS", cfg_get(cfg, "diffusion.timesteps", 1000)))
    schedule = str(cfg_get(cfg, "SCHEDULE", cfg_get(cfg, "diffusion.schedule", "linear"))).lower()
    beta_start = float(cfg_get(cfg, "BETA_START", cfg_get(cfg, "diffusion.beta_start", 1e-4)))
    beta_end = float(cfg_get(cfg, "BETA_END", cfg_get(cfg, "diffusion.beta_end", 2e-2)))

    # --- synthesis budget (Rule A: sampling writes under ctx.run_dir/samples/...) ---
    samples_per_class = int(
        cfg_get(cfg, "synth.n_per_class", cfg_get(cfg, "SAMPLES_PER_CLASS", cfg_get(cfg, "samples_per_class", 25)))
    )

    # A separate knob for quick preview sampling (smaller T for speed if desired)
    diffusion_steps_preview = int(
        cfg_get(cfg, "DIFFUSION_STEPS", cfg_get(cfg, "diffusion.steps_preview", 200))
    )

    return CDdpmParams(
        img_shape=(H, W, C),
        num_classes=num_classes,
        base_filters=base_filters,
        depth=depth,
        time_emb_dim=time_emb_dim,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        beta_1=beta_1,
        patience=patience,
        log_every=log_every,
        timesteps=timesteps,
        schedule=schedule,
        beta_start=beta_start,
        beta_end=beta_end,
        samples_per_class=samples_per_class,
        diffusion_steps_preview=diffusion_steps_preview,
    )


__all__ = [
    "FAMILY",
    "VARIANT",
    "MODEL_TAG",
    "CDdpmParams",
    "cfg_get",
    "cfg_get_int",
    "cfg_get_float",
    "resolve_params",
]
