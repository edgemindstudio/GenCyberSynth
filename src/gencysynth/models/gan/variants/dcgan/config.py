# src/gencysynth/models/gan/variants/dcgan/config.py
"""
GenCyberSynth — DCGAN Variant Config Defaults
=============================================

Purpose
-------
This module defines *default hyperparameters* for the DCGAN variant.

In GenCyberSynth, config is layered:
  1) Global defaults (configs/base.yaml etc.)
  2) Dataset config  (configs/datasets/<dataset>.yaml)
  3) Family config   (configs/families/gan.yaml)
  4) Variant config  (THIS FILE: gan/dcgan)
  5) Run overrides   (CLI --overrides or sweep tooling)

This file intentionally:
- DOES NOT read/write any files
- DOES NOT assume a dataset directory layout
- DOES NOT create artifact paths

Artifact locations are resolved elsewhere via Rule A:
  (dataset_id, model_tag, run_id)

Examples
--------
- model_tag: "gan/dcgan"
- outputs:
    artifacts/runs/<dataset_id>/<model_tag>/<run_id>/...
    artifacts/logs/<dataset_id>/<model_tag>/<run_id>/...
    artifacts/eval/<dataset_id>/<model_tag>/<run_id>/...

How to use
----------
In the DCGAN builder (e.g., gencysynth.models.gan.variants.dcgan:build),
you typically do:

    from .config import default_dcgan_config, merge_variant_config
    cfg = merge_variant_config(cfg)

Where user YAML can override any nested key under:
    cfg["model"]["params"]["dcgan"]

Design goals
------------
- Stable: defaults should not change frequently once experiments start.
- Explicit: all important knobs are visible and documented.
- Override_friendly: every value is plain JSON_serializable types.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping


# =============================================================================
# Default configuration (variant_scoped)
# =============================================================================
def default_dcgan_config() -> Dict[str, Any]:
    """
    Return a dict of default hyperparameters for DCGAN.

    Notes
    -----
    - These defaults are for small images (e.g., 40x40 grayscale) but should
      generalize to other sizes if the model implementation supports it.
    - Keep dataset_specific settings out of here unless they are truly DCGAN_specific.
      Prefer dataset.image_hw, dataset.channels, etc. in dataset configs.
    """
    return {
        # Human_friendly identity (variant_local; orchestration owns run_id)
        "name": "dcgan",
        "model_tag": "gan/dcgan",

        # ---- Architecture defaults
        "arch": {
            # Latent noise dimension z
            "z_dim": 128,

            # Generator feature base width (channels multiplier)
            "g_base_channels": 64,

            # Discriminator feature base width
            "d_base_channels": 64,

            # Normalization choices (implementation_dependent)
            "g_use_batchnorm": True,
            "d_use_batchnorm": False,

            # Activation choices
            "g_activation": "relu",
            "g_out_activation": "tanh",  # common for [-1,1] outputs
            "d_activation": "leaky_relu",
            "d_leaky_relu_alpha": 0.2,
        },

        # ---- Training defaults
        "train": {
            # Total epochs for adversarial training
            "epochs": 50,

            # Batch size (tune by GPU memory)
            "batch_size": 128,

            # Optional: number of discriminator steps per generator step
            "d_steps": 1,
            "g_steps": 1,

            # Optional: label smoothing / noisy labels
            "label_smoothing": 0.0,   # e.g., 0.1 means real labels become 0.9
            "label_noise": 0.0,       # probability to flip labels (rarely used)

            # Optional: gradient clipping
            "grad_clip_norm": None,   # e.g., 1.0

            # Logging cadence (steps)
            "log_every_steps": 200,

            # Checkpoint cadence (epochs)
            "ckpt_every_epochs": 1,

            # Sample preview cadence (epochs) — written under run_dir/samples/
            "sample_every_epochs": 1,

            # How many preview samples to generate per preview event
            "preview_num": 64,
        },

        # ---- Optimizers
        "optim": {
            # DCGAN commonly uses Adam with beta1=0.5
            "g_lr": 2e_4,
            "d_lr": 2e_4,
            "adam_beta1": 0.5,
            "adam_beta2": 0.999,
        },

        # ---- Output scaling convention
        # Your pipeline typically standardizes this at dataset/preprocess level,
        # but the model often needs to know the expected output range.
        "io": {
            # Expected generator output scaling:
            #   "tanh" output is typically [-1, 1]
            "output_range": "[-1,1]",

            # Whether training data is expected in the same scaling
            # (your dataset transforms should enforce consistency)
            "expect_input_range": "[-1,1]",
        },

        # ---- Sampling defaults (manifest writing is orchestration_owned)
        "sample": {
            # Number of synthetic samples to generate per class (if conditional)
            # For unconditional DCGAN, orchestration may ignore per_class and just
            # generate total = K * budget_per_class using label assignment rules.
            "budget_per_class": None,   # leave None; orchestrator/run_meta sets truth

            # Optional: cap total samples (if caller uses it)
            "max_total": None,

            # Whether to write image files during sampling (vs .npy arrays etc.)
            # The variant implementation decides the format; the manifest records paths.
            "write_images": True,
            "image_ext": "png",
        },
    }


# =============================================================================
# Config merge utilities
# =============================================================================
def _deep_update(base: Dict[str, Any], upd: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge upd into base (in place), returning base.
    This matches the style used in eval/runner.py.
    """
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)  # type: ignore[index]
        else:
            base[k] = v
    return base


def merge_variant_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Merge DCGAN defaults into a full experiment config, without overwriting user choices.

    Contract
    --------
    - Returns a *new* mutable dict (safe to modify downstream).
    - Merges into cfg["model"]["params"]["dcgan"] by default.
    - Also sets cfg["model"]["tag"] if missing.

    Expected config layout
    ----------------------
    model:
      tag: "gan/dcgan"
      params:
        dcgan:
          train:
            epochs: 100

    Why this shape?
    ---------------
    It allows:
    - multiple model families to coexist without key collisions
    - multiple variants under a family (dcgan/wgan/wgangp) to each have their own block
    """
    out: Dict[str, Any] = deepcopy(dict(cfg)) if isinstance(cfg, dict) else {}

    # Ensure top_level sections exist
    model = out.get("model")
    if not isinstance(model, dict):
        model = {}
        out["model"] = model

    params = model.get("params")
    if not isinstance(params, dict):
        params = {}
        model["params"] = params

    # Variant_scoped config lives under model.params.dcgan
    user_block = params.get("dcgan")
    if not isinstance(user_block, dict):
        user_block = {}
        params["dcgan"] = user_block

    defaults = default_dcgan_config()

    # Merge defaults -> user (user overrides win)
    merged = deepcopy(defaults)
    _deep_update(merged, user_block)

    # Write merged back into cfg
    params["dcgan"] = merged

    # Ensure model.tag is present and consistent
    if not isinstance(model.get("tag"), str) or not model.get("tag"):
        model["tag"] = defaults["model_tag"]

    # Optional convenience: family/variant fields (nice for dashboards)
    model.setdefault("family", "gan")
    model.setdefault("variant", "dcgan")

    return out


__all__ = ["default_dcgan_config", "merge_variant_config"]
