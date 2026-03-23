# src/gencysynth/models/gan/pipeline.py
"""
GenCyberSynth — GAN Family — Pipeline Helpers (non_variant)
==========================================================

This module provides a thin "pipeline" layer for the GAN family.

Why pipeline.py
---------------
You often want a stable programmatic API to run the typical lifecycle:
  1) train (optional)
  2) synth

without requiring the caller to know the variant module structure.

This file remains small and defers all work to:
  - gencysynth.models.gan.train (router)
  - gencysynth.models.gan.sample (router)

It does not evaluate metrics and does not read datasets.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from .base import build_identity, resolve_synth_root
from .train import train as train_router
from .sample import synth as synth_router


def run_train(cfg: Mapping[str, Any]) -> int:
    """
    Run GAN training using family router.

    Expectation:
    - cfg.model.variant selects which variant to train
    - variant trainer decides how to use cfg (or config_path if you invoke via CLI)

    Returns: 0 on success, non_zero on failure (variant_defined).
    """
    ident = build_identity(cfg)
    print(f"[{ident.tag}] pipeline: train")
    return int(train_router(cfg))


def run_synth(cfg: Mapping[str, Any], *, output_root: Optional[str] = None, seed: int = 42) -> Dict[str, Any]:
    """
    Run GAN synthesis using family router.

    By default, uses dataset_aware output_root:
      {artifacts}/synth/{dataset_id}

    Returns a manifest dict (variant_defined + enriched by router).
    """
    ident = build_identity(cfg)

    if output_root is None:
        output_root = str(resolve_synth_root(cfg))

    print(f"[{ident.tag}] pipeline: synth")
    return synth_router(cfg, output_root=output_root, seed=seed)


__all__ = ["run_train", "run_synth"]
