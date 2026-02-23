# src/gencysynth/models/gan/sample.py
"""
GenCyberSynth — GAN Family — Sampling Router (non-variant)
=========================================================

This module exposes a stable entrypoint:
    synth(cfg, output_root, seed) -> manifest

and routes to the correct GAN variant sampler:
    gencysynth.models.gan.variants.<variant>.sample:synth

Why this exists
---------------
- Keeps orchestrator/CLI code stable (call gan.sample.synth for any GAN run).
- Allows adding new variants without changing orchestrator logic.
- Enforces dataset-aware output layout:
    {artifacts}/synth/{dataset_id}/gan/<variant>/...

Important path rule
-------------------
The *family router* is responsible for choosing a good default output_root.
If output_root is provided by an adapter, it should already be dataset-aware.

Variant identity
----------------
Variant is always printed in logs and embedded in the returned manifest.
"""

from __future__ import annotations

from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .base import (
    build_identity,
    ensure_dir,
    resolve_synth_root,
    resolve_dataset_root,
)

# ---------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------
def synth(cfg: Mapping[str, Any], output_root: Optional[str] = None, seed: int = 42) -> Dict[str, Any]:
    """
    Family-level synth() that routes to variant implementation.

    Parameters
    ----------
    cfg:
        Full experiment configuration mapping.
    output_root:
        Where to write outputs. If None, we compute a dataset-aware default:
            {artifacts}/synth/{dataset_id}
        The variant will then write under:
            <output_root>/gan/<variant>/...
    seed:
        Seed for deterministic synthesis.

    Returns
    -------
    dict
        A manifest dict returned by the variant sampler, enriched with:
          - family, variant, dataset_id
          - created_at (if missing)
          - dataset_root (optional provenance)
    """
    ident = build_identity(cfg)
    ds_root = resolve_dataset_root(cfg)

    # Choose output_root in a dataset-aware manner if not provided.
    if output_root is None:
        out_root = resolve_synth_root(cfg)
    else:
        out_root = Path(output_root)

    ensure_dir(out_root)

    # Variant sampler module path and function name contract.
    # Each variant must expose: synth(cfg, output_root, seed) -> manifest
    mod_path = f"gencysynth.models.gan.variants.{ident.variant}.sample"

    print(f"[{ident.tag}] synth router")
    print(f"[{ident.tag}] dataset_id={ident.dataset_id}")
    if ds_root:
        print(f"[{ident.tag}] dataset_root={ds_root}")
    print(f"[{ident.tag}] output_root={out_root}")

    try:
        mod = import_module(mod_path)
    except Exception as e:
        # Hard failure: variant is missing or import fails.
        # We return a stub manifest that makes the failure explicit.
        print(f"[{ident.tag}][ERROR] Could not import variant sampler: {mod_path}")
        print(f"[{ident.tag}][ERROR] {type(e).__name__}: {e}")

        return {
            "family": ident.family,
            "variant": ident.variant,
            "dataset_id": ident.dataset_id,
            "dataset_root": ds_root,
            "seed": int(seed),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "error": f"import_failed:{type(e).__name__}:{e}",
            "paths": [],
            "per_class_counts": {},
            "num_fake": 0,
        }

    if not hasattr(mod, "synth"):
        msg = f"Variant sampler '{mod_path}' does not define synth(cfg, output_root, seed)."
        print(f"[{ident.tag}][ERROR] {msg}")
        return {
            "family": ident.family,
            "variant": ident.variant,
            "dataset_id": ident.dataset_id,
            "dataset_root": ds_root,
            "seed": int(seed),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "error": "missing_synth_entrypoint",
            "paths": [],
            "per_class_counts": {},
            "num_fake": 0,
        }

    # Delegate to variant sampler
    try:
        man = mod.synth(dict(cfg), str(out_root), seed=int(seed))  # type: ignore[attr-defined]
    except Exception as e:
        print(f"[{ident.tag}][ERROR] Variant synth failed: {type(e).__name__}: {e}")
        man = {
            "error": f"variant_synth_failed:{type(e).__name__}:{e}",
            "paths": [],
            "per_class_counts": {},
            "num_fake": 0,
        }

    # Enrich manifest with stable identity fields (do not overwrite if variant already set them)
    if not isinstance(man, dict):
        man = {"paths": [], "per_class_counts": {}, "num_fake": 0}

    man.setdefault("family", ident.family)
    man.setdefault("variant", ident.variant)
    man.setdefault("dataset_id", ident.dataset_id)
    if ds_root:
        man.setdefault("dataset_root", ds_root)
    man.setdefault("seed", int(seed))
    man.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
    man.setdefault("output_root", str(out_root))

    return man


__all__ = ["synth"]

