# src/gencysynth/models/gan/variants/dcgan/sample.py
"""
GenCyberSynth — GAN — DCGAN (Conditional) — Sampling / Synthesis
===============================================================

Scalability rule (Rule A)
-------------------------
Everything is keyed by:
    (dataset_id, model_tag, run_id)

Therefore, sampling outputs MUST be written under the run directory:

  artifacts/runs/<dataset_id>/<model_tag>/<run_id>/
    samples/generated/class_<k>/gan_00000.png
    manifest.json

Logs MUST go to:
  artifacts/logs/<dataset_id>/<model_tag>/<run_id>/run.log

This module provides:
- sample(cfg, ctx) -> SampleResult  (preferred orchestrator entrypoint)
- synth(cfg, output_root, seed) -> dict (backward_compatible wrapper)

It does NOT do evaluation; evaluation belongs under gencysynth/eval/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

from .model import build_models

from gencysynth.models.base_types import RunContext, SampleResult
from gencysynth.orchestration.context import resolve_run_context
from gencysynth.orchestration.logger import get_run_logger
from gencysynth.utils.io import write_json
from gencysynth.utils.paths import ensure_dir


# -----------------------------------------------------------------------------
# Variant identity (must match folder structure and registry tag)
# -----------------------------------------------------------------------------
FAMILY: str = "gan"
VARIANT: str = "dcgan"
MODEL_TAG: str = "gan/dcgan"


# -----------------------------------------------------------------------------
# Config helper utilities
# -----------------------------------------------------------------------------
def _cfg_get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    """
    Read nested config values using a dotted key, e.g. 'paths.artifacts'.

    Returns `default` if any key is missing or intermediate values are not dicts.
    """
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _resolve_samples_per_class(cfg: Mapping[str, Any], default: int = 25) -> int:
    """
    Resolve synthesis budget.

    Priority:
      1) synth.n_per_class   (repo_wide key)
      2) SAMPLES_PER_CLASS   (legacy)
      3) samples_per_class   (legacy)
      4) default
    """
    v = _cfg_get(cfg, "synth.n_per_class", None)
    if v is None:
        v = cfg.get("SAMPLES_PER_CLASS", cfg.get("samples_per_class", default))  # type: ignore[attr_defined]
    return int(v)


# -----------------------------------------------------------------------------
# Image I/O
# -----------------------------------------------------------------------------
def _to_uint8(img01: np.ndarray) -> np.ndarray:
    """Convert float image in [0,1] to uint8 [0,255]."""
    return np.clip(np.round(img01 * 255.0), 0, 255).astype(np.uint8)


def _save_png(img01: np.ndarray, out_path: Path) -> None:
    """
    Save a single image (float in [0,1]) as PNG.

    Handles grayscale (H,W,1) and RGB (H,W,3).
    """
    ensure_dir(out_path.parent)

    x = img01
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
        mode = "L"
    elif x.ndim == 3 and x.shape[-1] == 3:
        mode = "RGB"
    else:
        x = x.squeeze()
        mode = "L"

    Image.fromarray(_to_uint8(x), mode=mode).save(out_path)


# -----------------------------------------------------------------------------
# Latent sampling (deterministic)
# -----------------------------------------------------------------------------
def _latents(n: int, dim: int, seed: int) -> np.ndarray:
    """
    Sample latent vectors z ~ N(0, I), deterministically given seed.

    NOTE: We use numpy Generator for stable behavior across platforms.
    """
    rng = np.random.default_rng(int(seed))
    return rng.normal(0.0, 1.0, size=(n, dim)).astype(np.float32)


# -----------------------------------------------------------------------------
# Checkpoint resolution
# -----------------------------------------------------------------------------
def _resolve_ckpt_dir_for_run(cfg: Mapping[str, Any], ctx: RunContext) -> Path:
    """
    Resolve generator checkpoint directory.

    Preferred (Rule A):
      ctx.run_dir/checkpoints/

    Fallbacks (legacy compatibility):
      - cfg.paths.gan_checkpoints
      - cfg.paths.checkpoints_root/<family>/<variant>/<dataset_id>/seed_<seed>
      - cfg.paths.artifacts/checkpoints/<family>/<variant>/<dataset_id>/seed_<seed>
      - cfg.paths.artifacts/<family>/checkpoints
    """
    if ctx.run_dir is None:
        raise ValueError("RunContext.run_dir must be set for sampling.")

    # 1) Preferred: run_local checkpoints
    run_ckpts = Path(ctx.run_dir) / "checkpoints"
    if run_ckpts.exists():
        return run_ckpts

    # 2) Explicit override (legacy)
    override = _cfg_get(cfg, "paths.gan_checkpoints", None)
    if override:
        return Path(str(override))

    # 3) Recommended checkpoints root override
    ckpts_root = _cfg_get(cfg, "paths.checkpoints_root", None)
    if ckpts_root:
        return Path(str(ckpts_root)) / FAMILY / VARIANT / str(ctx.dataset_id) / f"seed_{int(ctx.seed)}"

    # 4) New_ish artifacts/checkpoints layout (kept as compatibility)
    artifacts_root = Path(str(_cfg_get(cfg, "paths.artifacts", "artifacts")))
    candidate = artifacts_root / "checkpoints" / FAMILY / VARIANT / str(ctx.dataset_id) / f"seed_{int(ctx.seed)}"
    if candidate.exists():
        return candidate

    # 5) Legacy fallback
    return artifacts_root / FAMILY / "checkpoints"


def _load_generator_from_checkpoints(
    ckpt_dir: Path,
    *,
    img_shape: Tuple[int, int, int],
    num_classes: int,
    latent_dim: int,
    lr: float,
    beta_1: float,
    logger,
) -> tuple[tf.keras.Model, Optional[Path]]:
    """
    Build a compatible generator and load weights if present.

    Selection:
      - prefer:  G_best.weights.h5
      - fallback: G_last.weights.h5
      - else: random init (warn)
    """
    m = build_models(
        latent_dim=latent_dim,
        num_classes=num_classes,
        img_shape=img_shape,
        lr=lr,
        beta_1=beta_1,
    )
    G: tf.keras.Model = m["generator"]

    # Ensure weights exist (some Keras configs require a forward call)
    dummy_z = tf.zeros((1, latent_dim), dtype=tf.float32)
    dummy_y = tf.one_hot([0], depth=num_classes, dtype=tf.float32)
    _ = G([dummy_z, dummy_y], training=False)

    best = ckpt_dir / "G_best.weights.h5"
    last = ckpt_dir / "G_last.weights.h5"
    to_load = best if best.exists() else (last if last.exists() else None)

    if to_load is None:
        logger.warning(f"No generator checkpoint found in {ckpt_dir} (expected {best.name} or {last.name}).")
        logger.warning("Continuing with randomly initialized generator.")
        return G, None

    try:
        G.load_weights(str(to_load))
        logger.info(f"Loaded generator checkpoint: {to_load}")
        return G, to_load
    except Exception as e:
        logger.warning(f"Failed to load checkpoint {to_load.name}: {e}")
        logger.warning("Continuing with randomly initialized generator.")
        return G, None


# -----------------------------------------------------------------------------
# Core generation (per_class)
# -----------------------------------------------------------------------------
def _generate_class_images_01(
    G: tf.keras.Model,
    *,
    class_id: int,
    count: int,
    latent_dim: int,
    num_classes: int,
    seed: int,
) -> np.ndarray:
    """
    Generate images for one class.

    Returns
    -------
    np.ndarray
        float32 images in [0,1], shape (count, H, W, C)
    """
    z = _latents(count, latent_dim, seed=seed)

    y = tf.keras.utils.to_categorical(
        np.full((count,), class_id),
        num_classes=num_classes,
    ).astype(np.float32)

    g = G.predict([z, y], verbose=0)          # [-1,1]
    g01 = np.clip((g + 1.0) / 2.0, 0.0, 1.0)  # -> [0,1]
    return g01.astype(np.float32, copy=False)


# -----------------------------------------------------------------------------
# Preferred orchestrator entrypoint: sample(cfg, ctx) -> SampleResult
# -----------------------------------------------------------------------------
def sample(cfg: Dict[str, Any], ctx: RunContext) -> SampleResult:
    """
    Sample balanced synthetic images for this (dataset_id, model_tag, run_id).

    Writes (Rule A):
      ctx.run_dir/samples/generated/class_<k>/gan_00000.png
      ctx.run_dir/manifest.json
    """
    if ctx.run_dir is None or ctx.logs_dir is None:
        raise ValueError("RunContext must have run_dir and logs_dir resolved.")

    run_dir = Path(ctx.run_dir)
    log_dir = Path(ctx.logs_dir)

    # Run_scoped logger: artifacts/logs/<dataset_id>/<model_tag>/<run_id>/run.log
    logger = get_run_logger(name=f"{MODEL_TAG}:{ctx.run_id}:sample", log_dir=log_dir)
    logger.info("=== DCGAN SAMPLE START ===")
    logger.info(f"dataset_id={ctx.dataset_id} model_tag={ctx.model_tag} run_id={ctx.run_id} seed={ctx.seed}")
    logger.info(f"run_dir={run_dir}")

    # -----------------------------
    # Resolve config knobs
    # -----------------------------
    img_shape = tuple(_cfg_get(cfg, "IMG_SHAPE", _cfg_get(cfg, "img.shape", (40, 40, 1))))
    H, W, C = (int(img_shape[0]), int(img_shape[1]), int(img_shape[2]))
    K = int(_cfg_get(cfg, "NUM_CLASSES", _cfg_get(cfg, "num_classes", 9)))
    latent_dim = int(_cfg_get(cfg, "LATENT_DIM", _cfg_get(cfg, "gan.latent_dim", 100)))
    samples_per_class = _resolve_samples_per_class(cfg, default=25)

    # Hyperparams only needed to rebuild a compatible generator
    lr_g = _cfg_get(cfg, "gan.LR_G", None)
    lr = float(lr_g if lr_g is not None else _cfg_get(cfg, "LR", _cfg_get(cfg, "gan.lr", 2e_4)))

    betas = _cfg_get(cfg, "gan.BETAS", None)
    if isinstance(betas, list) and len(betas) >= 1:
        beta_1 = float(betas[0])
    else:
        beta_1 = float(_cfg_get(cfg, "BETA_1", _cfg_get(cfg, "gan.beta_1", 0.5)))

    logger.info(f"IMG_SHAPE={(H, W, C)} K={K} Z={latent_dim} samples_per_class={samples_per_class}")
    logger.info(f"arch_build: lr={lr} beta_1={beta_1}")

    # -----------------------------
    # Canonical output directory (Rule A)
    # -----------------------------
    # Evaluators/reporters can consistently find generated images here.
    out_root = ensure_dir(run_dir / "samples" / "generated")

    # -----------------------------
    # Seed everything (reproducible sampling)
    # -----------------------------
    np.random.seed(int(ctx.seed))
    tf.keras.utils.set_random_seed(int(ctx.seed))

    # -----------------------------
    # Load generator weights (prefer run_local checkpoints)
    # -----------------------------
    ckpt_dir = _resolve_ckpt_dir_for_run(cfg, ctx)
    G, ckpt_used = _load_generator_from_checkpoints(
        ckpt_dir,
        img_shape=(H, W, C),
        num_classes=K,
        latent_dim=latent_dim,
        lr=lr,
        beta_1=beta_1,
        logger=logger,
    )

    # -----------------------------
    # Generate & write per_class images
    # -----------------------------
    per_class_counts: Dict[str, int] = {str(k): 0 for k in range(K)}
    paths: List[Dict[str, Any]] = []

    for k in range(K):
        # Make sampling deterministic but not identical across classes:
        # use (seed + 1000*k) as a stable per_class offset.
        class_seed = int(ctx.seed) + 1000 * int(k)

        imgs01 = _generate_class_images_01(
            G,
            class_id=k,
            count=samples_per_class,
            latent_dim=latent_dim,
            num_classes=K,
            seed=class_seed,
        )

        class_dir = ensure_dir(out_root / f"class_{k}")
        for j in range(samples_per_class):
            p = class_dir / f"gan_{j:05d}.png"
            _save_png(imgs01[j], p)

            # Store paths relative to run_dir for portability across machines.
            paths.append({"path": str(p.relative_to(run_dir)), "label": int(k)})

        per_class_counts[str(k)] = int(samples_per_class)

    # -----------------------------
    # Write manifest.json to the run directory (Rule A)
    # -----------------------------
    manifest: Dict[str, Any] = {
        # Run identity (primary key)
        "dataset_id": str(ctx.dataset_id),
        "model_tag": str(ctx.model_tag),
        "run_id": str(ctx.run_id),
        "seed": int(ctx.seed),

        # Variant identity (helpful for cross_checking)
        "family": FAMILY,
        "variant": VARIANT,

        # Outputs
        "run_dir": str(run_dir),
        "samples_dir": str(run_dir / "samples" / "generated"),
        "paths": paths,
        "per_class_counts": per_class_counts,

        # Shape + budget
        "img_shape": [H, W, C],
        "num_classes": int(K),
        "latent_dim": int(latent_dim),
        "samples_per_class": int(samples_per_class),
        "num_fake": int(len(paths)),

        # Checkpoint provenance (auditability)
        "checkpoints_dir": str(ckpt_dir),
        "checkpoint_used": str(ckpt_used) if ckpt_used else None,
    }

    manifest_path = run_dir / "manifest.json"
    write_json(manifest_path, manifest, indent=2, sort_keys=True, atomic=True)
    logger.info(f"Wrote manifest: {manifest_path}")
    logger.info("=== DCGAN SAMPLE END ===")

    return SampleResult(
        ok=True,
        message="sampling complete",
        num_generated=int(len(paths)),
        manifest_path=str(manifest_path),
        extra={"per_class_counts": per_class_counts, "samples_dir": str(out_root)},
    )


# -----------------------------------------------------------------------------
# Backward_compatible wrapper (old API): synth(cfg, output_root, seed) -> dict
# -----------------------------------------------------------------------------
def synth(cfg: Dict[str, Any], output_root: str, seed: int = 42) -> Dict[str, Any]:
    """
    Legacy wrapper.

    Older code called:
        synth(cfg, output_root="artifacts", seed=42) -> manifest dict

    Under Rule A, we should NOT write to <output_root>/synth/...
    Instead we resolve a proper RunContext and write to:
        artifacts/runs/<dataset_id>/<model_tag>/<run_id>/...

    This wrapper:
    - injects paths.artifacts = output_root
    - injects model.tag = gan/dcgan
    - injects run_meta.seed = seed (if missing)
    - resolves run context
    - calls sample(cfg, ctx)
    - returns the manifest dict (loaded from manifest.json)
    """
    # Shallow copy so we don't mutate caller unexpectedly
    cfg2: Dict[str, Any] = dict(cfg)

    # Ensure artifacts root is what the legacy caller requested
    cfg2.setdefault("paths", {})
    if isinstance(cfg2["paths"], dict):
        cfg2["paths"]["artifacts"] = str(output_root)

    # Ensure model tag is stable (used by resolve_run_context)
    cfg2.setdefault("model", {})
    if isinstance(cfg2["model"], dict):
        cfg2["model"].setdefault("tag", MODEL_TAG)

    # Ensure seed is present in run_meta so context resolution is deterministic
    cfg2.setdefault("run_meta", {})
    if isinstance(cfg2["run_meta"], dict):
        cfg2["run_meta"].setdefault("seed", int(seed))

    # Resolve context and run sampling
    resolved = resolve_run_context(cfg2, create_dirs=True)
    ctx = resolved.ctx
    res = sample(resolved.cfg, ctx)

    # Return manifest dict for legacy compatibility
    mp = Path(res.manifest_path) if res.manifest_path else (Path(ctx.run_dir) / "manifest.json")  # type: ignore[arg_type]
    return json.loads(mp.read_text())


__all__ = ["sample", "synth"]