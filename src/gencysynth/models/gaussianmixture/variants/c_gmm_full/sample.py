# src/gencysynth/models/gaussianmixture/variants/c_gmm_full/samply.py
"""
GenCyberSynth — GaussianMixture — c_gmm_full — Sampling / Synthesis
==================================================================

RULE A (Scalable artifact policy)
---------------------------------
Everything is keyed by:
    (dataset_id, model_tag, run_id)

Therefore, sampling outputs MUST be written under:

  artifacts/runs/<dataset_id>/<model_tag>/<run_id>/
    samples/generated/class_<k>/gmm_00000.png
    manifest.json

Logs MUST go to:
  artifacts/logs/<dataset_id>/<model_tag>/<run_id>/run.log

This module provides
--------------------
- sample(cfg, ctx) -> SampleResult     (preferred orchestrator entrypoint)
- synth(cfg, output_root, seed) -> dict (legacy wrapper, Rule A compliant)

It does NOT do evaluation; evaluation belongs under gencysynth/eval/.

Conventions
-----------
- Images are channels_last (H, W, C) with values in [0, 1].
- Checkpoints (joblib) are one per class:
      GMM_class_{k}.joblib
  plus optional:
      GMM_global_fallback.joblib
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import joblib
import numpy as np
from PIL import Image
from sklearn.mixture import GaussianMixture

from gencysynth.models.base_types import RunContext, SampleResult
from gencysynth.orchestration.context import resolve_run_context
from gencysynth.orchestration.logger import get_run_logger
from gencysynth.utils.io import write_json
from gencysynth.utils.paths import ensure_dir


# -----------------------------------------------------------------------------
# Variant identity (must match folder structure and registry tag)
# -----------------------------------------------------------------------------
FAMILY: str = "gaussianmixture"
VARIANT: str = "c_gmm_full"
MODEL_TAG: str = "gaussianmixture/c_gmm_full"


# -----------------------------------------------------------------------------
# Config helper utilities
# -----------------------------------------------------------------------------
def _cfg_get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    """Read nested config values using a dotted key, e.g. 'paths.artifacts'."""
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _resolve_samples_per_class(cfg: Dict[str, Any], default: int = 25) -> int:
    """
    Resolve synthesis budget.

    Priority:
      1) synth.n_per_class      (repo_wide key)
      2) SAMPLES_PER_CLASS      (legacy)
      3) samples_per_class      (legacy)
      4) default
    """
    v = _cfg_get(cfg, "synth.n_per_class", None)
    if v is None:
        v = cfg.get("SAMPLES_PER_CLASS", cfg.get("samples_per_class", default))
    return int(v)


# -----------------------------------------------------------------------------
# Image I/O (PNG)
# -----------------------------------------------------------------------------
def _to_uint8(img01: np.ndarray) -> np.ndarray:
    """Convert float image in [0,1] to uint8 [0,255]."""
    return np.clip(np.round(img01 * 255.0), 0, 255).astype(np.uint8)


def _save_png(img01: np.ndarray, out_path: Path) -> None:
    """Save a single image (float in [0,1]) as PNG."""
    ensure_dir(out_path.parent)

    x = np.asarray(img01, dtype=np.float32)
    x = np.clip(x, 0.0, 1.0)

    if x.ndim == 3 and x.shape[-1] == 1:
        x2 = x[..., 0]
        mode = "L"
    elif x.ndim == 3 and x.shape[-1] == 3:
        x2 = x
        mode = "RGB"
    else:
        x2 = x.squeeze()
        mode = "L"

    Image.fromarray(_to_uint8(x2), mode=mode).save(out_path)


# -----------------------------------------------------------------------------
# GMM loading (Rule A + legacy fallbacks)
# -----------------------------------------------------------------------------
def _resolve_ckpt_dir_for_run(cfg: Dict[str, Any], ctx: RunContext) -> Path:
    """
    Resolve checkpoint directory for this run.

    Preferred (Rule A):
      ctx.run_dir/checkpoints/

    Fallbacks (legacy compatibility):
      - cfg.paths.gaussianmixture_checkpoints
      - cfg.ARTIFACTS.gaussianmixture_checkpoints
      - cfg.paths.checkpoints_root/<family>/<variant>/<dataset_id>/seed_<seed>
      - cfg.paths.artifacts/checkpoints/<family>/<variant>/<dataset_id>/seed_<seed>
      - cfg.paths.artifacts/<family>/checkpoints
    """
    assert ctx.run_dir is not None
    run_ckpts = Path(ctx.run_dir) / "checkpoints"
    if run_ckpts.exists():
        return run_ckpts

    override = _cfg_get(cfg, "paths.gaussianmixture_checkpoints", None)
    if override:
        return Path(override)

    override2 = _cfg_get(cfg, "ARTIFACTS.gaussianmixture_checkpoints", None)
    if override2:
        return Path(override2)

    ckpts_root = _cfg_get(cfg, "paths.checkpoints_root", None)
    if ckpts_root:
        return Path(ckpts_root) / FAMILY / VARIANT / str(ctx.dataset_id) / f"seed_{int(ctx.seed)}"

    artifacts_root = Path(_cfg_get(cfg, "paths.artifacts", "artifacts"))
    candidate = artifacts_root / "checkpoints" / FAMILY / VARIANT / str(ctx.dataset_id) / f"seed_{int(ctx.seed)}"
    if candidate.exists():
        return candidate

    return artifacts_root / FAMILY / "checkpoints"


def load_gmms_from_dir(
    ckpt_dir: Path | str,
    num_classes: int,
) -> tuple[list[Optional[GaussianMixture]], Optional[GaussianMixture]]:
    """
    Load per_class GMMs from a checkpoint directory.

    Returns
    -------
    (models, global_fallback)
        models: list length K; each item a GaussianMixture or None if missing
        global_fallback: a GaussianMixture or None if not present
    """
    ckpt_dir = Path(ckpt_dir)
    models: List[Optional[GaussianMixture]] = [None] * int(num_classes)

    for k in range(num_classes):
        p = ckpt_dir / f"GMM_class_{k}.joblib"
        if p.exists():
            models[k] = joblib.load(p)

    fb_path = ckpt_dir / "GMM_global_fallback.joblib"
    fallback = joblib.load(fb_path) if fb_path.exists() else None
    return models, fallback


# -----------------------------------------------------------------------------
# Sampling (balanced per class)
# -----------------------------------------------------------------------------
def _clip_and_sanitize(x: np.ndarray) -> np.ndarray:
    """Clamp to [0,1] and replace non_finite values with 0 (rare, but safe)."""
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, 0.0, 1.0)
    mask = np.isfinite(x)
    if not mask.all():
        x = np.where(mask, x, 0.0)
    return x


def _reshape_to_images(flat: np.ndarray, img_shape: Tuple[int, int, int]) -> np.ndarray:
    """(N,D) -> (N,H,W,C) float32."""
    H, W, C = img_shape
    flat = np.asarray(flat, dtype=np.float32)
    imgs = flat.reshape((-1, H, W, C))
    return np.clip(imgs, 0.0, 1.0).astype(np.float32, copy=False)


def sample_balanced_from_models(
    models: list[Optional[GaussianMixture]],
    *,
    img_shape: Tuple[int, int, int],
    samples_per_class: int,
    seed: int,
    global_fallback: Optional[GaussianMixture] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw a class_balanced batch from per_class GMMs.

    Returns
    -------
    x_synth : float32 (N, H, W, C) in [0,1]
    y_onehot: float32 (N, K)        one_hot labels
    """
    H, W, C = img_shape
    K = len(models)
    per_class = int(samples_per_class)

    rng = np.random.default_rng(int(seed))

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for k in range(K):
        gmm = models[k] if models[k] is not None else global_fallback
        if gmm is None:
            raise FileNotFoundError(f"No GMM for class {k} and no global fallback provided.")

        # Ensure deterministic per_class but non_identical streams:
        # (seed + 1000*k) pattern matches your GAN sampler style.
        rs = int(seed) + 1000 * int(k)

        flat, _ = gmm.sample(per_class, random_state=rs)  # (per_class, D)
        flat = _clip_and_sanitize(flat)
        imgs = _reshape_to_images(flat, (H, W, C))        # (per_class, H, W, C)

        y1h = np.zeros((per_class, K), dtype=np.float32)
        y1h[:, k] = 1.0

        xs.append(imgs)
        ys.append(y1h)

    x_synth = np.concatenate(xs, axis=0).astype(np.float32)
    y_onehot = np.concatenate(ys, axis=0).astype(np.float32)
    return x_synth, y_onehot


# -----------------------------------------------------------------------------
# Preferred orchestrator entrypoint: sample(cfg, ctx) -> SampleResult
# -----------------------------------------------------------------------------
def sample(cfg: Dict[str, Any], ctx: RunContext) -> SampleResult:
    """
    Sample balanced synthetic images for this (dataset_id, model_tag, run_id).

    Writes:
      ctx.run_dir/samples/generated/class_<k>/gmm_00000.png
      ctx.run_dir/manifest.json
    """
    if ctx.run_dir is None or ctx.logs_dir is None:
        raise ValueError("RunContext must have run_dir and logs_dir resolved.")

    run_dir = Path(ctx.run_dir)
    log_dir = Path(ctx.logs_dir)

    # Run_scoped logger
    logger = get_run_logger(name=f"{MODEL_TAG}:{ctx.run_id}:sample", log_dir=log_dir)
    logger.info("=== c_gmm_full SAMPLE START ===")
    logger.info(f"dataset_id={ctx.dataset_id} model_tag={ctx.model_tag} run_id={ctx.run_id} seed={ctx.seed}")
    logger.info(f"run_dir={run_dir}")

    # Resolve config knobs
    img_shape = tuple(_cfg_get(cfg, "IMG_SHAPE", _cfg_get(cfg, "img.shape", (40, 40, 1))))
    H, W, C = (int(img_shape[0]), int(img_shape[1]), int(img_shape[2]))
    K = int(_cfg_get(cfg, "NUM_CLASSES", _cfg_get(cfg, "num_classes", 9)))
    samples_per_class = _resolve_samples_per_class(cfg, default=25)

    logger.info(f"IMG_SHAPE={(H, W, C)} K={K} samples_per_class={samples_per_class}")

    # Canonical output directory under this run (Rule A)
    out_root = ensure_dir(run_dir / "samples" / "generated")

    # Resolve checkpoints (Rule A first)
    ckpt_dir = _resolve_ckpt_dir_for_run(cfg, ctx)
    logger.info(f"checkpoints_dir={ckpt_dir}")

    models, fallback = load_gmms_from_dir(ckpt_dir, K)

    # Generate balanced batch in_memory
    x_synth, y_onehot = sample_balanced_from_models(
        models,
        img_shape=(H, W, C),
        samples_per_class=samples_per_class,
        seed=int(ctx.seed),
        global_fallback=fallback,
    )

    labels_int = np.argmax(y_onehot, axis=1).astype(int)

    # Write per_class PNGs and manifest paths (relative to run_dir)
    per_class_counts: Dict[str, int] = {str(k): 0 for k in range(K)}
    paths: List[Dict[str, Any]] = []

    for k in range(K):
        idxs = np.where(labels_int == k)[0]
        class_dir = ensure_dir(out_root / f"class_{k}")
        for j, i in enumerate(idxs):
            p = class_dir / f"gmm_{j:05d}.png"
            _save_png(x_synth[i], p)
            paths.append({"path": str(p.relative_to(run_dir)), "label": int(k)})
        per_class_counts[str(k)] = int(idxs.size)

    # Manifest written to run directory (Rule A)
    manifest: Dict[str, Any] = {
        # Identity
        "dataset_id": str(ctx.dataset_id),
        "model_tag": str(ctx.model_tag),
        "run_id": str(ctx.run_id),
        "seed": int(ctx.seed),

        # Variant identity
        "family": FAMILY,
        "variant": VARIANT,

        # Outputs
        "run_dir": str(run_dir),
        "samples_dir": str((run_dir / "samples" / "generated")),
        "paths": paths,
        "per_class_counts": per_class_counts,

        # Shape + budget
        "img_shape": [H, W, C],
        "num_classes": int(K),
        "samples_per_class": int(samples_per_class),
        "num_fake": int(len(paths)),

        # Checkpoint provenance
        "checkpoints_dir": str(ckpt_dir),
        "global_fallback_used": bool(fallback is not None),
    }

    manifest_path = run_dir / "manifest.json"
    write_json(manifest_path, manifest, indent=2, sort_keys=True, atomic=True)
    logger.info(f"Wrote manifest: {manifest_path}")
    logger.info("=== c_gmm_full SAMPLE END ===")

    return SampleResult(
        ok=True,
        message="sampling complete",
        num_generated=int(len(paths)),
        manifest_path=str(manifest_path),
        extra={"per_class_counts": per_class_counts, "samples_dir": str(out_root)},
    )


# -----------------------------------------------------------------------------
# Backward_compatible wrapper (old API)
# -----------------------------------------------------------------------------
def synth(cfg: Dict[str, Any], output_root: str, seed: int = 42) -> Dict[str, Any]:
    """
    Legacy wrapper.

    Older code called:
        synth(cfg, output_root="artifacts", seed=42) -> manifest dict

    Under Rule A, we should NOT write to <output_root>/<class>/<seed>/...
    Instead we resolve a proper RunContext and write to:
        artifacts/runs/<dataset_id>/<model_tag>/<run_id>/...

    This wrapper:
    - injects paths.artifacts = output_root
    - injects model.tag = gaussianmixture/c_gmm_full
    - injects run_meta.seed
    - resolves run context
    - calls sample(cfg, ctx)
    - returns the manifest dict (loaded from manifest.json)
    """
    cfg = dict(cfg)  # shallow copy so we don't mutate caller unexpectedly
    cfg.setdefault("paths", {})
    if isinstance(cfg["paths"], dict):
        cfg["paths"]["artifacts"] = str(output_root)

    cfg.setdefault("model", {})
    if isinstance(cfg["model"], dict):
        cfg["model"].setdefault("tag", MODEL_TAG)

    cfg.setdefault("run_meta", {})
    if isinstance(cfg["run_meta"], dict):
        cfg["run_meta"].setdefault("seed", int(seed))

    resolved = resolve_run_context(cfg, create_dirs=True)
    ctx = resolved.ctx
    res = sample(resolved.cfg, ctx)

    mp = Path(res.manifest_path) if res.manifest_path else (Path(ctx.run_dir) / "manifest.json")
    return json.loads(mp.read_text())


__all__ = [
    "load_gmms_from_dir",
    "sample_balanced_from_models",
    "sample",
    "synth",
]