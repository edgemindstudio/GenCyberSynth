# src/gencysynth/models/diffusion/variants/c-ddpm/samply.py
"""
GenCyberSynth — Diffusion family — c-DDPM variant (Conditional) — Sampling / Synthesis
====================================================================================

RULE A (Scalable artifact policy)
---------------------------------
Everything is keyed by:
    (dataset_id, model_tag, run_id)

Therefore, sampling outputs MUST be written under the run directory:

  artifacts/
    runs/<dataset_id>/<model_tag>/<run_id>/
      samples/generated/class_<k>/diff_00000.png
      manifest.json

Logs MUST go to:
  artifacts/logs/<dataset_id>/<model_tag>/<run_id>/run.log

This module provides
--------------------
- save_grid_from_model(...)    : lightweight preview grid (debug helper)
- sample_batch(...)            : generate a batch (returns NumPy arrays)
- sample(cfg, ctx) -> SampleResult
    Preferred orchestrator entrypoint that:
      - loads the model weights from ctx.run_dir/checkpoints (preferred)
      - writes images under ctx.run_dir/samples/generated/...
      - writes ctx.run_dir/manifest.json
- synth(cfg, output_root, seed) -> dict
    Backward-compatible wrapper: resolves a RunContext then calls sample(...)

What this module MUST NOT do
----------------------------
- Evaluation metrics (belongs under gencysynth/eval/)
- Reporting / tables (belongs under reporting/ or papers/* scripts)
- Dataset loading (sampling only needs labels + reverse diffusion)

Model assumptions
-----------------
Assumes a Keras model built via diffusion.models.build_diffusion_model with signature:
    model([x_t, y_onehot, t_vec]) -> eps_hat  (predicted noise)

Conventions
-----------
- Images are channels-last (H, W, C).
- Reverse diffusion produces images in [-1, 1]; we rescale to [0, 1] for saving.
- Labels are one-hot vectors of length num_classes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

# NOTE: build_diffusion_model is imported lazily in _load_diffusion_from_checkpoints
# so this module remains import-safe for CLI tooling.

from gencysynth.models.base_types import RunContext, SampleResult
from gencysynth.orchestration.context import resolve_run_context
from gencysynth.orchestration.logger import get_run_logger
from gencysynth.utils.io import write_json
from gencysynth.utils.paths import ensure_dir


# -----------------------------------------------------------------------------
# Variant identity (must match folder + registry tag)
# -----------------------------------------------------------------------------
FAMILY: str = "diffusion"
VARIANT: str = "c-ddpm"
MODEL_TAG: str = "diffusion/c-ddpm"


# =============================================================================
# Config helpers
# =============================================================================
def _cfg_get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    """Read nested config values using a dotted key, e.g. 'paths.artifacts'."""
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
      1) synth.n_per_class   (repo-wide key)
      2) SAMPLES_PER_CLASS   (legacy)
      3) samples_per_class   (legacy)
      4) default
    """
    v = _cfg_get(cfg, "synth.n_per_class", None)
    if v is None:
        v = _cfg_get(cfg, "SAMPLES_PER_CLASS", _cfg_get(cfg, "samples_per_class", default))
    return int(v)


def _resolve_dataset_provenance(cfg: Mapping[str, Any]) -> str:
    """
    Provenance string recorded into manifest.

    Priority:
      1) data.root
      2) DATA_DIR
      3) unknown_dataset
    """
    return str(_cfg_get(cfg, "data.root", _cfg_get(cfg, "DATA_DIR", "unknown_dataset")))


# =============================================================================
# Image I/O helpers
# =============================================================================
def _to_uint8(img01: np.ndarray) -> np.ndarray:
    """Convert float image in [0,1] -> uint8 [0,255]."""
    return np.clip(np.rint(img01 * 255.0), 0, 255).astype(np.uint8)


def _save_png(img01: np.ndarray, out_path: Path) -> None:
    """
    Save a single image (float in [0,1]) as PNG.

    Handles:
    - grayscale (H,W,1) → mode "L"
    - rgb (H,W,3)      → mode "RGB"
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


# =============================================================================
# Diffusion schedule + reverse sampling (preview-friendly)
# =============================================================================
def _linear_alpha_hat_schedule(
    T: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> np.ndarray:
    """
    Linear beta schedule; returns ᾱ_t = ∏_{s<=t} (1 - β_s) for t=0..T-1.
    """
    betas = np.linspace(beta_start, beta_end, T, dtype=np.float32)
    alphas = 1.0 - betas
    alpha_hat = np.cumprod(alphas, axis=0)
    return alpha_hat.astype("float32")


def _reverse_diffuse(
    model: tf.keras.Model,
    *,
    y_onehot: np.ndarray,
    img_shape: Tuple[int, int, int],
    T: int,
    alpha_hat: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Perform a DDPM-style reverse diffusion pass given class-condition labels.

    Returns
    -------
    np.ndarray
        Images in [-1, 1], shape (B,H,W,C), float32.

    Notes
    -----
    This is a lightweight sampler intended for preview + baseline synthesis.
    (If you later add DDIM/CFG/advanced samplers, keep this as the simplest path.)
    """
    if seed is not None:
        np.random.seed(int(seed))
        tf.keras.utils.set_random_seed(int(seed))

    H, W, C = img_shape
    B = int(y_onehot.shape[0])

    if alpha_hat is None:
        alpha_hat = _linear_alpha_hat_schedule(T)
    alpha_hat_tf = tf.constant(alpha_hat, dtype=tf.float32)

    # Start from Gaussian noise at step T
    x = tf.random.normal((B, H, W, C))
    y = tf.convert_to_tensor(y_onehot, dtype=tf.float32)

    # Reverse process: t = T-1 ... 0
    for t in reversed(range(T)):
        t_vec = tf.fill([B], tf.cast(t, tf.int32))
        eps_pred = model([x, y, t_vec], training=False)  # ε̂(x_t, t, y)

        a = tf.reshape(alpha_hat_tf[t], (1, 1, 1, 1))   # ᾱ_t
        one_minus_a = 1.0 - a

        # Add noise except at t = 0
        noise = tf.random.normal(tf.shape(x)) if t > 0 else 0.0

        # Simple update using ᾱ_t (good enough for previews / consistent output)
        x = (x - (one_minus_a / tf.sqrt(one_minus_a)) * eps_pred) / tf.sqrt(a) + tf.sqrt(one_minus_a) * noise

    return x.numpy().astype("float32")


# =============================================================================
# Public sampling helpers (no filesystem assumptions)
# =============================================================================
def sample_batch(
    model: tf.keras.Model,
    *,
    num_samples: int,
    num_classes: int,
    img_shape: Tuple[int, int, int],
    T: int = 200,
    alpha_hat: Optional[np.ndarray] = None,
    class_ids: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a batch of conditional samples.

    Returns
    -------
    x_01 : np.ndarray
        float32 images in [0,1], shape (N,H,W,C)
    y_onehot : np.ndarray
        float32 labels, shape (N,num_classes)
    """
    if class_ids is None:
        class_ids = np.random.randint(0, num_classes, size=(num_samples,), dtype=np.int32)
    else:
        class_ids = np.asarray(class_ids, dtype=np.int32).reshape(-1)
        num_samples = int(class_ids.shape[0])

    y_onehot = tf.keras.utils.to_categorical(class_ids, num_classes=num_classes).astype("float32")

    # Reverse diffusion yields [-1,1]
    x_m11 = _reverse_diffuse(
        model,
        y_onehot=y_onehot,
        img_shape=img_shape,
        T=T,
        alpha_hat=alpha_hat,
        seed=seed,
    )

    # Rescale to [0,1] for saving/visualization
    x_01 = np.clip((x_m11 + 1.0) / 2.0, 0.0, 1.0).astype("float32")
    return x_01, y_onehot


def save_grid_from_model(
    model: tf.keras.Model,
    *,
    num_classes: int,
    img_shape: Tuple[int, int, int],
    path: Path,
    T: int = 200,
    alpha_hat: Optional[np.ndarray] = None,
    dpi: int = 200,
    titles: bool = True,
    seed: Optional[int] = None,
) -> Path:
    """
    Debug helper: generate ONE sample per class and save a horizontal grid PNG.

    This is intentionally NOT the canonical synthesis output layout (Rule A).
    Canonical per-class PNG outputs are produced by sample(...).
    """
    import matplotlib.pyplot as plt

    H, W, C = img_shape

    class_ids = np.arange(num_classes, dtype=np.int32)
    x_01, _ = sample_batch(
        model=model,
        num_samples=num_classes,
        num_classes=num_classes,
        img_shape=img_shape,
        T=T,
        alpha_hat=alpha_hat,
        class_ids=class_ids,
        seed=seed,
    )

    n = num_classes
    fig_w = max(1.2 * n, 6.0)
    fig_h = 1.6
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h))
    if n == 1:
        axes = [axes]

    for i in range(n):
        ax = axes[i]
        img = x_01[i]
        if C == 1:
            ax.imshow(img[:, :, 0], cmap="gray", vmin=0.0, vmax=1.0)
        else:
            ax.imshow(np.squeeze(img))
        ax.set_axis_off()
        if titles:
            ax.set_title(f"C{i}", fontsize=9)

    path = Path(path)
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


# =============================================================================
# Checkpoint loading (Rule A preferred, with safe fallbacks)
# =============================================================================
def _resolve_ckpt_dir_for_run(cfg: Mapping[str, Any], ctx: RunContext) -> Path:
    """
    Resolve checkpoint directory for sampling.

    Preferred (Rule A):
      <ctx.run_dir>/checkpoints/

    Fallbacks (legacy compatibility / optional shared layouts):
      - cfg.paths.diffusion_checkpoints (if you add it)
      - cfg.paths.checkpoints_root/<family>/<variant>/<dataset_id>/seed_<seed>
      - cfg.paths.artifacts/checkpoints/<family>/<variant>/<dataset_id>/seed_<seed>
      - cfg.paths.artifacts/<family>/checkpoints
    """
    assert ctx.run_dir is not None
    run_ckpts = Path(ctx.run_dir) / "checkpoints"
    if run_ckpts.exists():
        return run_ckpts

    override = _cfg_get(cfg, "paths.diffusion_checkpoints", None)
    if override:
        return Path(str(override))

    ckpts_root = _cfg_get(cfg, "paths.checkpoints_root", None)
    if ckpts_root:
        return Path(str(ckpts_root)) / FAMILY / VARIANT / str(ctx.dataset_id) / f"seed_{int(ctx.seed)}"

    artifacts_root = Path(str(_cfg_get(cfg, "paths.artifacts", "artifacts")))
    candidate = artifacts_root / "checkpoints" / FAMILY / VARIANT / str(ctx.dataset_id) / f"seed_{int(ctx.seed)}"
    if candidate.exists():
        return candidate

    return artifacts_root / FAMILY / "checkpoints"


def _load_diffusion_from_checkpoints(
    ckpt_dir: Path,
    *,
    img_shape: Tuple[int, int, int],
    num_classes: int,
    base_filters: int,
    depth: int,
    time_emb_dim: int,
    learning_rate: float,
    beta_1: float,
    logger,
) -> tuple[tf.keras.Model, Optional[Path]]:
    """
    Build diffusion model and try to load weights from {best → last}.

    Returns
    -------
    (model, ckpt_used)
      ckpt_used is None if no checkpoint was found or load failed.
    """
    from diffusion.models import build_diffusion_model  # local import

    H, W, C = img_shape
    model = build_diffusion_model(
        img_shape=img_shape,
        num_classes=num_classes,
        base_filters=base_filters,
        depth=depth,
        time_emb_dim=time_emb_dim,
        learning_rate=learning_rate,
        beta_1=beta_1,
    )

    # Build variables (Keras 3 friendly) by running one forward pass
    dummy_x = tf.zeros((1, H, W, C), dtype=tf.float32)
    dummy_y = tf.one_hot([0], depth=num_classes, dtype=tf.float32)
    dummy_t = tf.zeros((1,), dtype=tf.int32)
    _ = model([dummy_x, dummy_y, dummy_t], training=False)

    best = ckpt_dir / "DDPM_best.weights.h5"
    last = ckpt_dir / "DDPM_last.weights.h5"
    to_load = best if best.exists() else last if last.exists() else None

    if to_load is None:
        logger.warning(f"No DDPM checkpoint found in {ckpt_dir} (expected {best.name} or {last.name}).")
        logger.warning("Continuing with randomly initialized model.")
        return model, None

    try:
        model.load_weights(str(to_load))
        logger.info(f"Loaded checkpoint: {to_load}")
        return model, to_load
    except Exception as e:
        logger.warning(f"Failed to load checkpoint {to_load.name}: {e}")
        logger.warning("Continuing with randomly initialized model.")
        return model, None


# =============================================================================
# Preferred orchestrator entrypoint: sample(cfg, ctx) -> SampleResult
# =============================================================================
def sample(cfg: Dict[str, Any], ctx: RunContext) -> SampleResult:
    """
    Sample balanced synthetic images for this (dataset_id, model_tag, run_id).

    Writes:
      <ctx.run_dir>/samples/generated/class_<k>/diff_00000.png
      <ctx.run_dir>/manifest.json

    Returns SampleResult with manifest_path + summary info.
    """
    if ctx.run_dir is None or ctx.logs_dir is None:
        raise ValueError("RunContext must have run_dir and logs_dir resolved.")

    run_dir = Path(ctx.run_dir)
    log_dir = Path(ctx.logs_dir)

    logger = get_run_logger(name=f"{MODEL_TAG}:{ctx.run_id}:sample", log_dir=log_dir)
    logger.info("=== c-DDPM SAMPLE START ===")
    logger.info(f"dataset_id={ctx.dataset_id} model_tag={ctx.model_tag} run_id={ctx.run_id} seed={ctx.seed}")
    logger.info(f"run_dir={run_dir}")

    # -----------------------------
    # Resolve knobs (shape, classes, sampler steps, budgets)
    # -----------------------------
    img_shape = tuple(_cfg_get(cfg, "IMG_SHAPE", _cfg_get(cfg, "img.shape", (40, 40, 1))))
    H, W, C = int(img_shape[0]), int(img_shape[1]), int(img_shape[2])
    K = int(_cfg_get(cfg, "NUM_CLASSES", _cfg_get(cfg, "num_classes", 9)))
    S = int(_resolve_samples_per_class(cfg, default=25))

    # Reverse diffusion steps for sampling (keep name flexible)
    T = int(_cfg_get(cfg, "DIFFUSION_STEPS", _cfg_get(cfg, "diffusion.steps", _cfg_get(cfg, "diffusion.timesteps", 200))))

    # Model architecture knobs (must match training)
    base_filters = int(_cfg_get(cfg, "DIFF.base_filters", _cfg_get(cfg, "diffusion.base_filters", 64)))
    depth = int(_cfg_get(cfg, "DIFF.depth", _cfg_get(cfg, "diffusion.depth", 2)))
    time_dim = int(_cfg_get(cfg, "DIFF.time_dim", _cfg_get(cfg, "diffusion.time_emb_dim", 128)))

    # Optimizer params only needed to build a compatible model graph
    lr = float(_cfg_get(cfg, "LR", _cfg_get(cfg, "diffusion.lr", 2e-4)))
    beta_1 = float(_cfg_get(cfg, "BETA_1", _cfg_get(cfg, "diffusion.beta_1", 0.9)))

    logger.info(f"IMG_SHAPE={(H, W, C)} K={K} samples_per_class={S} sample_steps(T)={T}")
    logger.info(f"arch_build: base_filters={base_filters} depth={depth} time_dim={time_dim} lr={lr} beta_1={beta_1}")

    # Canonical output directory under this run (Rule A)
    out_root = ensure_dir(run_dir / "samples" / "generated")

    # Seed everything for reproducible sampling
    np.random.seed(int(ctx.seed))
    tf.keras.utils.set_random_seed(int(ctx.seed))

    # Precompute schedule once for speed
    alpha_hat = _linear_alpha_hat_schedule(T)

    # Load model from run-scoped checkpoints (preferred)
    ckpt_dir = _resolve_ckpt_dir_for_run(cfg, ctx)
    model, ckpt_used = _load_diffusion_from_checkpoints(
        ckpt_dir,
        img_shape=(H, W, C),
        num_classes=K,
        base_filters=base_filters,
        depth=depth,
        time_emb_dim=time_dim,
        learning_rate=lr,
        beta_1=beta_1,
        logger=logger,
    )

    # -----------------------------
    # Generate and write per-class images
    # -----------------------------
    per_class_counts: Dict[str, int] = {str(k): 0 for k in range(K)}
    paths: List[Dict[str, Any]] = []

    for k in range(K):
        # Per-class seed offset: deterministic but different across classes
        class_seed = int(ctx.seed) + 1000 * int(k)

        class_ids = np.full((S,), k, dtype=np.int32)
        x01, _ = sample_batch(
            model=model,
            num_samples=S,
            num_classes=K,
            img_shape=(H, W, C),
            T=T,
            alpha_hat=alpha_hat,
            class_ids=class_ids,
            seed=class_seed,
        )

        class_dir = ensure_dir(out_root / f"class_{k}")
        for j in range(S):
            p = class_dir / f"diff_{j:05d}.png"
            _save_png(x01[j], p)
            # Store relative paths for portability across machines
            paths.append({"path": str(p.relative_to(run_dir)), "label": int(k)})

        per_class_counts[str(k)] = int(S)

    # -----------------------------
    # Manifest (written to the run dir)
    # -----------------------------
    manifest: Dict[str, Any] = {
        # Run identity (Rule A keys)
        "dataset_id": str(ctx.dataset_id),
        "model_tag": str(ctx.model_tag),
        "run_id": str(ctx.run_id),
        "seed": int(ctx.seed),

        # Variant identity
        "family": FAMILY,
        "variant": VARIANT,

        # Dataset provenance (for auditability)
        "dataset": _resolve_dataset_provenance(cfg),

        # Output locations
        "run_dir": str(run_dir),
        "samples_dir": str(run_dir / "samples" / "generated"),
        "paths": paths,
        "per_class_counts": per_class_counts,

        # Shape + sampling budget
        "img_shape": [H, W, C],
        "num_classes": int(K),
        "samples_per_class": int(S),
        "num_fake": int(len(paths)),

        # Sampler knobs (useful for reproducibility)
        "sample_steps": int(T),

        # Checkpoint provenance
        "checkpoints_dir": str(ckpt_dir),
        "checkpoint_used": str(ckpt_used) if ckpt_used else None,
    }

    manifest_path = run_dir / "manifest.json"
    write_json(manifest_path, manifest, indent=2, sort_keys=True, atomic=True)
    logger.info(f"Wrote manifest: {manifest_path}")
    logger.info("=== c-DDPM SAMPLE END ===")

    return SampleResult(
        ok=True,
        message="sampling complete",
        num_generated=int(len(paths)),
        manifest_path=str(manifest_path),
        extra={"per_class_counts": per_class_counts, "samples_dir": str(out_root)},
    )


# =============================================================================
# Backward-compatible wrapper (old API): synth(cfg, output_root, seed) -> dict
# =============================================================================
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
    - injects model.tag = diffusion/c-ddpm
    - injects run_meta.seed = seed
    - resolves run context
    - calls sample(cfg, ctx)
    - returns the manifest dict (loaded from manifest.json)
    """
    cfg = dict(cfg)  # shallow copy to avoid mutating caller unexpectedly

    cfg.setdefault("paths", {})
    if isinstance(cfg["paths"], dict):
        cfg["paths"]["artifacts"] = str(output_root)

    cfg.setdefault("model", {})
    if isinstance(cfg["model"], dict):
        cfg["model"].setdefault("tag", MODEL_TAG)
        cfg["model"].setdefault("family", FAMILY)
        cfg["model"].setdefault("variant", VARIANT)

    cfg.setdefault("run_meta", {})
    if isinstance(cfg["run_meta"], dict):
        cfg["run_meta"].setdefault("seed", int(seed))

    resolved = resolve_run_context(cfg, create_dirs=True)
    res = sample(resolved.cfg, resolved.ctx)

    mp = Path(res.manifest_path) if res.manifest_path else (Path(resolved.ctx.run_dir) / "manifest.json")
    return json.loads(mp.read_text())


__all__ = ["sample_batch", "save_grid_from_model", "sample", "synth"]
