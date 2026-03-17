# src/gencysynth/models/maskedautoflow/variants/maf_affine/sample.py

"""
GenCyberSynth — MaskedAutoFlow — maf_affine — Sampling (Rule A)

This module provides:
- Checkpoint loading for a trained MAF (best → last).
- Unconditional sampling in flattened space, reshaped back to images in [0,1].
- Two output modes:
    1) Unified_CLI `synth(cfg, output_root, seed)`:
         Writes PNGs to {output_root}/{class}/{seed}/... and returns a manifest.
    2) Evaluator_friendly `.npy` dumps (optional helper):
         Writes per_class gen_class_k.npy / labels_class_k.npy + x_synth/y_synth.

Rule A (artifacts) contract
---------------------------
We keep ALL model_owned artifacts dataset_aware:

  <paths.artifacts>/<data.name>/<model.family>/<model.variant>/
      checkpoints/   (read: MAF_best.weights.h5 / MAF_last.weights.h5)
      summaries/     (write: preview PNGs, small markers, logs)
      synthetic/     (optional: evaluator_friendly .npy dumps, if used)

Unified_CLI output contract
---------------------------
The orchestrator passes `output_root` and we write:

  {output_root}/{class_id}/{seed}/maf_00000.png
  {output_root}/{class_id}/{seed}/maf_00001.png
  ...

We do NOT decide where `output_root` is; we only write inside it.

Config compatibility
--------------------
Supports BOTH:
- NEW keys:
    data.root, data.name
    model.family, model.variant
    model.img_shape, model.num_classes
    model.num_flows, model.hidden_dims
    train.seed, synth.samples_per_class
    paths.artifacts
- LEGACY keys:
    DATA_DIR, DATASET, IMG_SHAPE, NUM_CLASSES, NUM_FLOWS, HIDDEN_DIMS, SAMPLES_PER_CLASS
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime

import numpy as np
import tensorflow as tf
from PIL import Image

from maskedautoflow.models import (
    MAF,
    MAFConfig,
    build_maf_model,
    reshape_to_images,
)


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def _ensure_dir(p: Path) -> None:
    """mkdir -p."""
    p.mkdir(parents=True, exist_ok=True)


def _cfg_get(cfg: Dict[str, Any], dotted: str, default=None):
    """Read nested dict values using dot notation (e.g., 'model.img_shape')."""
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _to_uint8(img01: np.ndarray) -> np.ndarray:
    """[0,1] float -> uint8."""
    return np.clip(np.round(img01 * 255.0), 0, 255).astype(np.uint8)


def _save_png(img01: np.ndarray, out_path: Path) -> None:
    """
    Save a single (H,W,C) image in [0,1] to PNG.
    Supports grayscale (C=1) and RGB (C=3).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.asarray(img01, dtype=np.float32)
    if x.ndim == 3 and x.shape[-1] == 1:
        Image.fromarray(_to_uint8(x[..., 0]), mode="L").save(out_path)
    elif x.ndim == 3 and x.shape[-1] == 3:
        Image.fromarray(_to_uint8(x), mode="RGB").save(out_path)
    else:
        # fallback: squeeze to 2D grayscale
        Image.fromarray(_to_uint8(x.squeeze()), mode="L").save(out_path)


# -----------------------------------------------------------------------------
# Rule A: dataset_aware artifact paths
# -----------------------------------------------------------------------------
def _resolve_artifact_paths(cfg: Dict[str, Any]) -> tuple[Path, Path, Path]:
    """
    Resolve artifact directories following Rule A.

    Base:
      <paths.artifacts>/<data.name>/<model.family>/<model.variant>/

    Subdirs:
      checkpoints/  (read for sampling)
      summaries/    (write preview images)
      synthetic/    (optional .npy dumps)

    Optional overrides:
      artifacts.checkpoints / artifacts.summaries / artifacts.synthetic
    """
    artifacts_root = Path(_cfg_get(cfg, "paths.artifacts", _cfg_get(cfg, "PATHS.artifacts", "artifacts")))

    dataset_name = str(_cfg_get(cfg, "data.name", _cfg_get(cfg, "DATASET", _cfg_get(cfg, "dataset", "dataset"))))
    family = str(_cfg_get(cfg, "model.family", "maskedautoflow"))
    variant = str(_cfg_get(cfg, "model.variant", "maf_affine"))

    base = artifacts_root / dataset_name / family / variant

    ckpt_dir = Path(_cfg_get(cfg, "artifacts.checkpoints", base / "checkpoints"))
    sums_dir = Path(_cfg_get(cfg, "artifacts.summaries", base / "summaries"))
    synth_dir = Path(_cfg_get(cfg, "artifacts.synthetic", base / "synthetic"))

    return ckpt_dir, sums_dir, synth_dir


# -----------------------------------------------------------------------------
# Checkpoint loading
# -----------------------------------------------------------------------------
def load_maf_from_checkpoints(
    ckpt_dir: Path | str,
    img_shape: Tuple[int, int, int],
    *,
    num_flows: int = 5,
    hidden_dims: Tuple[int, ...] = (128, 128),
) -> MAF:
    """
    Instantiate a MAF and load weights from best→last checkpoint.

    Expected files
    --------------
    ckpt_dir/
      MAF_best.weights.h5   (preferred)
      MAF_last.weights.h5   (fallback)

    Raises
    ------
    FileNotFoundError if neither checkpoint exists.
    """
    ckpt_dir = Path(ckpt_dir)
    H, W, C = img_shape
    D = H * W * C

    # Build the model exactly as training did (same config)
    model = build_maf_model(
        MAFConfig(IMG_SHAPE=img_shape, NUM_FLOWS=int(num_flows), HIDDEN_DIMS=tuple(hidden_dims))
    )

    # Create variables (Keras 3 requires a first call before load_weights)
    _ = model(tf.zeros((1, D), dtype=tf.float32))

    best = ckpt_dir / "MAF_best.weights.h5"
    last = ckpt_dir / "MAF_last.weights.h5"
    to_load = best if best.exists() else last

    if not to_load.exists():
        raise FileNotFoundError(f"No MAF checkpoint found under: {ckpt_dir}")

    model.load_weights(str(to_load))
    print(f"[ckpt] loaded {to_load.name} from {ckpt_dir}")
    return model


# -----------------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------------
def sample_unconditional(
    model: MAF,
    n_total: int,
    img_shape: Tuple[int, int, int],
    *,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Draw unconditional samples from the flow and reshape to images.

    Returns
    -------
    x : float32 array, shape (n_total, H, W, C) with values in [0,1]
    """
    H, W, C = img_shape
    D = H * W * C

    if seed is not None:
        tf.keras.utils.set_random_seed(int(seed))
        np.random.seed(int(seed))

    # Sample latent z ~ N(0, I) and map back via inverse flow
    z = tf.random.normal(shape=(int(n_total), D), dtype=tf.float32, seed=None if seed is None else int(seed))
    x_flat = model.inverse(z).numpy().astype(np.float32, copy=False)

    # Clamp to valid pixel range
    x_flat = np.clip(x_flat, 0.0, 1.0)
    return reshape_to_images(x_flat, img_shape, clip=True)


# -----------------------------------------------------------------------------
# Optional: evaluator_friendly .npy dumps (NOT used by unified synth output_root)
# -----------------------------------------------------------------------------
def write_balanced_per_class(
    x: np.ndarray,
    *,
    num_classes: int,
    per_class: int,
    synth_dir: Path | str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split unconditional samples into K equal blocks and write evaluator .npy files.

    This is helpful when your evaluator expects:
      gen_class_k.npy, labels_class_k.npy, x_synth.npy, y_synth.npy

    Parameters
    ----------
    x : (K*per_class, H, W, C) float32 in [0,1]
    num_classes : K
    per_class : samples per class
    synth_dir : destination folder (Rule A synthetic/)

    Returns
    -------
    x_synth : (K*per_class, H, W, C)
    y_synth : (K*per_class, K) one_hot
    """
    synth_dir = Path(synth_dir)
    _ensure_dir(synth_dir)

    xs, ys = [], []
    for k in range(int(num_classes)):
        start, end = k * int(per_class), (k + 1) * int(per_class)
        xk = x[start:end]

        np.save(synth_dir / f"gen_class_{k}.npy", xk)
        np.save(synth_dir / f"labels_class_{k}.npy", np.full((len(xk),), k, dtype=np.int32))

        y1h = np.zeros((len(xk), int(num_classes)), dtype=np.float32)
        y1h[:, k] = 1.0

        xs.append(xk)
        ys.append(y1h)

    x_synth = np.concatenate(xs, axis=0).astype(np.float32, copy=False)
    y_synth = np.concatenate(ys, axis=0).astype(np.float32, copy=False)

    np.save(synth_dir / "x_synth.npy", x_synth)
    np.save(synth_dir / "y_synth.npy", y_synth)

    print(f"[npy] wrote {x_synth.shape[0]} samples ({per_class}/class) -> {synth_dir}")
    return x_synth, y_synth


def save_preview_row_png(x: np.ndarray, out_path: Path) -> None:
    """
    Save a compact preview row PNG (one image per class).

    Args
    ----
    x : (K, H, W, C) float32 in [0,1]
    """
    import matplotlib.pyplot as plt

    k = int(x.shape[0])
    x_vis = x[..., 0] if x.shape[-1] == 1 else x

    fig, axes = plt.subplots(1, k, figsize=(1.4 * k, 1.6))
    if k == 1:
        axes = [axes]

    for i in range(k):
        axes[i].imshow(x_vis[i], cmap="gray" if x.shape[-1] == 1 else None, vmin=0.0, vmax=1.0)
        axes[i].set_axis_off()
        axes[i].set_title(f"C{i}", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Unified CLI synth (writes PNGs under output_root; reads ckpts via Rule A)
# -----------------------------------------------------------------------------
def synth(cfg: Dict[str, Any], output_root: str, seed: int = 42) -> Dict[str, Any]:
    """
    Unified_CLI entrypoint.

    Behavior
    --------
    1) Resolve model/dataset config (new + legacy keys).
    2) Resolve Rule_A artifacts for checkpoint + summaries (preview).
    3) Load the trained MAF checkpoint.
    4) Sample K*S unconditional images.
    5) Write PNGs to:
         {output_root}/{class}/{seed}/maf_00000.png
    6) Return a manifest compatible with your aggregator.

    Note
    ----
    Since MAF is unconditional, we *assign* samples to classes by slicing
    the batch into K equal contiguous blocks.
    """
    # ----- Resolve core settings (NEW → LEGACY fallbacks) -----
    H, W, C = tuple(_cfg_get(cfg, "model.img_shape", _cfg_get(cfg, "IMG_SHAPE", _cfg_get(cfg, "img.shape", (40, 40, 1)))))
    K = int(_cfg_get(cfg, "model.num_classes", _cfg_get(cfg, "NUM_CLASSES", _cfg_get(cfg, "num_classes", 9))))
    S = int(_cfg_get(cfg, "synth.samples_per_class", _cfg_get(cfg, "SAMPLES_PER_CLASS", _cfg_get(cfg, "samples_per_class", 25))))

    num_flows = int(_cfg_get(cfg, "model.num_flows", _cfg_get(cfg, "NUM_FLOWS", 5)))
    hidden_dims_cfg = _cfg_get(cfg, "model.hidden_dims", _cfg_get(cfg, "HIDDEN_DIMS", (128, 128)))
    hidden_dims = tuple(int(h) for h in hidden_dims_cfg)

    dataset_root = str(_cfg_get(cfg, "data.root", _cfg_get(cfg, "DATA_DIR", "data")))
    dataset_name = str(_cfg_get(cfg, "data.name", _cfg_get(cfg, "DATASET", _cfg_get(cfg, "dataset", "dataset"))))

    # ----- Rule A artifacts (where checkpoints & summaries live) -----
    ckpt_dir, sums_dir, synth_dir = _resolve_artifact_paths(cfg)
    _ensure_dir(ckpt_dir)
    _ensure_dir(sums_dir)
    _ensure_dir(synth_dir)

    # ----- Load model and sample -----
    model = load_maf_from_checkpoints(
        ckpt_dir=ckpt_dir,
        img_shape=(H, W, C),
        num_flows=num_flows,
        hidden_dims=hidden_dims,
    )

    x = sample_unconditional(model, n_total=int(K) * int(S), img_shape=(H, W, C), seed=int(seed))

    # ----- Preview (one per class) written under Rule A summaries -----
    try:
        save_preview_row_png(x[:K], sums_dir / "maf_synth_preview.png")
    except Exception as e:
        print(f"[warn] preview failed: {type(e).__name__}: {e}")

    # ----- Unified_CLI outputs under output_root -----
    out_root = Path(output_root)
    per_class_counts: Dict[str, int] = {str(k): 0 for k in range(int(K))}
    paths: List[Dict[str, Any]] = []

    for k in range(int(K)):
        start, end = k * int(S), (k + 1) * int(S)
        cls_imgs = x[start:end]

        cls_dir = out_root / str(k) / str(seed)
        cls_dir.mkdir(parents=True, exist_ok=True)

        for j in range(int(cls_imgs.shape[0])):
            p = cls_dir / f"maf_{j:05d}.png"
            _save_png(cls_imgs[j], p)
            paths.append({"path": str(p), "label": int(k)})

        per_class_counts[str(k)] = int(cls_imgs.shape[0])

    # ----- Optional: evaluator_friendly npy dumps into Rule A synthetic/ -----
    # Uncomment if/when your evaluator wants these dumps for MAF:
    # write_balanced_per_class(x, num_classes=K, per_class=S, synth_dir=synth_dir)

    manifest: Dict[str, Any] = {
        "dataset": dataset_root,
        "dataset_name": dataset_name,
        "model_family": str(_cfg_get(cfg, "model.family", "maskedautoflow")),
        "model_variant": str(_cfg_get(cfg, "model.variant", "maf_affine")),
        "seed": int(seed),
        "per_class_counts": per_class_counts,
        "paths": paths,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    return manifest


__all__ = [
    "load_maf_from_checkpoints",
    "sample_unconditional",
    "write_balanced_per_class",
    "save_preview_row_png",
    "synth",
]