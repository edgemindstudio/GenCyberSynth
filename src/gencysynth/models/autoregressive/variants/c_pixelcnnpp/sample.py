# src/gencysynth/models/autoregressive/variants/c_pixelcnnpp/sample.py

"""
GenCyberSynth — Autoregressive — c_pixelcnnpp — Sampling + Unified CLI synth
============================================================================

This module provides:
  - load_ar_from_checkpoints(...): build model + load weights (best→last)
  - save_grid_from_ar(...): quick preview PNG grid
  - synth(cfg, output_root, seed): unified_CLI entrypoint that writes PNGs + manifest

Rule A (Artifacts + scalability)
--------------------------------
Rule A standardizes *where* artifacts live so we can scale across datasets and
model variants cleanly.

Preferred layout (Rule A):
  artifacts/runs/<dataset_id>/<model_tag>/<run_id>/
    checkpoints/   # training writes here
    summaries/     # previews / logs
    synthetic/     # (optional for .npy dumps; this file writes PNGs to output_root)
    tensorboard/

This sampler:
- *reads checkpoints* from Rule_A run directories when run context is provided
- falls back to legacy checkpoint locations when run context is not provided

Back_compat: if Rule A keys are missing, we use:
  <paths.artifacts>/autoregressive/checkpoints
or ARTIFACTS.autoregressive_checkpoints if explicitly set.

Output contract (Unified CLI)
-----------------------------
`synth(cfg, output_root, seed)` writes:
  {output_root}/{class}/{seed}/ar_00000.png ...
and returns a manifest:
  {
    "dataset": <dataset_id>,
    "seed": <int>,
    "per_class_counts": {"0": S, ...},
    "paths": [{"path": "...png", "label": k}, ...],
    "created_at": "...",
    "model_tag": "...",
    "run_id": "...",
  }

Notes
-----
- Raster_scan sampling is sequential over pixels and therefore slow; that's OK for
  small preview/sample jobs and consistent with the baseline.
- The model is expected to output probabilities in [0,1] with same shape as x.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Mapping

import numpy as np
import tensorflow as tf
from PIL import Image

import matplotlib.pyplot as plt

# Local builder (inside this variant folder)
from .models import build_conditional_pixelcnn


# =============================================================================
# Small utils
# =============================================================================
def _cfg_get(cfg: Mapping[str, Any], dotted: str, default=None):
    """Dotted key getter: _cfg_get(cfg, "paths.artifacts", "artifacts")."""
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_slug(x: str) -> str:
    """Filesystem_safe slug for run ids / tags."""
    return (
        str(x)
        .strip()
        .replace("\\", "/")
        .replace(" ", "_")
        .replace(":", "_")
    )


def _to_uint8(img01: np.ndarray) -> np.ndarray:
    """[0,1] float -> uint8 for PNG writing."""
    return np.clip(np.rint(img01 * 255.0), 0, 255).astype(np.uint8)


def _save_png(img01: np.ndarray, out_path: Path) -> None:
    """Write a single image as PNG (grayscale or RGB)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(img01, dtype=np.float32)

    # Be strict about range; sampling should already be [0,1], but clamp for safety.
    x = np.clip(x, 0.0, 1.0)

    if x.ndim == 3 and x.shape[-1] == 1:
        Image.fromarray(_to_uint8(x[..., 0]), mode="L").save(out_path)
    elif x.ndim == 3 and x.shape[-1] == 3:
        Image.fromarray(_to_uint8(x), mode="RGB").save(out_path)
    else:
        Image.fromarray(_to_uint8(x.squeeze()), mode="L").save(out_path)


# =============================================================================
# Rule A: checkpoint directory resolution
# =============================================================================
def _resolve_rule_a_run_dirs(cfg: Dict) -> Optional[Dict[str, Path]]:
    """
    Resolve Rule_A run directories if run context is present.

    Accepted keys (any of these patterns):
      dataset.id          or data.dataset_id or DATASET_ID
      model.tag           or model_tag
      run.id              or run_id or run_meta.run_id

    Returns None if required fields are missing.
    """
    dataset_id = (
        _cfg_get(cfg, "dataset.id", None)
        or _cfg_get(cfg, "data.dataset_id", None)
        or cfg.get("DATASET_ID", None)
    )
    model_tag = _cfg_get(cfg, "model.tag", None) or cfg.get("model_tag", None)
    run_id = (
        _cfg_get(cfg, "run.id", None)
        or _cfg_get(cfg, "run_meta.run_id", None)
        or cfg.get("run_id", None)
    )

    if not (dataset_id and model_tag and run_id):
        return None

    runs_root = Path(_cfg_get(cfg, "paths.runs", _cfg_get(cfg, "paths.artifacts_runs", "artifacts/runs")))
    run_root = runs_root / _safe_slug(str(dataset_id)) / _safe_slug(str(model_tag)) / _safe_slug(str(run_id))
    return {
        "root": run_root,
        "checkpoints": run_root / "checkpoints",
        "summaries": run_root / "summaries",
        "tensorboard": run_root / "tensorboard",
        "synthetic": run_root / "synthetic",
    }


def _resolve_checkpoint_dir(cfg: Dict) -> Path:
    """
    Prefer Rule_A checkpoints if available; else fall back to legacy locations.
    """
    ra = _resolve_rule_a_run_dirs(cfg)
    if ra is not None:
        return ra["checkpoints"]

    artifacts_root = Path(_cfg_get(cfg, "paths.artifacts", "artifacts"))
    legacy_root = artifacts_root / "autoregressive"
    return Path(_cfg_get(cfg, "ARTIFACTS.autoregressive_checkpoints", legacy_root / "checkpoints"))


# =============================================================================
# Checkpoint loading
# =============================================================================
def load_ar_from_checkpoints(
    ckpt_dir: Path | str,
    *,
    img_shape: Tuple[int, int, int],
    num_classes: int,
) -> tf.keras.Model:
    """
    Build model and try to load AR_best→AR_last.
    If no checkpoint exists (or load fails), proceed with random weights.
    """
    ckpt_dir = Path(ckpt_dir)
    H, W, C = img_shape

    model = build_conditional_pixelcnn(img_shape, num_classes)

    # Build variables (Keras 3 friendly) by doing a single forward pass.
    dummy_x = tf.zeros((1, H, W, C), dtype=tf.float32)
    dummy_y = tf.one_hot([0], depth=num_classes, dtype=tf.float32)
    _ = model([dummy_x, dummy_y], training=False)

    best = ckpt_dir / "AR_best.weights.h5"
    last = ckpt_dir / "AR_last.weights.h5"
    to_load = best if best.exists() else last

    if not to_load.exists():
        print(f"[ckpt][warn] no AR checkpoint in {ckpt_dir}; using random weights.")
        return model

    try:
        model.load_weights(str(to_load))
        print(f"[ckpt] loaded {to_load.name}")
    except Exception as e:
        print(f"[ckpt][warn] failed to load {to_load.name}: {e}\n→ continuing with random weights.")
    return model


# =============================================================================
# Sampling (raster scan)
# =============================================================================
def _sample_autoregressive(
    model: tf.keras.Model,
    *,
    class_ids: np.ndarray,
    img_shape: Tuple[int, int, int],
    num_classes: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Raster_scan sampling; vectorized over batch, sequential over pixels.

    Returns images in [0,1], float32, shape (B,H,W,C).
    For binary_ish pixels, we sample Bernoulli(u < p).
    """
    if seed is not None:
        np.random.seed(int(seed))

    H, W, C = img_shape
    B = int(class_ids.shape[0])

    # Start with an empty canvas.
    imgs = np.zeros((B, H, W, C), dtype=np.float32)

    # One_hot conditioning.
    onehot = tf.keras.utils.to_categorical(
        class_ids.astype(int), num_classes=num_classes
    ).astype("float32")

    # Sequentially fill pixels.
    for i in range(H):
        for j in range(W):
            # Model outputs per_pixel probs in [0,1]
            probs = model.predict([imgs, onehot], verbose=0)  # (B,H,W,C)
            pij = probs[:, i, j, :]                          # (B,C)

            # Bernoulli sampling per channel
            u = np.random.rand(B, C).astype(np.float32)
            imgs[:, i, j, :] = (u < pij).astype(np.float32)

    return imgs


# =============================================================================
# Preview grid
# =============================================================================
def save_grid_from_ar(
    model: tf.keras.Model,
    num_classes: int,
    img_shape: Tuple[int, int, int],
    n_per_class: int = 1,
    path: Path | str = "artifacts/autoregressive/summaries/preview.png",
    seed: Optional[int] = 42,
) -> Path:
    """
    Render a small grid for previews:
      rows = num_classes
      cols = n_per_class
    """
    H, W, C = img_shape
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    class_ids = np.repeat(np.arange(num_classes, dtype=np.int32), n_per_class)
    imgs = _sample_autoregressive(
        model,
        class_ids=class_ids,
        img_shape=img_shape,
        num_classes=num_classes,
        seed=seed,
    )
    imgs = np.clip(imgs, 0.0, 1.0).astype("float32")

    fig_h = max(2.0, num_classes * 1.2)
    fig_w = max(2.0, n_per_class * 1.2)
    fig, axes = plt.subplots(num_classes, n_per_class, figsize=(fig_w, fig_h))

    if num_classes == 1 and n_per_class == 1:
        axes = np.array([[axes]])
    elif num_classes == 1:
        axes = axes.reshape(1, -1)
    elif n_per_class == 1:
        axes = axes.reshape(-1, 1)

    idx = 0
    for r in range(num_classes):
        for c in range(n_per_class):
            ax = axes[r, c]
            img = imgs[idx]
            idx += 1
            if C == 1:
                ax.imshow(img[:, :, 0], cmap="gray", vmin=0.0, vmax=1.0)
            else:
                ax.imshow(img, vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"C{r}", rotation=0, labelpad=10, fontsize=9, va="center")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# =============================================================================
# Unified CLI: synth()
# =============================================================================
def synth(cfg: dict, output_root: str, seed: int = 42) -> Dict:
    """
    Unified_CLI entrypoint.

    Writes:
      {output_root}/{class}/{seed}/ar_00000.png ...

    Reads checkpoints from:
      - Rule_A run checkpoints if dataset/model/run context provided
      - else legacy <paths.artifacts>/autoregressive/checkpoints

    Returns a manifest dict compatible with the orchestrator.
    """
    # ---------------- Resolve shape/classes/sample counts ----------------
    H, W, C = tuple(_cfg_get(cfg, "IMG_SHAPE", _cfg_get(cfg, "img.shape", (40, 40, 1))))
    K = int(_cfg_get(cfg, "NUM_CLASSES", _cfg_get(cfg, "data.num_classes", _cfg_get(cfg, "num_classes", 9))))
    S = int(_cfg_get(cfg, "SAMPLES_PER_CLASS", _cfg_get(cfg, "samples_per_class", 25)))

    # ---------------- Seeding ----------------
    np.random.seed(int(seed))
    tf.keras.utils.set_random_seed(int(seed))

    # ---------------- Locate checkpoints (Rule A preferred) ----------------
    ckpt_dir = _resolve_checkpoint_dir(cfg)
    ckpt_dir = Path(ckpt_dir)
    _ensure_dir(ckpt_dir)  # harmless if already exists

    # ---------------- Build + load model ----------------
    model = load_ar_from_checkpoints(ckpt_dir, img_shape=(H, W, C), num_classes=K)

    # ---------------- Output layout (orchestrator provides output_root) ----------------
    out_root = Path(output_root)
    per_class_counts: Dict[str, int] = {str(k): 0 for k in range(K)}
    paths: List[Dict] = []

    # Generate per class (keeps paths deterministic for manifests)
    for k in range(K):
        class_ids = np.full((S,), k, dtype=np.int32)
        imgs = _sample_autoregressive(
            model,
            class_ids=class_ids,
            img_shape=(H, W, C),
            num_classes=K,
            seed=seed,
        )
        imgs = np.clip(imgs, 0.0, 1.0).astype("float32")

        cls_dir = out_root / str(k) / str(seed)
        cls_dir.mkdir(parents=True, exist_ok=True)

        for j in range(S):
            p = cls_dir / f"ar_{j:05d}.png"
            _save_png(imgs[j], p)
            paths.append({"path": str(p), "label": int(k)})

        per_class_counts[str(k)] = int(S)

    # ---------------- Manifest ----------------
    dataset_id = (
        _cfg_get(cfg, "dataset.id", None)
        or _cfg_get(cfg, "data.dataset_id", None)
        or _cfg_get(cfg, "data.root", None)
        or _cfg_get(cfg, "DATA_DIR", "data")
    )
    model_tag = _cfg_get(cfg, "model.tag", None) or cfg.get("model_tag", "autoregressive/c_pixelcnnpp")
    run_id = _cfg_get(cfg, "run.id", None) or cfg.get("run_id", None)

    manifest = {
        "dataset": str(dataset_id),
        "seed": int(seed),
        "per_class_counts": per_class_counts,
        "paths": paths,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_tag": str(model_tag),
    }
    if run_id is not None:
        manifest["run_id"] = str(run_id)

    return manifest


__all__ = [
    "load_ar_from_checkpoints",
    "save_grid_from_ar",
    "synth",
]