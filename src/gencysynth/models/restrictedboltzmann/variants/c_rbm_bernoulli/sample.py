# src/gencysynth/models/restrictedboltzmann/variants/c_rbm_bernoulli/sample.py
"""
Sampling utilities for the Bernoulli–Bernoulli RBM (c_rbm_bernoulli).

Rule A (Artifact + Dataset Scoping)
-----------------------------------
This variant MUST read/write artifacts in a dataset_scoped, variant_scoped layout:

  artifacts/<dataset_id>/restrictedboltzmann/c_rbm_bernoulli/
    checkpoints/
      class_<k>/RBM_{best,last,epoch_XXXX}.weights.h5
    synthetic/
      gen_class_<k>.npy
      labels_class_<k>.npy
      x_synth.npy
      y_synth.npy
    summaries/
      rbm_train_preview.png   (optional)
      rbm_synth_preview.png   (optional)

Where <dataset_id> is resolved from config in this priority order:
  1) cfg['data']['id']
  2) cfg['dataset']['id']
  3) basename(cfg['data']['root'] or cfg['DATA_DIR'])
  4) "default_dataset"

You may override specific artifact dirs via:
  cfg['artifacts']['restrictedboltzmann']['c_rbm_bernoulli']['checkpoints'|'synthetic'|'summaries']

Public API
----------
- sample_gibbs(...)                  -> numpy float32 samples ({0,1} or probs) with optional image reshape
- save_grid_from_checkpoints(...)    -> compact preview PNG grid (optional helper)
- synthesize_to_artifacts(cfg, ...)  -> writes evaluator_friendly .npy files (Rule A)
- synth(cfg, output_root, seed=...)  -> "PNG synth" for legacy paths (kept for compatibility)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import tensorflow as tf
from PIL import Image

from .models import BernoulliRBM


# =============================================================================
# Rule A: dataset_scoped artifact paths
# =============================================================================
@dataclass(frozen=True)
class ArtifactPaths:
    """
    Dataset_scoped, variant_scoped artifact directories for this RBM variant.

    Default layout:
      artifacts/<dataset_id>/restrictedboltzmann/c_rbm_bernoulli/{checkpoints,synthetic,summaries}
    """
    root: Path
    checkpoints: Path
    synthetic: Path
    summaries: Path


def _cfg_get(cfg: Dict[str, Any], dotted: str, default=None):
    """Read nested dict values using dot notation; returns default if missing."""
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _resolve_dataset_id(cfg: Dict[str, Any]) -> str:
    """Resolve a stable dataset identifier used to scope artifacts."""
    ds_id = _cfg_get(cfg, "data.id", None) or _cfg_get(cfg, "dataset.id", None)
    if ds_id:
        return str(ds_id)

    data_root = _cfg_get(cfg, "data.root", None) or cfg.get("DATA_DIR", None)
    if data_root:
        return Path(str(data_root)).expanduser().resolve().name

    return "default_dataset"


def _resolve_artifact_paths(cfg: Dict[str, Any]) -> ArtifactPaths:
    """
    Resolve artifact paths under Rule A. Supports overrides under:
      cfg.artifacts.restrictedboltzmann.c_rbm_bernoulli.{checkpoints,synthetic,summaries}
    """
    artifacts_root = Path(_cfg_get(cfg, "paths.artifacts", cfg.get("paths.artifacts", "artifacts")))
    dataset_id = _resolve_dataset_id(cfg)

    base = artifacts_root / dataset_id / "restrictedboltzmann" / "c_rbm_bernoulli"

    ckpt_override = _cfg_get(cfg, "artifacts.restrictedboltzmann.c_rbm_bernoulli.checkpoints", None)
    syn_override = _cfg_get(cfg, "artifacts.restrictedboltzmann.c_rbm_bernoulli.synthetic", None)
    sum_override = _cfg_get(cfg, "artifacts.restrictedboltzmann.c_rbm_bernoulli.summaries", None)

    ckpt_dir = Path(ckpt_override) if ckpt_override else (base / "checkpoints")
    synth_dir = Path(syn_override) if syn_override else (base / "synthetic")
    sums_dir = Path(sum_override) if sum_override else (base / "summaries")

    return ArtifactPaths(root=base, checkpoints=ckpt_dir, synthetic=synth_dir, summaries=sums_dir)


def _ensure_dir(p: Path) -> None:
    """Create directory (parents included) if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Gibbs sampling (RBM core)
# =============================================================================
@tf.function(reduce_retracing=True)
def _gibbs_step(rbm: BernoulliRBM, v: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    One blocked Gibbs update: v -> h -> v.

    Returns
    -------
    v_sample : (B,V) float32 in {0,1}
    v_prob   : (B,V) float32 in [0,1]  (probabilities for the sampled step)
    """
    h_sample, _ = rbm.sample_h_given_v(v)                 # (B,H) {0,1}
    v_sample, v_prob = rbm.sample_v_given_h(h_sample)     # (B,V) {0,1}, [0,1]
    return tf.cast(v_sample, tf.float32), tf.cast(v_prob, tf.float32)


def _rand_bernoulli(shape, seed: Optional[int] = None) -> tf.Tensor:
    """Random Bernoulli(0.5) tensor in {0,1}."""
    rnd = tf.random.uniform(shape, dtype=tf.float32, seed=seed)
    return tf.cast(rnd > 0.5, tf.float32)


def sample_gibbs(
    rbm: BernoulliRBM,
    num_samples: int,
    k: int = 1,
    *,
    init: Optional[np.ndarray | tf.Tensor] = None,
    img_shape: Optional[Tuple[int, int, int]] = None,
    binarize_init: bool = False,
    burn_in: int = 0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Draw visible samples via blocked Gibbs sampling.

    Parameters
    ----------
    num_samples : int
        Number of parallel chains/samples to draw.
    k : int
        Number of Gibbs steps to run after burn_in.
    burn_in : int
        Number of Gibbs steps to run before sampling.
    init : optional
        Initial visible state (num_samples, V) or image_shaped (num_samples,H,W,C).
        If provided and `binarize_init` is False, it is interpreted as probs and
        sampled to {0,1} once at initialization.
    img_shape : optional
        If provided, reshape flat visibles to (N,H,W,C) before returning.

    Returns
    -------
    out : np.ndarray
        float32 array in {0,1}, shape (N,V) or (N,H,W,C) if img_shape is provided.
    """
    V = int(rbm.W.shape[0])

    if seed is not None:
        tf.random.set_seed(int(seed))

    # ---- Initialize v ----
    if init is None:
        v = _rand_bernoulli((int(num_samples), V), seed=seed)
    else:
        v = tf.convert_to_tensor(init, dtype=tf.float32)
        v = tf.reshape(v, (int(num_samples), V))

        if binarize_init:
            v = tf.cast(v > 0.5, tf.float32)
        else:
            # Treat init as probabilities and sample once to get {0,1}
            v = tf.clip_by_value(v, 0.0, 1.0)
            rnd = tf.random.uniform(tf.shape(v), dtype=tf.float32, seed=seed)
            v = tf.cast(rnd < v, tf.float32)

    # ---- Burn_in ----
    for _ in range(int(burn_in)):
        v, _ = _gibbs_step(rbm, v)

    # ---- Sampling steps ----
    for _ in range(int(k)):
        v, _ = _gibbs_step(rbm, v)

    out = v.numpy().astype("float32", copy=False)
    if img_shape is not None:
        H, W, C = img_shape
        out = out.reshape((-1, int(H), int(W), int(C)))
    return out


# =============================================================================
# Checkpoint discovery/loading
# =============================================================================
def _find_ckpt_for_class(ckpt_root: Path, k: int) -> Optional[Path]:
    """
    Find a checkpoint for class k. Under Rule A we *prefer*:
      {ckpt_root}/class_{k}/RBM_best.weights.h5
      {ckpt_root}/class_{k}/RBM_last.weights.h5
      newest {ckpt_root}/class_{k}/RBM_epoch_*.weights.h5

    For robustness, we also attempt legacy "flat" names (older experiments).
    """
    per_class = ckpt_root / f"class_{k}"
    candidates: List[Path] = [
        per_class / "RBM_best.weights.h5",
        per_class / "RBM_last.weights.h5",
    ]

    if per_class.exists():
        epochs = sorted(per_class.glob("RBM_epoch_*.weights.h5"))
        if epochs:
            candidates.append(epochs[-1])

    # ---- Legacy fallbacks (kept to avoid breaking old runs) ----
    candidates += [
        ckpt_root / f"RBM_class_{k}.weights.h5",
        ckpt_root / f"RBM_class_{k}.h5",
    ]

    for p in candidates:
        # NOTE: stub markers may "touch" empty files; RBM load will fail on those.
        if p.exists() and p.is_file() and p.stat().st_size > 0:
            return p

    return None


def _load_rbm(ckpt_path: Path, visible_dim: int, hidden_dim: int) -> BernoulliRBM:
    """
    Instantiate an RBM and load weights from a Keras_compatible weights file.

    Raises
    ------
    Any exception raised by `load_weights` (caller should handle).
    """
    rbm = BernoulliRBM(visible_dim=int(visible_dim), hidden_dim=int(hidden_dim))
    rbm.load_weights(str(ckpt_path))
    return rbm


# =============================================================================
# Evaluator_friendly artifact writing (Rule A)
# =============================================================================
def _one_hot_from_ids(ids: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer labels -> one_hot float32."""
    ids = ids.astype(np.int32).ravel()
    out = np.zeros((len(ids), int(num_classes)), dtype=np.float32)
    out[np.arange(len(ids)), ids] = 1.0
    return out


def synthesize_to_artifacts(
    cfg: Dict[str, Any],
    *,
    model_hidden_dim: Optional[int] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rule A synthesis:
      - Loads per_class RBM checkpoints from dataset_scoped ckpt root
      - Samples S samples per class (Gibbs)
      - Writes evaluator_friendly .npy files into dataset_scoped synthetic dir:
          gen_class_<k>.npy, labels_class_<k>.npy, x_synth.npy, y_synth.npy

    Returns
    -------
    x_synth : float32 (N,H,W,C) in {0,1}
    y_synth : float32 (N,K) one_hot
    """
    paths = _resolve_artifact_paths(cfg)
    _ensure_dir(paths.synthetic)

    # Resolve shapes & counts (support a couple of key aliases)
    H, W, C = tuple(_cfg_get(cfg, "IMG_SHAPE", _cfg_get(cfg, "img.shape", (40, 40, 1))))
    K = int(_cfg_get(cfg, "NUM_CLASSES", _cfg_get(cfg, "num_classes", 9)))
    S = int(_cfg_get(cfg, "SAMPLES_PER_CLASS", _cfg_get(cfg, "samples_per_class", 25)))

    # RBM_specific sampling knobs
    gibbs_k = int(_cfg_get(cfg, "rbm.gibbs_k", _cfg_get(cfg, "RBM_GIBBS_K", _cfg_get(cfg, "CD_K", 1))))
    burn_in = int(_cfg_get(cfg, "rbm.burn_in", _cfg_get(cfg, "RBM_BURN_IN", 0)))

    # Hidden dim: prefer explicit arg, then cfg, then default
    hidden = int(
        model_hidden_dim
        if model_hidden_dim is not None
        else _cfg_get(cfg, "rbm.hidden_dim", _cfg_get(cfg, "RBM_HIDDEN", 256))
    )

    V = int(H) * int(W) * int(C)
    tf.random.set_seed(int(seed))

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    for k in range(int(K)):
        ckpt = _find_ckpt_for_class(paths.checkpoints, k)
        if ckpt is None:
            # Keep contract: if missing class, emit zeros (and mark)
            xk = np.zeros((int(S), int(H), int(W), int(C)), dtype=np.float32)
            labels = np.full((int(S),), int(k), dtype=np.int32)

            np.save(paths.synthetic / f"gen_class_{k}.npy", xk)
            np.save(paths.synthetic / f"labels_class_{k}.npy", labels)
            (paths.synthetic / f"MISSING_CKPT_CLASS_{k}.txt").write_text(
                f"missing checkpoint for class {k}\n", encoding="utf_8"
            )

            xs.append(xk)
            ys.append(_one_hot_from_ids(labels, int(K)))
            continue

        # Load RBM and sample
        rbm = _load_rbm(ckpt, visible_dim=V, hidden_dim=hidden)
        xk = sample_gibbs(
            rbm,
            num_samples=int(S),
            k=int(gibbs_k),
            img_shape=(int(H), int(W), int(C)),
            burn_in=int(burn_in),
            seed=int(seed) + int(k),
        )
        xk = xk.astype(np.float32, copy=False)

        labels = np.full((xk.shape[0],), int(k), dtype=np.int32)

        # Per_class dumps (evaluator contract used across families)
        np.save(paths.synthetic / f"gen_class_{k}.npy", xk)
        np.save(paths.synthetic / f"labels_class_{k}.npy", labels)

        xs.append(xk)
        ys.append(_one_hot_from_ids(labels, int(K)))

    x_synth = np.concatenate(xs, axis=0).astype(np.float32, copy=False)
    y_synth = np.concatenate(ys, axis=0).astype(np.float32, copy=False)

    # Convenience combined dumps
    np.save(paths.synthetic / "x_synth.npy", x_synth)
    np.save(paths.synthetic / "y_synth.npy", y_synth)

    print(f"[rbm][synthesize][ruleA] {x_synth.shape[0]} samples ({S} per class) -> {paths.synthetic}")
    return x_synth, y_synth


# =============================================================================
# PNG helpers (kept for legacy synth + previews)
# =============================================================================
def _to_uint8(img01: np.ndarray) -> np.ndarray:
    return np.clip(np.round(img01 * 255.0), 0, 255).astype(np.uint8)


def _save_png(img01: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
# Preview grid (optional)
# =============================================================================
def save_grid_from_checkpoints(
    *,
    ckpt_root: Path,
    img_shape: Tuple[int, int, int],
    num_classes: int,
    path: Path,
    per_class: int = 1,
    hidden_dim: int = 256,
    gibbs_k: int = 1,
    burn_in: int = 0,
    seed: int = 42,
) -> None:
    """
    Sample a small (per_class x num_classes) grid and save as PNG.

    This is meant for quick sanity checks, not for evaluation.
    """
    import matplotlib.pyplot as plt

    H, W, C = img_shape
    V = int(H) * int(W) * int(C)

    tiles: List[np.ndarray] = []
    for k in range(int(num_classes)):
        ckpt = _find_ckpt_for_class(Path(ckpt_root), int(k))
        if ckpt is None:
            tiles.append(np.zeros((int(per_class), int(H), int(W), int(C)), dtype=np.float32))
            continue

        rbm = _load_rbm(ckpt, visible_dim=V, hidden_dim=int(hidden_dim))
        xk = sample_gibbs(
            rbm,
            num_samples=int(per_class),
            k=int(gibbs_k),
            img_shape=img_shape,
            burn_in=int(burn_in),
            seed=int(seed) + int(k),
        )
        tiles.append(xk)

    x = np.stack(tiles, axis=0)  # (K, per_class, H, W, C)
    rows, cols = int(per_class), int(num_classes)

    fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 1.6 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, 0)

    for j in range(cols):
        for i in range(rows):
            ax = axes[i, j]
            img = x[j, i]
            if int(C) == 1:
                ax.imshow(img[:, :, 0], cmap="gray", vmin=0.0, vmax=1.0)
            else:
                ax.imshow(np.clip(img, 0.0, 1.0))
            ax.set_axis_off()
            if i == 0:
                ax.set_title(f"C{j}", fontsize=9)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# =============================================================================
# Legacy: PNG synth(cfg, output_root, seed)
# =============================================================================
def synth(cfg: dict, output_root: str, seed: int = 42) -> dict:
    """
    Legacy PNG synthesis entrypoint (kept for backward compatibility).

    It writes per_sample PNGs to:
      {output_root}/{class}/{seed}/rbm_00000.png ...

    Under Rule A, prefer `synthesize_to_artifacts(cfg, ...)` for evaluator use.
    """
    # Shapes & counts (support a couple of key aliases)
    H, W, C = tuple(_cfg_get(cfg, "IMG_SHAPE", _cfg_get(cfg, "img.shape", (40, 40, 1))))
    K = int(_cfg_get(cfg, "NUM_CLASSES", _cfg_get(cfg, "num_classes", 9)))
    S = int(_cfg_get(cfg, "SAMPLES_PER_CLASS", _cfg_get(cfg, "samples_per_class", 25)))

    # RBM knobs
    hidden = int(_cfg_get(cfg, "rbm.hidden_dim", _cfg_get(cfg, "RBM_HIDDEN", 256)))
    gibbs_k = int(_cfg_get(cfg, "rbm.gibbs_k", _cfg_get(cfg, "RBM_GIBBS_K", _cfg_get(cfg, "CD_K", 1))))
    burn_in = int(_cfg_get(cfg, "rbm.burn_in", _cfg_get(cfg, "RBM_BURN_IN", 0)))

    # Rule A checkpoint root
    pathsA = _resolve_artifact_paths(cfg)
    ckpt_root = pathsA.checkpoints

    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    V = int(H) * int(W) * int(C)
    per_class_counts: Dict[str, int] = {str(k): 0 for k in range(K)}
    paths: List[dict] = []

    tf.random.set_seed(int(seed))

    for k in range(int(K)):
        ckpt = _find_ckpt_for_class(ckpt_root, int(k))
        if ckpt is None:
            print(f"[synth][rbm] missing checkpoint for class {k}; skipping.")
            continue

        rbm = _load_rbm(ckpt, visible_dim=V, hidden_dim=int(hidden))
        xk = sample_gibbs(
            rbm,
            num_samples=int(S),
            k=int(gibbs_k),
            img_shape=(int(H), int(W), int(C)),
            burn_in=int(burn_in),
            seed=int(seed) + int(k),
        )

        cls_dir = out_root / str(k) / str(seed)
        cls_dir.mkdir(parents=True, exist_ok=True)

        for j in range(xk.shape[0]):
            out_path = cls_dir / f"rbm_{j:05d}.png"
            _save_png(xk[j], out_path)
            paths.append({"path": str(out_path), "label": int(k)})

        per_class_counts[str(k)] = int(xk.shape[0])

    manifest = {
        "dataset": _cfg_get(cfg, "data.root", _cfg_get(cfg, "DATA_DIR", "data")),
        "seed": int(seed),
        "per_class_counts": per_class_counts,
        "paths": paths,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    return manifest


__all__ = [
    "ArtifactPaths",
    "sample_gibbs",
    "_gibbs_step",
    "save_grid_from_checkpoints",
    "synthesize_to_artifacts",
    "synth",
]