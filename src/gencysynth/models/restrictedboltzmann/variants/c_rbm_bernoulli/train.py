# src/gencysynth/models/restrictedboltzmann/variants/c_rbm_bernoulli/train.py
"""
Bernoulli–Bernoulli Restricted Boltzmann Machine (RBM) training (per_class) for GenCyberSynth.

Rule A (Artifact + Dataset Scoping)
-----------------------------------
This variant MUST read/write artifacts in a dataset_scoped, variant_scoped layout:

  artifacts/<dataset_id>/restrictedboltzmann/c_rbm_bernoulli/
    checkpoints/
      class_<k>/
        RBM_best.weights.h5
        RBM_last.weights.h5
        RBM_epoch_XXXX.weights.h5
    summaries/
      rbm_train_preview.png (optional)

Where <dataset_id> is resolved from config in this priority order:
  1) cfg['data']['id']
  2) cfg['dataset']['id']
  3) basename(cfg['data']['root'] or cfg['DATA_DIR'])
  4) "default_dataset"

You may override specific artifact dirs via:
  cfg['artifacts']['restrictedboltzmann']['c_rbm_bernoulli']['checkpoints' | 'summaries']

What this module provides
-------------------------
- build_visible_dataset(...)  → tf.data over flattened visible vectors (B, V) in {0,1}
- cd_k_update(...)            → one Contrastive Divergence (CD_k) update
- train_rbm(...)              → per_class training loop with early stopping + checkpoints
- main(argv=None)             → argv_style entrypoint (works with app.main)
- train(cfg_or_argv)          → dict/argv adapter (works with app.main)

Notes
-----
- Inputs are expected in [0,1], shaped (N,H,W,C) or flat (N,V). We binarize at 0.5.
- This RBM is trained *per class* (K independent RBMs) to match the evaluator’s
  class_conditional synthetic generation contract used across model families.
- If a class has too few samples, we emit a small stub marker so downstream
  synthesis orchestration can remain robust.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Any, List

import numpy as np
import tensorflow as tf
import yaml

# Shared loader (preferred). Falls back to raw .npy if unavailable.
try:
    from common.data import load_dataset_npy  # type: ignore
except Exception:
    load_dataset_npy = None  # noqa: N816

# Local model
from .models import BernoulliRBM


# ----------------------------- GPU niceness ----------------------------- #
# Harmless on CPU nodes; avoids TF grabbing all GPU memory at once.
for d in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(d, True)
    except Exception:
        pass


# ===========================
# Small helpers
# ===========================
def _ensure_dir(p: Path) -> None:
    """Create directory (parents included) if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)


def _as_float(x) -> float:
    """Convert scalars / 0_D tensors / arrays to float."""
    try:
        return float(x)
    except Exception:
        return float(np.asarray(x).reshape(-1)[0])


def _cfg_get(cfg: Dict[str, Any], dotted: str, default=None):
    """Read nested dict values using dot notation; returns default if missing."""
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _coerce_cfg(cfg_or_argv) -> Dict[str, Any]:
    """
    Accept either:
      - a Python dict (already_parsed unified config), OR
      - an argv list/tuple like ['--config', 'configs/exp.yaml'].

    Returns a config dict.
    """
    if isinstance(cfg_or_argv, dict):
        return dict(cfg_or_argv)

    if isinstance(cfg_or_argv, (list, tuple)):
        import argparse

        p = argparse.ArgumentParser(description="RBM trainer (c_rbm_bernoulli)")
        p.add_argument("--config", default="configs/config.yaml")
        args = p.parse_args(list(cfg_or_argv))

        with open(args.config, "r", encoding="utf_8") as f:
            return yaml.safe_load(f) or {}

    raise TypeError(f"Unsupported config payload type: {type(cfg_or_argv)}")


# ===========================
# Rule A: dataset_scoped artifact paths
# ===========================
@dataclass(frozen=True)
class ArtifactPaths:
    """
    Dataset_scoped, variant_scoped artifact directories for this RBM variant.

    Default layout:
      artifacts/<dataset_id>/restrictedboltzmann/c_rbm_bernoulli/{checkpoints,summaries}
    """
    root: Path
    checkpoints: Path
    summaries: Path


def _resolve_dataset_id(cfg: Dict[str, Any]) -> str:
    """
    Resolve a stable dataset identifier used to scope artifacts.

    Priority:
      1) cfg.data.id
      2) cfg.dataset.id
      3) basename(cfg.data.root or cfg.DATA_DIR)
      4) "default_dataset"
    """
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
      cfg.artifacts.restrictedboltzmann.c_rbm_bernoulli.{checkpoints,summaries}
    """
    artifacts_root = Path(_cfg_get(cfg, "paths.artifacts", cfg.get("paths.artifacts", "artifacts")))
    dataset_id = _resolve_dataset_id(cfg)

    base = artifacts_root / dataset_id / "restrictedboltzmann" / "c_rbm_bernoulli"

    ckpt_override = _cfg_get(cfg, "artifacts.restrictedboltzmann.c_rbm_bernoulli.checkpoints", None)
    sum_override = _cfg_get(cfg, "artifacts.restrictedboltzmann.c_rbm_bernoulli.summaries", None)

    ckpt_dir = Path(ckpt_override) if ckpt_override else (base / "checkpoints")
    sums_dir = Path(sum_override) if sum_override else (base / "summaries")

    return ArtifactPaths(root=base, checkpoints=ckpt_dir, summaries=sums_dir)


# ===========================
# Dataset loading
# ===========================
def _load_dataset(cfg: Dict[str, Any], img_shape: Tuple[int, int, int], num_classes: int):
    """
    Load (x_train, y_train, x_val, y_val, x_test, y_test).

    Preferred path: common.data.load_dataset_npy
    Fallback path: raw .npy files under data.root / DATA_DIR.
    """
    data_dir = Path(cfg.get("DATA_DIR", _cfg_get(cfg, "data.root", "data"))).expanduser()
    val_frac = float(cfg.get("VAL_FRACTION", _cfg_get(cfg, "data.val_fraction", 0.5)))

    if load_dataset_npy is not None:
        return load_dataset_npy(data_dir, img_shape, num_classes, val_fraction=val_frac)

    # ---- Fallback: expect 4 files under data_dir ----
    xtr = np.load(data_dir / "train_data.npy").astype("float32")
    ytr = np.load(data_dir / "train_labels.npy")
    xte = np.load(data_dir / "test_data.npy").astype("float32")
    yte = np.load(data_dir / "test_labels.npy")

    # Normalize to [0,1] if byte_like
    if float(np.nanmax(xtr)) > 1.5:
        xtr /= 255.0
        xte /= 255.0

    H, W, C = img_shape
    xtr = xtr.reshape((-1, H, W, C))
    xte = xte.reshape((-1, H, W, C))

    # Split test into val + test (best_effort)
    n_val = int(len(xte) * val_frac)
    xva, yva = xte[:n_val], yte[:n_val]
    xte, yte = xte[n_val:], yte[n_val:]
    return xtr, ytr, xva, yva, xte, yte


def _int_labels(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Coerce one_hot labels → int labels; pass through if already int."""
    if y.ndim == 2 and y.shape[1] == num_classes:
        return np.argmax(y, axis=1).astype(np.int32)
    return y.astype(np.int32)


# ===========================
# Data pipeline (RBM visibles)
# ===========================
def build_visible_dataset(
    x: np.ndarray,
    *,
    img_shape: Tuple[int, int, int],
    batch_size: int,
    shuffle: bool = True,
    binarize: bool = True,
    threshold: float = 0.5,
) -> tf.data.Dataset:
    """
    Build a tf.data pipeline yielding flattened visible vectors.

    Parameters
    ----------
    x : np.ndarray
        Images in shape (N,H,W,C) or flattened (N,V), float in [0,1] (or byte_like).
    binarize : bool
        If True, threshold at `threshold` to produce {0,1} visibles (Bernoulli RBM).

    Returns
    -------
    tf.data.Dataset yielding (B,V) float32 tensors (either {0,1} or [0,1]).
    """
    H, W, C = img_shape
    V = H * W * C

    x = np.asarray(x)
    x = x.astype("float32", copy=False)

    # Normalize byte_like arrays to [0,1]
    if float(np.nanmax(x)) > 1.5:
        x = x / 255.0

    # Flatten to (N,V)
    if x.ndim == 4:
        x = x.reshape((-1, V))
    elif x.ndim != 2:
        raise ValueError(f"Expected (N,H,W,C) or (N,V); got shape {x.shape}")

    # Binarize if requested (Bernoulli visible units)
    if binarize:
        x = (x > float(threshold)).astype("float32", copy=False)

    ds = tf.data.Dataset.from_tensor_slices(x)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(x), 8192), reshuffle_each_iteration=True)
    return ds.batch(int(batch_size)).prefetch(tf.data.AUTOTUNE)


# ===========================
# Contrastive Divergence (CD_k)
# ===========================
@tf.function(reduce_retracing=True)
def cd_k_update(
    W: tf.Variable,
    v_bias: tf.Variable,
    h_bias: tf.Variable,
    v0: tf.Tensor,
    *,
    k: int = 1,
    lr: float = 1e_3,
    weight_decay: float = 0.0,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    One CD_k parameter update step.

    We use probabilities for expectations (less noisy) but still sample binary
    states for the Gibbs chain.

    Returns
    -------
    vk_prob : tf.Tensor
        Reconstructed visible probabilities (B,V).
    mse : tf.Tensor
        Mean squared reconstruction error (scalar).
    """
    # Positive phase: p(h|v0)
    h0_prob = tf.nn.sigmoid(tf.matmul(v0, W) + h_bias)  # (B,H)

    # Start Gibbs chain by sampling h ~ Bernoulli(h0_prob)
    h = tf.cast(tf.random.uniform(tf.shape(h0_prob)) < h0_prob, tf.float32)

    # Negative phase: k steps
    for _ in range(int(k)):
        v_prob = tf.nn.sigmoid(tf.matmul(h, tf.transpose(W)) + v_bias)  # (B,V)
        v = tf.cast(tf.random.uniform(tf.shape(v_prob)) < v_prob, tf.float32)
        h_prob = tf.nn.sigmoid(tf.matmul(v, W) + h_bias)                # (B,H)
        h = tf.cast(tf.random.uniform(tf.shape(h_prob)) < h_prob, tf.float32)

    vk_prob = v_prob
    hk_prob = h_prob

    B = tf.cast(tf.shape(v0)[0], tf.float32)

    # Gradients of log_likelihood approximation
    dW = (tf.matmul(tf.transpose(v0), h0_prob) - tf.matmul(tf.transpose(vk_prob), hk_prob)) / B
    dvb = tf.reduce_mean(v0 - vk_prob, axis=0)
    dhb = tf.reduce_mean(h0_prob - hk_prob, axis=0)

    # Optional L2 weight decay (on W)
    if weight_decay > 0.0:
        dW -= float(weight_decay) * W

    W.assign_add(float(lr) * dW)
    v_bias.assign_add(float(lr) * dvb)
    h_bias.assign_add(float(lr) * dhb)

    mse = tf.reduce_mean(tf.square(v0 - vk_prob))
    return vk_prob, mse


# ===========================
# Training loop (per class)
# ===========================
@dataclass
class RBMTrainConfig:
    """Strongly_typed knobs for RBM training."""
    img_shape: Tuple[int, int, int] = (40, 40, 1)
    epochs: int = 50
    batch_size: int = 128
    hidden_dim: int = 256
    cd_k: int = 1
    lr: float = 1e_3
    weight_decay: float = 0.0
    patience: int = 10
    save_every: int = 10


def train_rbm(
    rbm: BernoulliRBM,
    x_train: np.ndarray,
    x_val: Optional[np.ndarray] = None,
    *,
    cfg: RBMTrainConfig,
    ckpt_dir: Path,
    log_cb: Optional[Callable[[int, float, Optional[float]], None]] = None,
) -> Dict[str, Any]:
    """
    Train a single RBM on one class.

    Checkpoints written into ckpt_dir:
      - RBM_best.weights.h5
      - RBM_last.weights.h5
      - RBM_epoch_XXXX.weights.h5 (periodic)

    Returns a small summary dict suitable for orchestration logs.
    """
    H, W, C = cfg.img_shape
    V = H * W * C

    ckpt_dir = Path(ckpt_dir)
    _ensure_dir(ckpt_dir)

    best_path = ckpt_dir / "RBM_best.weights.h5"
    last_path = ckpt_dir / "RBM_last.weights.h5"

    # Build datasets (RBM expects flattened visible vectors)
    ds_tr = build_visible_dataset(
        x_train, img_shape=cfg.img_shape, batch_size=cfg.batch_size, shuffle=True, binarize=True
    )
    ds_va = (
        build_visible_dataset(x_val, img_shape=cfg.img_shape, batch_size=cfg.batch_size, shuffle=False, binarize=True)
        if x_val is not None
        else None
    )

    # Ensure model variables exist
    try:
        _ = rbm.W.shape
    except Exception:
        rbm(tf.zeros((1, V), dtype=tf.float32))

    best_val = float("inf")
    best_epoch = -1
    patience_ctr = 0

    for epoch in range(1, cfg.epochs + 1):
        # ---- Train epoch ----
        tr_losses: List[float] = []
        for v0 in ds_tr:
            v0 = tf.convert_to_tensor(v0, dtype=tf.float32)
            _, mse = cd_k_update(
                rbm.W,
                rbm.v_bias,
                rbm.h_bias,
                v0,
                k=cfg.cd_k,
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )
            tr_losses.append(_as_float(mse))

        tr = float(np.mean(tr_losses)) if tr_losses else float("nan")

        # ---- Validation ----
        va: Optional[float] = None
        if ds_va is not None:
            va_losses: List[float] = []
            for vv in ds_va:
                vv = tf.convert_to_tensor(vv, dtype=tf.float32)
                h_prob = tf.nn.sigmoid(tf.matmul(vv, rbm.W) + rbm.h_bias)
                v_prob = tf.nn.sigmoid(tf.matmul(h_prob, tf.transpose(rbm.W)) + rbm.v_bias)
                va_losses.append(_as_float(tf.reduce_mean(tf.square(vv - v_prob))))
            va = float(np.mean(va_losses)) if va_losses else float("nan")

        if log_cb is not None:
            log_cb(epoch, tr, va)

        # ---- Periodic snapshot ----
        if (epoch == 1) or (epoch % max(1, int(cfg.save_every)) == 0):
            rbm.save_weights(str(ckpt_dir / f"RBM_epoch_{epoch:04d}.weights.h5"))

        # ---- Early stopping ----
        monitor = va if va is not None else tr
        if np.isfinite(monitor) and (monitor < best_val - 1e_9):
            best_val = float(monitor)
            best_epoch = int(epoch)
            patience_ctr = 0
            rbm.save_weights(str(best_path))
        else:
            patience_ctr += 1
            if patience_ctr >= int(cfg.patience):
                rbm.save_weights(str(last_path))
                return {
                    "best_val": float(best_val),
                    "best_epoch": int(best_epoch),
                    "last_train": float(tr),
                    "stopped_early": True,
                    "ckpt_dir": str(ckpt_dir),
                }

    rbm.save_weights(str(last_path))
    return {
        "best_val": float(best_val if np.isfinite(best_val) else tr),
        "best_epoch": int(best_epoch if best_epoch > 0 else cfg.epochs),
        "last_train": float(tr),
        "stopped_early": False,
        "ckpt_dir": str(ckpt_dir),
    }


# ===========================
# High_level runner (unified)
# ===========================
def _run_train(cfg: Dict[str, Any]) -> int:
    """
    Unified entrypoint:
      - Resolves shapes, dataset, artifacts (Rule A)
      - Loads data
      - Trains per_class RBMs into checkpoints/class_<k>/
      - Optionally writes a small preview image under summaries/
    """
    cfg = dict(cfg)

    # ---- Reproducibility ----
    seed = int(_cfg_get(cfg, "SEED", _cfg_get(cfg, "train.seed", 42)))
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # ---- Shapes / classes ----
    img_shape = tuple(_cfg_get(cfg, "IMG_SHAPE", _cfg_get(cfg, "img.shape", (40, 40, 1))))
    H, W, C = (int(img_shape[0]), int(img_shape[1]), int(img_shape[2]))
    K = int(_cfg_get(cfg, "NUM_CLASSES", _cfg_get(cfg, "num_classes", 9)))
    V = H * W * C

    # ---- Hyperparameters (RBM_specific keys with legacy fallbacks) ----
    hidden_dim = int(_cfg_get(cfg, "rbm.hidden_dim", _cfg_get(cfg, "RBM_HIDDEN", 256)))
    epochs = int(_cfg_get(cfg, "rbm.epochs", _cfg_get(cfg, "RBM_EPOCHS", _cfg_get(cfg, "EPOCHS", 50))))
    batch_size = int(_cfg_get(cfg, "rbm.batch_size", _cfg_get(cfg, "RBM_BATCH", _cfg_get(cfg, "BATCH_SIZE", 128))))
    cd_k = int(_cfg_get(cfg, "rbm.cd_k", _cfg_get(cfg, "CD_K", 1)))
    lr = float(_cfg_get(cfg, "rbm.lr", _cfg_get(cfg, "RBM_LR", _cfg_get(cfg, "LR", 1e_3))))
    weight_decay = float(_cfg_get(cfg, "rbm.weight_decay", _cfg_get(cfg, "WEIGHT_DECAY", 0.0)))
    patience = int(_cfg_get(cfg, "rbm.patience", _cfg_get(cfg, "PATIENCE", 10)))
    save_every = int(_cfg_get(cfg, "rbm.save_every", _cfg_get(cfg, "SAVE_EVERY", 10)))

    # ---- Rule A artifact directories ----
    paths = _resolve_artifact_paths(cfg)
    ckpt_root = paths.checkpoints
    sums_dir = paths.summaries
    _ensure_dir(ckpt_root)
    _ensure_dir(sums_dir)

    # ---- Load dataset ----
    x_tr, y_tr, x_va, y_va, _x_te, _y_te = _load_dataset(cfg, (H, W, C), K)
    y_tr_i = _int_labels(y_tr, K)
    y_va_i = _int_labels(y_va, K) if y_va is not None else None

    print(
        f"[rbm][ruleA] dataset_id={_resolve_dataset_id(cfg)} "
        f"img_shape={(H, W, C)} K={K} V={V} hidden={hidden_dim} "
        f"epochs={epochs} batch={batch_size} cd_k={cd_k} lr={lr} seed={seed}"
    )
    print(f"[rbm][paths] ckpts={ckpt_root.resolve()} | summaries={sums_dir.resolve()}")

    # ---- Train per class ----
    for k in range(K):
        class_dir = ckpt_root / f"class_{k}"
        _ensure_dir(class_dir)

        idx_tr = (y_tr_i == k)
        n_k = int(idx_tr.sum())

        # Optional class_specific val split
        x_val_k = None
        if y_va_i is not None and x_va is not None:
            idx_va = (y_va_i == k)
            if int(idx_va.sum()) > 0:
                x_val_k = x_va[idx_va]

        # If the class is too small, write a stub marker (keeps synth robust)
        if n_k < 2:
            (class_dir / "RBM_best.weights.h5").touch()
            (class_dir / "RBM_last.weights.h5").touch()
            (class_dir / f"RBM_STUB_CLASS_{k}.txt").write_text(
                f"stub: too few training samples (n={n_k})\n", encoding="utf_8"
            )
            print(f"[rbm] class {k}: too few samples (n={n_k}); wrote stub markers → {class_dir}")
            continue

        rbm = BernoulliRBM(visible_dim=V, hidden_dim=hidden_dim)

        tcfg = RBMTrainConfig(
            img_shape=(H, W, C),
            epochs=epochs,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            cd_k=cd_k,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            save_every=save_every,
        )

        def _log(ep: int, tr: float, va: Optional[float]):
            # Keep logs concise on HPC: print first + periodic + last
            if (ep == 1) or (ep % max(1, save_every) == 0) or (ep == epochs):
                msg = f"[rbm][k={k}] epoch={ep:04d} train_mse={tr:.6f}"
                if va is not None:
                    msg += f" val_mse={va:.6f}"
                print(msg)

        print(f"[rbm] training class {k} | n_train={n_k} | ckpt_dir={class_dir}")
        _ = train_rbm(
            rbm,
            x_train=x_tr[idx_tr],
            x_val=x_val_k,
            cfg=tcfg,
            ckpt_dir=class_dir,
            log_cb=_log,
        )

    # ---- Optional preview grid (only if a compatible helper exists) ----
    try:
        # If you have a variant_local preview helper, keep it optional.
        # This import will fail harmlessly if not implemented.
        from .sample import save_grid_from_checkpoints  # type: ignore

        out = sums_dir / "rbm_train_preview.png"
        save_grid_from_checkpoints(
            ckpt_root=ckpt_root,
            img_shape=(H, W, C),
            num_classes=K,
            path=out,
            per_class=1,
        )
        print(f"[rbm] preview grid → {out}")
    except Exception as e:
        print(f"[rbm][warn] preview grid skipped: {e}")

    # Marker file for orchestrators
    (sums_dir / "train_done.txt").write_text("ok\n", encoding="utf_8")
    return 0


# ===========================
# Public entrypoints
# ===========================
def main(argv=None) -> int:
    """
    argv_style entrypoint.

    Examples
    --------
    - python -m ...train --config configs/exp.yaml
    - app.main calling: main(['--config','...'])
    - app.main calling: main(cfg_dict)
    """
    if isinstance(argv, dict):
        return _run_train(argv)

    if argv is None:
        import argparse

        p = argparse.ArgumentParser(description="Train RBM (c_rbm_bernoulli)")
        p.add_argument("--config", default="configs/config.yaml")
        args = p.parse_args()

        with open(args.config, "r", encoding="utf_8") as f:
            cfg = yaml.safe_load(f) or {}
        return _run_train(cfg)

    cfg = _coerce_cfg(argv)
    return _run_train(cfg)


def train(cfg_or_argv) -> int:
    """
    Dict/argv adapter so orchestrators can call:
      - train(config_dict)
      - train(['--config','configs/exp.yaml'])

    Returns 0 on success.
    """
    cfg = _coerce_cfg(cfg_or_argv)
    return _run_train(cfg)


__all__ = [
    "RBMTrainConfig",
    "ArtifactPaths",
    "build_visible_dataset",
    "cd_k_update",
    "train_rbm",
    "main",
    "train",
]

if __name__ == "__main__":
    raise SystemExit(main())