# src/gencysynth/models/autoregressive/variants/c_pixelcnnpp/train.py

"""
GenCyberSynth — Autoregressive — c_pixelcnnpp — Training
=======================================================

This module trains a **conditional autoregressive** (PixelCNN++) model in a
small, explicit, framework_idiomatic way:

- tf.data pipelines for (image, one_hot label) pairs
- @tf.function train/val steps
- Early stopping on validation loss
- Optional TensorBoard logging
- Robust checkpointing using Keras 3_friendly names (*.weights.h5)

Rule A (Artifacts + scalability)
--------------------------------
All artifact I/O is keyed by: (dataset_id, model_tag, run_id)

Canonical layout (Rule A):
  artifacts/runs/<dataset_id>/<model_tag>/<run_id>/
    checkpoints/
      AR_best.weights.h5
      AR_last.weights.h5
      AR_epoch_XXXX.weights.h5
    summaries/
      train_done.txt
    tensorboard/   (optional)

Back_compat:
- If a Rule_A run context is NOT provided, we fall back to legacy locations:
    <paths.artifacts>/autoregressive/{checkpoints,summaries,tensorboard}
  or explicit ARTIFACTS.autoregressive_* keys if present.

Assumptions
-----------
- model([x, y_onehot]) -> probs with same shape as x, probs in [0,1]
- x_* inputs in [0,1] (float32)
- y_* labels are one_hot (N, K)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Mapping

import numpy as np
import tensorflow as tf

from common.data import load_dataset_npy, to_01, one_hot
from autoregressive.models import build_conditional_pixelcnn


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class TrainConfig:
    """Lightweight knobs for the training loop (defaults are safe)."""
    epochs: int = 200
    batch_size: int = 256
    patience: int = 10
    save_every: int = 25
    from_logits: bool = False  # PixelCNN++ typically outputs probs, not logits


# =============================================================================
# Small helpers
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
    """
    Convert arbitrary strings (e.g., model tags with slashes) into a filesystem_safe slug.
    We keep it readable but safe across POSIX filesystems.
    """
    return (
        str(x)
        .strip()
        .replace("\\", "/")
        .replace(" ", "_")
        .replace(":", "_")
    )


# =============================================================================
# Rule A: artifact path resolution
# =============================================================================
def _resolve_rule_a_dirs(cfg: Dict) -> Optional[Dict[str, Path]]:
    """
    If the config includes a Rule_A run context, return canonical dirs.

    Accepted keys (any one of these patterns is enough):
      dataset.id          or data.dataset_id
      model.tag           or model_tag
      run.id              or run_id or run_meta.run_id

    Returns None if required information is missing.
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

    # Root for all runs (Rule A)
    runs_root = Path(_cfg_get(cfg, "paths.runs", _cfg_get(cfg, "paths.artifacts_runs", "artifacts/runs")))
    dataset_slug = _safe_slug(str(dataset_id))
    model_slug = _safe_slug(str(model_tag))
    run_slug = _safe_slug(str(run_id))

    run_root = runs_root / dataset_slug / model_slug / run_slug
    return {
        "root": run_root,
        "checkpoints": run_root / "checkpoints",
        "summaries": run_root / "summaries",
        "tensorboard": run_root / "tensorboard",
    }


def _resolve_artifact_dirs(cfg: Dict) -> Dict[str, Path]:
    """
    Resolve artifact directories following Rule A when possible, else fall back
    to the legacy layout under `paths.artifacts` (or explicit ARTIFACTS keys).
    """
    # 1) Rule A (preferred)
    rule_a = _resolve_rule_a_dirs(cfg)
    if rule_a is not None:
        return {k: _ensure_dir(v) if k != "root" else v for k, v in rule_a.items()}

    # 2) Legacy fallback: <paths.artifacts>/autoregressive/...
    artifacts_root = Path(_cfg_get(cfg, "paths.artifacts", "artifacts"))
    legacy_root = artifacts_root / "autoregressive"

    ckpt_dir = Path(_cfg_get(cfg, "ARTIFACTS.autoregressive_checkpoints", legacy_root / "checkpoints"))
    sums_dir = Path(_cfg_get(cfg, "ARTIFACTS.autoregressive_summaries", legacy_root / "summaries"))
    tb_dir = Path(_cfg_get(cfg, "ARTIFACTS.autoregressive_tensorboard", legacy_root / "tensorboard"))

    return {
        "root": legacy_root,
        "checkpoints": _ensure_dir(ckpt_dir),
        "summaries": _ensure_dir(sums_dir),
        "tensorboard": _ensure_dir(tb_dir),
    }


# =============================================================================
# Data helpers
# =============================================================================
def make_datasets(
    x_train01: np.ndarray,
    y_train_1h: np.ndarray,
    x_val01: np.ndarray,
    y_val_1h: np.ndarray,
    batch_size: int,
    shuffle_buffer: int = 10240,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Build shuffling, batched tf.data pipelines for training and validation.
    Inputs are assumed already normalized to [0,1] and labels are one_hot.
    """
    def _ds(x, y, training=False):
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if training:
            ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = _ds(x_train01, y_train_1h, training=True)
    val_ds   = _ds(x_val01,   y_val_1h,   training=False)
    return train_ds, val_ds


def make_writer(tensorboard_dir: str | Path | None) -> Optional[tf.summary.SummaryWriter]:
    """
    Create a TensorBoard writer if a directory is provided.
    If None/empty, returns None (training runs without TB).
    """
    if not tensorboard_dir:
        return None
    tb_path = Path(tensorboard_dir)
    tb_path.mkdir(parents=True, exist_ok=True)
    return tf.summary.create_file_writer(str(tb_path))


# =============================================================================
# Core training loop
# =============================================================================
def fit_autoregressive(
    *,
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int,
    checkpoint_dir: Path,
    writer: Optional[tf.summary.SummaryWriter] = None,
    patience: int = 10,
    save_every: int = 25,
    from_logits: bool = False,
) -> None:
    """
    Train a conditional autoregressive model with early stopping & checkpoints.

    Checkpoints (Keras 3 style)
    ---------------------------
    - AR_best.weights.h5  (lowest validation loss)
    - AR_last.weights.h5  (last epoch finished)
    - AR_epoch_XXXX.weights.h5 (periodic snapshots)
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)
    train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
    val_loss_metric   = tf.keras.metrics.Mean(name="val_loss")

    @tf.function(reduce_retracing=True)
    def train_step(x, y1h):
        with tf.GradientTape() as tape:
            probs = model([x, y1h], training=True)
            loss = bce(x, probs)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss_metric.update_state(loss)

    @tf.function(reduce_retracing=True)
    def val_step(x, y1h):
        probs = model([x, y1h], training=False)
        loss = bce(x, probs)
        val_loss_metric.update_state(loss)

    best_val = float("inf")
    no_improve = 0

    # Optional resume (safe; if incompatible we just continue from scratch)
    last_ckpt = checkpoint_dir / "AR_last.weights.h5"
    if last_ckpt.exists():
        try:
            model.load_weights(str(last_ckpt))
            print(f"[resume] Loaded {last_ckpt.name}")
        except Exception:
            print("[resume] Found AR_last but failed to load; training from scratch.")

    for epoch in range(1, epochs + 1):
        # ---------------- Train ----------------
        train_loss_metric.reset_state()
        for xb, yb in train_ds:
            train_step(xb, yb)
        train_loss = float(train_loss_metric.result())

        # ---------------- Validate ----------------
        val_loss_metric.reset_state()
        for xb, yb in val_ds:
            val_step(xb, yb)
        val_loss = float(val_loss_metric.result())

        # ---------------- Log ----------------
        print(f"[epoch {epoch:05d}] train={train_loss:.4f} | val={val_loss:.4f}")
        if writer:
            with writer.as_default():
                tf.summary.scalar("loss/train_total", train_loss, step=epoch)
                tf.summary.scalar("loss/val_total", val_loss, step=epoch)
                writer.flush()

        # ---------------- Snapshot ----------------
        if (epoch == 1) or (epoch % max(1, save_every) == 0):
            snap = checkpoint_dir / f"AR_epoch_{epoch:04d}.weights.h5"
            model.save_weights(str(snap))

        # ---------------- Best + early stop ----------------
        if val_loss < best_val - 1e_6:
            best_val = val_loss
            no_improve = 0
            model.save_weights(str(checkpoint_dir / "AR_best.weights.h5"))
        else:
            no_improve += 1
            if no_improve >= max(1, patience):
                print(f"[early_stop] No val improvement for {patience} epochs.")
                break

    # Always write a final "last" checkpoint.
    model.save_weights(str(checkpoint_dir / "AR_last.weights.h5"))


# =============================================================================
# Unified CLI adapter plumbing
# =============================================================================
def _coerce_cfg(cfg_or_argv):
    """
    Accept either:
      - a config dict
      - an argv list/tuple like ['--config','configs/config.yaml']

    Returns a Python dict.
    """
    if isinstance(cfg_or_argv, dict):
        return dict(cfg_or_argv)

    if isinstance(cfg_or_argv, (list, tuple)):
        import yaml
        cfg_path = None
        if "--config" in cfg_or_argv:
            i = cfg_or_argv.index("--config")
            if i + 1 < len(cfg_or_argv):
                cfg_path = Path(cfg_or_argv[i + 1])
        if cfg_path is None:
            cfg_path = Path("configs/config.yaml")
        with open(cfg_path, "r", encoding="utf_8") as f:
            return yaml.safe_load(f) or {}

    raise TypeError(f"Unsupported config payload type: {type(cfg_or_argv)}")


def _train_from_cfg(cfg: Dict) -> None:
    """
    Orchestrator_facing training entrypoint.

    Minimal expected keys (new + legacy compatible):
      - IMG_SHAPE or img.shape
      - NUM_CLASSES or data.num_classes
      - DATA_DIR or data.root
      - EPOCHS / BATCH_SIZE / PATIENCE / SAVE_EVERY / LR / FROM_LOGITS / SEED
      - Rule A run context (preferred):
          dataset.id, model.tag, run.id
        else legacy:
          paths.artifacts and/or ARTIFACTS.autoregressive_*
    """
    # ---------------- Shapes & classes ----------------
    H, W, C = tuple(_cfg_get(cfg, "IMG_SHAPE", _cfg_get(cfg, "img.shape", (40, 40, 1))))
    K = int(_cfg_get(cfg, "NUM_CLASSES", _cfg_get(cfg, "data.num_classes", _cfg_get(cfg, "num_classes", 9))))
    img_shape = (int(H), int(W), int(C))

    # ---------------- Hyperparameters ----------------
    epochs      = int(_cfg_get(cfg, "EPOCHS", 200))
    batch_size  = int(_cfg_get(cfg, "BATCH_SIZE", 256))
    patience    = int(_cfg_get(cfg, "PATIENCE", 10))
    save_every  = int(_cfg_get(cfg, "SAVE_EVERY", 25))
    lr          = float(_cfg_get(cfg, "LR", 2e_4))
    from_logits = bool(_cfg_get(cfg, "FROM_LOGITS", False))
    seed        = int(_cfg_get(cfg, "SEED", _cfg_get(cfg, "run_meta.seed", 42)))

    tf.keras.utils.set_random_seed(seed)

    # ---------------- Artifact dirs (Rule A preferred) ----------------
    dirs = _resolve_artifact_dirs(cfg)
    ckpt_dir = dirs["checkpoints"]
    sums_dir = dirs["summaries"]
    tb_dir   = dirs["tensorboard"]

    # ---------------- Data root ----------------
    data_dir_str = _cfg_get(cfg, "DATA_DIR", _cfg_get(cfg, "data.root", None))
    if not data_dir_str:
        raise ValueError("DATA_DIR not set in config (or data.root).")
    data_dir = Path(str(data_dir_str)).expanduser()
    if not data_dir.exists():
        raise FileNotFoundError(f"DATA_DIR path not found: {data_dir}")

    # ---------------- Load dataset ----------------
    # Preferred: shared loader (keeps consistent splits across all model families)
    try:
        x_tr, y_tr_1h, x_va, y_va_1h, _x_te, _y_te = load_dataset_npy(
            data_dir, img_shape, K, val_fraction=_cfg_get(cfg, "VAL_FRACTION", 0.5)
        )
    except Exception:
        # Fallback: minimal legacy loader (expects raw *.npy files)
        x_tr = np.load(data_dir / "train_data.npy")
        y_tr = np.load(data_dir / "train_labels.npy")
        x_te = np.load(data_dir / "test_data.npy")
        y_te = np.load(data_dir / "test_labels.npy")

        n_val = int(len(x_te) * float(_cfg_get(cfg, "VAL_FRACTION", 0.5)))
        x_va, y_va = x_te[:n_val], y_te[:n_val]

        x_tr = to_01(x_tr).reshape((-1, *img_shape))
        x_va = to_01(x_va).reshape((-1, *img_shape))
        y_tr_1h = one_hot(y_tr, K)
        y_va_1h = one_hot(y_va, K)

    # ---------------- tf.data pipelines ----------------
    train_ds, val_ds = make_datasets(x_tr, y_tr_1h, x_va, y_va_1h, batch_size=batch_size)

    # ---------------- Build model & optimizer ----------------
    # The builder should produce a model that returns probs in [0,1] with same shape as x.
    model = build_conditional_pixelcnn(img_shape, K)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5)

    # ---------------- TensorBoard ----------------
    writer = make_writer(tb_dir)

    # ---------------- Train ----------------
    # Print paths explicitly so HPC logs show the exact artifact destination.
    rule_a_hint = _resolve_rule_a_dirs(cfg)
    if rule_a_hint is not None:
        print(f"[rule_a] dataset={_cfg_get(cfg,'dataset.id')} | model={_cfg_get(cfg,'model.tag')} | run={_cfg_get(cfg,'run.id')}")
        print(f"[rule_a] root={Path(rule_a_hint['root']).resolve()}")

    print(f"[config] img_shape={img_shape} | classes={K} | epochs={epochs} | bs={batch_size} | lr={lr} | seed={seed}")
    print(f"[paths]  ckpts={ckpt_dir.resolve()} | summaries={sums_dir.resolve()} | tb={tb_dir.resolve()}")
    print(f"[data]   root={data_dir.resolve()}")

    fit_autoregressive(
        model=model,
        optimizer=optimizer,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=epochs,
        checkpoint_dir=ckpt_dir,
        writer=writer,
        patience=patience,
        save_every=save_every,
        from_logits=from_logits,
    )

    # Simple marker file so orchestrators can detect completion quickly.
    (sums_dir / "train_done.txt").write_text("ok", encoding="utf_8")


def train(cfg_or_argv) -> int:
    """
    Unified_CLI entrypoint. Returns rc=0 on success (or raises on error).
    """
    cfg = _coerce_cfg(cfg_or_argv)
    _train_from_cfg(cfg)
    return 0


__all__ = [
    "TrainConfig",
    "make_datasets",
    "make_writer",
    "fit_autoregressive",
    "train",
]