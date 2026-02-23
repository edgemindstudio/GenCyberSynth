# src/gencysynth/utility/train_classifier.py
"""
Train a lightweight image classifier for utility testing.

Why this exists
---------------
In GenCyberSynth, a common "utility" question is:
  "Does Real+Synthetic improve downstream classification on REAL TEST?"

This module trains a small CNN baseline quickly and consistently, producing:
- checkpoints (Keras 3 compatible *.weights.h5)
- a JSON summary dict (returned to caller; writing handled by runner if desired)

Rule A notes
------------
- This module can be used by a unified orchestrator.
- It resolves artifact locations under:
    {paths.artifacts}/utility/classifier/<dataset_id>/...
  where dataset_id should come from cfg["data.id"] (or cfg["data.root"] fallback).
- It does not assume a single dataset layout; callers can pass arrays directly.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
import yaml

from .metrics import classification_report_dict


# ---------------------------------------------------------------------
# Small config/path helpers
# ---------------------------------------------------------------------
def _cfg_get(cfg: Dict, dotted: str, default=None):
    cur = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _as_int_labels(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Accept (N,) int labels or (N,K) one-hot labels.
    """
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == int(num_classes):
        return np.argmax(y, axis=1).astype(np.int64)
    return y.reshape(-1).astype(np.int64)


def _to_float01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x).astype(np.float32, copy=False)
    if x.max() > 1.5:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


def _normalize_artifacts(cfg: Dict) -> Dict:
    """
    Populate cfg["ARTIFACTS"]["utility_classifier"] consistently.

    Expected structure:
      paths:
        artifacts: artifacts/
      data:
        id: <dataset_id>   (preferred)
        root: <path>       (fallback)
    """
    cfg = dict(cfg)
    arts_root = Path(_cfg_get(cfg, "paths.artifacts", "artifacts"))

    dataset_id = _cfg_get(cfg, "data.id", None)
    if dataset_id is None:
        # fallback: use a filesystem-safe version of data.root
        root = str(_cfg_get(cfg, "data.root", _cfg_get(cfg, "DATA_DIR", "data")))
        dataset_id = root.replace("/", "_").replace("\\", "_").replace(" ", "_")

    cfg.setdefault("ARTIFACTS", {})
    A = cfg["ARTIFACTS"]
    base = arts_root / "utility" / "classifier" / str(dataset_id)
    A.setdefault("utility_classifier_root", str(base))
    A.setdefault("utility_classifier_ckpts", str(base / "checkpoints"))
    A.setdefault("utility_classifier_summaries", str(base / "summaries"))
    return cfg


# ---------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------
def build_small_cnn(
    *,
    img_shape: Tuple[int, int, int],
    num_classes: int,
    width: int = 32,
    dropout: float = 0.2,
) -> tf.keras.Model:
    """
    Small CNN that trains quickly (good for <5 epoch smoke tests).
    """
    H, W, C = img_shape
    inputs = tf.keras.Input(shape=(H, W, C), name="x")
    x = inputs

    x = tf.keras.layers.Conv2D(width, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(width * 2, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(width * 4, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    if dropout and dropout > 0:
        x = tf.keras.layers.Dropout(float(dropout))(x)

    logits = tf.keras.layers.Dense(int(num_classes), name="logits")(x)
    outputs = tf.keras.layers.Softmax(name="probs")(logits)

    model = tf.keras.Model(inputs, outputs, name="utility_small_cnn")
    return model


# ---------------------------------------------------------------------
# Training API
# ---------------------------------------------------------------------
@dataclass
class TrainClassifierConfig:
    img_shape: Tuple[int, int, int] = (40, 40, 1)
    num_classes: int = 9

    epochs: int = 5
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0

    width: int = 32
    dropout: float = 0.2
    seed: int = 42

    # Artifact roots (filled from cfg normalization)
    ckpt_dir: str = "artifacts/utility/classifier/default/checkpoints"
    summaries_dir: str = "artifacts/utility/classifier/default/summaries"


def train_classifier(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    cfg: Dict,
) -> Dict[str, Any]:
    """
    Train a small CNN and write checkpoints.

    Inputs
    ------
    x_*: (N,H,W,C) images in [0,1] or [0,255]
    y_*: (N,) int labels or (N,K) one-hot labels

    Returns
    -------
    dict containing:
      - paths: checkpoint paths + summary paths
      - metrics: final classification report on test
    """
    cfg = _normalize_artifacts(cfg)

    img_shape = tuple(_cfg_get(cfg, "IMG_SHAPE", _cfg_get(cfg, "img.shape", (40, 40, 1))))
    num_classes = int(_cfg_get(cfg, "NUM_CLASSES", _cfg_get(cfg, "num_classes", 9)))

    # training knobs (smoke-test friendly defaults)
    epochs = int(_cfg_get(cfg, "utility.classifier.epochs", _cfg_get(cfg, "UTILITY_EPOCHS", 5)))
    batch_size = int(_cfg_get(cfg, "utility.classifier.batch_size", _cfg_get(cfg, "BATCH_SIZE", 256)))
    lr = float(_cfg_get(cfg, "utility.classifier.lr", _cfg_get(cfg, "LR", 1e-3)))
    wd = float(_cfg_get(cfg, "utility.classifier.weight_decay", 0.0))
    width = int(_cfg_get(cfg, "utility.classifier.width", 32))
    dropout = float(_cfg_get(cfg, "utility.classifier.dropout", 0.2))
    seed = int(_cfg_get(cfg, "SEED", 42))

    ckpt_dir = Path(cfg["ARTIFACTS"]["utility_classifier_ckpts"])
    sums_dir = Path(cfg["ARTIFACTS"]["utility_classifier_summaries"])
    _ensure_dir(ckpt_dir)
    _ensure_dir(sums_dir)

    # Seeds (best-effort reproducibility)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Prepare arrays
    xtr = _to_float01(x_train).reshape((-1, *img_shape))
    xte = _to_float01(x_test).reshape((-1, *img_shape))
    ytr = _as_int_labels(y_train, num_classes=num_classes)
    yte = _as_int_labels(y_test, num_classes=num_classes)

    # Build model
    model = build_small_cnn(img_shape=img_shape, num_classes=num_classes, width=width, dropout=dropout)

    # Optimizer + loss (with optional weight decay via AdamW if available)
    try:
        opt = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
    except Exception:
        opt = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )

    # Callbacks: save best + last
    best_path = ckpt_dir / "clf_best.weights.h5"
    last_path = ckpt_dir / "clf_last.weights.h5"

    cb_best = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(best_path),
        monitor="val_acc",
        mode="max",
        save_best_only=True,
        save_weights_only=True,
        verbose=0,
    )

    # Train
    history = model.fit(
        xtr,
        ytr,
        validation_split=float(_cfg_get(cfg, "utility.classifier.val_split", 0.1)),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[cb_best],
        verbose=1,
    )

    # Save last snapshot
    model.save_weights(str(last_path))

    # Evaluate on test (prefer best if available)
    if best_path.exists():
        model.load_weights(str(best_path))

    probs = model.predict(xte, batch_size=batch_size, verbose=0)
    ypred = np.argmax(probs, axis=1).astype(np.int64)

    report = classification_report_dict(y_true=yte, y_pred=ypred, num_classes=num_classes)

    # Optional: write a compact JSON summary (Rule A: predictable, easy to inspect)
    summary_path = sums_dir / "classifier_report.json"
    summary_payload = {
        "img_shape": list(img_shape),
        "num_classes": int(num_classes),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(wd),
        "width": int(width),
        "dropout": float(dropout),
        "checkpoints": {"best": str(best_path), "last": str(last_path)},
        "metrics": report,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return {
        "paths": {
            "ckpt_dir": str(ckpt_dir),
            "best": str(best_path),
            "last": str(last_path),
            "summary": str(summary_path),
        },
        "metrics": report,
    }


# ---------------------------------------------------------------------
# CLI wrapper (optional, but handy for smoke tests)
# ---------------------------------------------------------------------
def _load_yaml(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def main(argv=None) -> int:
    """
    CLI entrypoint for quick testing.

    Example
    -------
    python -m gencysynth.utility.train_classifier --config configs/smoke.yaml

    Notes
    -----
    This CLI expects the caller's config points to dataset arrays or a loader in your repo.
    If you already have a shared loader, replace the stub in `_load_arrays_from_cfg`.
    """
    ap = argparse.ArgumentParser(description="Train a small utility classifier (smoke-test friendly).")
    ap.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    args = ap.parse_args(argv)

    cfg = _load_yaml(Path(args.config))
    cfg = _normalize_artifacts(cfg)

    # ---- Replace this stub with your repo's dataset loader ----
    # We keep it explicit so it doesn't silently guess wrong.
    data_root = Path(_cfg_get(cfg, "data.root", _cfg_get(cfg, "DATA_DIR", "data")))
    xtr_p = data_root / "train_data.npy"
    ytr_p = data_root / "train_labels.npy"
    xte_p = data_root / "test_data.npy"
    yte_p = data_root / "test_labels.npy"
    if not (xtr_p.exists() and ytr_p.exists() and xte_p.exists() and yte_p.exists()):
        raise FileNotFoundError(
            f"Expected dataset .npy files under {data_root}:\n"
            f"  train_data.npy, train_labels.npy, test_data.npy, test_labels.npy"
        )

    x_train = np.load(xtr_p)
    y_train = np.load(ytr_p)
    x_test = np.load(xte_p)
    y_test = np.load(yte_p)

    out = train_classifier(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, cfg=cfg)
    print(f"[utility] wrote summary: {out['paths']['summary']}")
    return 0


__all__ = ["train_classifier",
           "build_small_cnn",
           "main"]
