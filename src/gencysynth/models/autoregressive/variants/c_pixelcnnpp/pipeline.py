# src/gencysynth/models/autoregressive/variants/c_pixelcnnpp/pipeline.py

"""
GenCyberSynth — Autoregressive — c_pixelcnnpp — Pipeline
=======================================================

This file provides a *model_family consistent* train + synth wrapper, matching
the same expectations we used for GAN (Rule A), so the evaluator and orchestrator
can treat every generator identically.

Rule A (Artifacts + scalability)
--------------------------------
We want artifacts to scale across:
  - many datasets (different roots, shapes, label sets)
  - many model families (gan, diffusion, autoregressive, gmm, rbm, ...)
  - many variants (c_pixelcnnpp, c_maf_rq, c_ddpm, ...)

Preferred layout (Rule A):
  artifacts/runs/<dataset_id>/<model_tag>/<run_id>/
    checkpoints/     # *.weights.h5 etc.
    synthetic/        # evaluator_friendly .npy dumps
    summaries/        # preview images, small status markers
    tensorboard/      # optional

This pipeline:
  - reads/writes from Rule_A run folders *when run context is present*
  - falls back to legacy artifact folders if run context is absent
  - writes evaluator_friendly per_class .npy dumps to synthetic/

Contracts
---------
Training:
  - checkpts: AR_best.weights.h5, AR_last.weights.h5, AR_epoch_XXXX.weights.h5

Synthesis:
  - synthetic/gen_class_{k}.npy
  - synthetic/labels_class_{k}.npy
  - synthetic/x_synth.npy, synthetic/y_synth.npy

Data conventions:
  - x: (N,H,W,C) float32 in [0,1]
  - y: (N,K) one_hot float32

Notes
-----
Autoregressive sampling is raster_scan and will be slow compared to GANs.
This is fine for baselines and for smaller sample budgets on HPC.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any, Mapping

import numpy as np
import tensorflow as tf

# IMPORTANT: keep imports local to this variant folder structure.
# The builder should live alongside this pipeline (or be re_exported cleanly).
# If your repo uses a different import path, adjust ONLY here.
from autoregressive.models import build_ar_model


# =============================================================================
# Small utilities
# =============================================================================
def _as_float(x) -> float:
    """Convert scalars / 0_D tensors / arrays to a Python float (for logs)."""
    try:
        return float(x)
    except Exception:
        arr = np.asarray(x)
        return float(arr.reshape(-1)[0])


def _ensure_dir(p: Path) -> Path:
    """Create directory (including parents) if missing; return the Path."""
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cfg_get(cfg: Mapping[str, Any], dotted: str, default=None):
    """Dotted key getter: _cfg_get(cfg, "paths.artifacts", "artifacts")."""
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _safe_slug(x: str) -> str:
    """Filesystem_safe slug for dataset/model/run identifiers."""
    return (
        str(x)
        .strip()
        .replace("\\", "/")
        .replace(" ", "_")
        .replace(":", "_")
    )


def _one_hot_from_ids(ids: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer class ids -> one_hot float32."""
    return tf.keras.utils.to_categorical(ids.astype(int), num_classes=num_classes).astype("float32")


# =============================================================================
# Rule A: run directory resolution
# =============================================================================
def _resolve_rule_a_run_dirs(cfg: Dict) -> Optional[Dict[str, Path]]:
    """
    Resolve Rule_A run directories if run context is present.

    Accepted keys (any of these patterns):
      dataset.id          or data.dataset_id or DATASET_ID
      model.tag           or model_tag
      run.id              or run_id or run_meta.run_id

    Returns
    -------
    dict with run_root and standard subfolders, or None if any field missing.
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

    # Where all runs live (separate from generic "artifacts/")
    runs_root = Path(_cfg_get(cfg, "paths.runs", _cfg_get(cfg, "paths.artifacts_runs", "artifacts/runs")))
    run_root = runs_root / _safe_slug(str(dataset_id)) / _safe_slug(str(model_tag)) / _safe_slug(str(run_id))

    return {
        "root": run_root,
        "checkpoints": run_root / "checkpoints",
        "synthetic": run_root / "synthetic",
        "summaries": run_root / "summaries",
        "tensorboard": run_root / "tensorboard",
    }


def _resolve_artifact_dirs(cfg: Dict) -> Dict[str, Path]:
    """
    Resolve artifact directories.

    Priority:
      1) Rule_A run folders (if dataset/model/run is set)
      2) Explicit per_model ARTIFACTS.* overrides (legacy style)
      3) Legacy default under <paths.artifacts>/autoregressive/*
    """
    ra = _resolve_rule_a_run_dirs(cfg)
    if ra is not None:
        return ra

    artifacts_root = Path(_cfg_get(cfg, "paths.artifacts", "artifacts"))
    legacy_root = artifacts_root / "autoregressive"

    ckpt = Path(_cfg_get(cfg, "ARTIFACTS.autoregressive_checkpoints", legacy_root / "checkpoints"))
    synth = Path(_cfg_get(cfg, "ARTIFACTS.autoregressive_synthetic", legacy_root / "synthetic"))
    sums = Path(_cfg_get(cfg, "ARTIFACTS.autoregressive_summaries", legacy_root / "summaries"))
    tb = Path(_cfg_get(cfg, "ARTIFACTS.autoregressive_tensorboard", legacy_root / "tensorboard"))

    return {"root": legacy_root, "checkpoints": ckpt, "synthetic": synth, "summaries": sums, "tensorboard": tb}


# =============================================================================
# Pipeline
# =============================================================================
@dataclass
class ARPipelineDefaults:
    """Reasonable defaults for the PixelCNN++-style conditional baseline."""
    IMG_SHAPE: Tuple[int, int, int] = (40, 40, 1)
    NUM_CLASSES: int = 9

    # Training
    EPOCHS: int = 200
    BATCH_SIZE: int = 128
    LR: float = 2e_4
    BETA_1: float = 0.5
    LOG_EVERY: int = 25
    PATIENCE: int = 10

    # Model architecture knobs (builder_specific)
    NUM_FILTERS: int = 64
    NUM_LAYERS: int = 4
    NUM_HEADS: int = 4
    FF_MULT: int = 2

    # Synthesis
    SAMPLES_PER_CLASS: int = 1000


class ARAutoregressivePipeline:
    """
    Orchestrates train + synth for the conditional autoregressive model.

    This object is intentionally "small glue" around:
      - a model builder
      - checkpointing
      - evaluator_friendly artifact dumps
    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg or {}
        d = ARPipelineDefaults()

        # ---------- Shapes / basics ----------
        self.img_shape: Tuple[int, int, int] = tuple(self.cfg.get("IMG_SHAPE", d.IMG_SHAPE))
        self.num_classes: int = int(self.cfg.get("NUM_CLASSES", d.NUM_CLASSES))

        # ---------- Training hyperparameters ----------
        self.epochs: int = int(self.cfg.get("EPOCHS", d.EPOCHS))
        self.batch_size: int = int(self.cfg.get("BATCH_SIZE", d.BATCH_SIZE))
        self.lr: float = float(self.cfg.get("LR", d.LR))
        self.beta_1: float = float(self.cfg.get("BETA_1", d.BETA_1))
        self.log_every: int = int(self.cfg.get("LOG_EVERY", d.LOG_EVERY))
        self.patience: int = int(self.cfg.get("PATIENCE", d.PATIENCE))

        # ---------- Model architecture ----------
        self.num_filters: int = int(self.cfg.get("NUM_FILTERS", d.NUM_FILTERS))
        self.num_layers: int = int(self.cfg.get("NUM_LAYERS", d.NUM_LAYERS))
        self.num_heads: int = int(self.cfg.get("NUM_HEADS", d.NUM_HEADS))
        self.ff_mult: int = int(self.cfg.get("FF_MULT", d.FF_MULT))

        # ---------- Synthesis ----------
        self.samples_per_class: int = int(self.cfg.get("SAMPLES_PER_CLASS", d.SAMPLES_PER_CLASS))

        # ---------- Artifacts (Rule A preferred) ----------
        dirs = _resolve_artifact_dirs(self.cfg)
        self.run_root = Path(dirs["root"])
        self.ckpt_dir = _ensure_dir(Path(dirs["checkpoints"]))
        self.synth_dir = _ensure_dir(Path(dirs["synthetic"]))
        self.summ_dir = _ensure_dir(Path(dirs["summaries"]))

        # Optional external logging hook:
        #   LOG_CB(epoch:int, train_loss:float|None, val_loss:float|None)
        self.log_cb = self.cfg.get("LOG_CB", None)

        # Build a fresh compiled model (builder should compile internally).
        self.model: tf.keras.Model = build_ar_model(
            img_shape=self.img_shape,
            num_classes=self.num_classes,
            num_filters=self.num_filters,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            ff_multiplier=self.ff_mult,
            learning_rate=self.lr,
            beta_1=self.beta_1,
        )

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> tf.keras.Model:
        """
        Fit the AR model and write checkpoints using Rule_A directories.

        Expected inputs:
          x_train: float32 (N,H,W,C) in [0,1]
          y_train: float32 (N,K) one_hot
        """
        H, W, C = self.img_shape
        if tuple(x_train.shape[1:]) != (H, W, C):
            raise ValueError(f"x_train shape mismatch: expected (*,{H},{W},{C}) got {x_train.shape}")
        if y_train.ndim != 2 or y_train.shape[1] != self.num_classes:
            raise ValueError(f"y_train must be one_hot (N,{self.num_classes}); got {y_train.shape}")

        has_val = (x_val is not None) and (y_val is not None)
        if has_val:
            if tuple(x_val.shape[1:]) != (H, W, C):
                raise ValueError(f"x_val shape mismatch: expected (*,{H},{W},{C}) got {x_val.shape}")
            if y_val.ndim != 2 or y_val.shape[1] != self.num_classes:
                raise ValueError(f"y_val must be one_hot (N,{self.num_classes}); got {y_val.shape}")

        # ---------------- Callbacks (checkpointing, early stopping) ----------------
        callbacks: List[tf.keras.callbacks.Callback] = []

        # If no validation set is provided, monitor training loss.
        monitor_metric = "val_loss" if has_val else "loss"

        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor_metric,
                patience=self.patience,
                restore_best_weights=True,
            )
        )
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.ckpt_dir / "AR_best.weights.h5"),
                monitor=monitor_metric,
                save_best_only=True,
                save_weights_only=True,
            )
        )

        # Periodic epoch snapshots + optional external logging callback.
        class _PeriodicSaver(tf.keras.callbacks.Callback):
            def __init__(self, outer, log_every: int):
                super().__init__()
                self.outer = outer
                self.log_every = max(1, int(log_every))

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                e = epoch + 1

                # external logger hook (optional)
                if self.outer.log_cb is not None:
                    tr = logs.get("loss")
                    vl = logs.get("val_loss")
                    self.outer.log_cb(
                        e,
                        _as_float(tr) if tr is not None else None,
                        _as_float(vl) if vl is not None else None,
                    )

                # periodic snapshot
                if (e == 1) or (e % self.log_every == 0):
                    path = self.outer.ckpt_dir / f"AR_epoch_{e:04d}.weights.h5"
                    self.model.save_weights(str(path))

        callbacks.append(_PeriodicSaver(self, self.log_every))

        # ---------------- Fit ----------------
        self.model.fit(
            x=[x_train, y_train],
            y=x_train,  # autoregressive models learn p(x|y); training is "reconstruct x"
            validation_data=([x_val, y_val], x_val) if has_val else None,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            callbacks=callbacks,
            verbose=0,  # use LOG_CB or downstream logs
        )

        # Save "last" checkpoint always.
        self.model.save_weights(str(self.ckpt_dir / "AR_last.weights.h5"))

        # Lightweight marker for orchestrators (optional but nice).
        (self.summ_dir / "train_done.txt").write_text("ok\n", encoding="utf_8")

        return self.model

    # -------------------------------------------------------------------------
    # Synthesis
    # -------------------------------------------------------------------------
    def synthesize(self, model: Optional[tf.keras.Model] = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a class_balanced synthetic dataset via raster_scan sampling.

        If `model` is None:
          - build a fresh model
          - load best/last/latest_epoch checkpoint from self.ckpt_dir if present

        Writes evaluator_friendly artifacts to:
          <run>/synthetic/
        """
        if model is None:
            model = build_ar_model(
                img_shape=self.img_shape,
                num_classes=self.num_classes,
                num_filters=self.num_filters,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                ff_multiplier=self.ff_mult,
                learning_rate=self.lr,
                beta_1=self.beta_1,
            )
            ckpt = self._latest_checkpoint()
            if ckpt is not None:
                model.load_weights(str(ckpt))

        per_class = int(self.samples_per_class)
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []

        # Ensure destination exists.
        _ensure_dir(self.synth_dir)

        # Generate class_by_class to keep memory bounded and filenames deterministic.
        for cls in range(self.num_classes):
            labels = np.full((per_class,), cls, dtype=np.int32)
            y_onehot = _one_hot_from_ids(labels, self.num_classes)

            gen = self._sample_autoregressive(model, batch_size=per_class, y_onehot=y_onehot)
            gen = np.clip(gen, 0.0, 1.0).astype("float32")

            xs.append(gen)
            ys.append(y_onehot)

            # Evaluator contract: per_class dumps.
            np.save(self.synth_dir / f"gen_class_{cls}.npy", gen)
            np.save(self.synth_dir / f"labels_class_{cls}.npy", labels)

        x_synth = np.concatenate(xs, axis=0).astype("float32")
        y_synth = np.concatenate(ys, axis=0).astype("float32")

        # Convenience combined dumps.
        np.save(self.synth_dir / "x_synth.npy", x_synth)
        np.save(self.synth_dir / "y_synth.npy", y_synth)

        # Optional metadata for provenance.
        meta = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "img_shape": list(self.img_shape),
            "num_classes": int(self.num_classes),
            "samples_per_class": int(per_class),
            "run_root": str(self.run_root),
        }
        try:
            import json
            (self.synth_dir / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf_8")
        except Exception:
            # Non_fatal: evaluator only needs the .npy files.
            pass

        print(f"[synthesize] {x_synth.shape[0]} samples ({per_class} per class) -> {self.synth_dir}")
        return x_synth, y_synth

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------
    def _latest_checkpoint(self) -> Optional[Path]:
        """
        Choose a checkpoint to load for synthesis:
          prefer AR_best.weights.h5, then AR_last.weights.h5,
          else newest AR_epoch_*.weights.h5 (or legacy *.h5).
        """
        order = [
            self.ckpt_dir / "AR_best.weights.h5",
            self.ckpt_dir / "AR_last.weights.h5",
        ]

        epoch_ckpts = sorted(self.ckpt_dir.glob("AR_epoch_*.weights.h5"))
        if epoch_ckpts:
            order.append(max(epoch_ckpts, key=lambda p: p.stat().st_mtime))

        for p in order:
            if p.exists():
                return p

        # Legacy: older naming without ".weights"
        legacy = sorted(self.ckpt_dir.glob("AR_epoch_*.h5"))
        return max(legacy, key=lambda p: p.stat().st_mtime) if legacy else None

    def _sample_autoregressive(
        self,
        model: tf.keras.Model,
        batch_size: int,
        y_onehot: np.ndarray,
    ) -> np.ndarray:
        """
        Pixel_wise raster scan sampling (vectorized over batch).

        The model predicts per_pixel probabilities p(x_ij | x_<ij, y).
        We sample each pixel (and channel) as Bernoulli(u < p).

        Returns
        -------
        imgs : (B,H,W,C) float32 in [0,1]
        """
        H, W, C = self.img_shape
        imgs = np.zeros((batch_size, H, W, C), dtype=np.float32)

        # Raster order: top_left -> bottom_right.
        for i in range(H):
            for j in range(W):
                # One forward pass gives probabilities for all pixels.
                # Simpler and consistent with baseline; still vectorized across batch.
                probs = model.predict([imgs, y_onehot], verbose=0)  # (B,H,W,C)
                pij = probs[:, i, j, :]  # (B,C)

                # Bernoulli sampling (channel_wise).
                u = np.random.rand(batch_size, C).astype(np.float32)
                imgs[:, i, j, :] = (u < pij).astype(np.float32)

        return imgs


# Back_compat alias (some code imports AutoregressivePipeline)
AutoregressivePipeline = ARAutoregressivePipeline

__all__ = ["ARAutoregressivePipeline", "AutoregressivePipeline"]