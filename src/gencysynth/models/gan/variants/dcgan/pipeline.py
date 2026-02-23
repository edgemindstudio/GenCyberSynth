# src/gencysynth/models/gan/variants/dcgan/pipeline.py
"""
GenCyberSynth — GAN family — DCGAN variant (Conditional) — Pipeline

This pipeline provides:
- Training loop (fit D and G)
- Checkpoint management (Keras 3 *.weights.h5)
- Balanced synthesis to disk (numpy artifacts)

Variant contract location
-------------------------
This file must live under:

  src/gencysynth/models/gan/variants/dcgan/
    ├── model.py
    ├── train.py
    ├── sample.py
    ├── pipeline.py   <-- (this file)
    └── defaults.yaml

Key design rule (professional repo hygiene)
-------------------------------------------
This module should NOT hardcode ad-hoc artifact directories.
Instead it resolves artifacts from cfg["paths"] with reasonable defaults.

Recommended canonical paths
---------------------------
cfg["paths"]["artifacts"] = "artifacts"

Then derive:
- checkpoints:
    artifacts/models/<family>/<variant>/checkpoints[/<run_id>]
- synthetic:
    artifacts/synth[/<run_id>]/<family>/<variant>/

If your orchestration passes a run_id, all outputs become run-scoped and auditable.
"""

from __future__ import annotations

import glob
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import numpy as np
import tensorflow as tf

# This should be a relative import in the variant folder.
# If you keep absolute imports, ensure your package installation supports it.
from .model import build_models


# ---------------------------------------------------------------------
# Variant identity (makes logs + artifacts self-describing)
# ---------------------------------------------------------------------
FAMILY: str = "gan"
VARIANT: str = "dcgan"


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------
def _to_scalar(x) -> float:
    """
    Normalize TF/Keras outputs to float.

    Keras can return:
    - scalar tensors
    - numpy scalars
    - lists like [loss, accuracy]
    """
    if isinstance(x, (list, tuple)):
        return _to_scalar(x[0])
    try:
        return float(x)
    except Exception:
        arr = np.asarray(x)
        return float(arr.reshape(-1)[0])


def _loss_and_acc(batch_out) -> Tuple[float, Optional[float]]:
    """
    Parse model.train_on_batch(...) output into:
        (loss, accuracy_or_None)
    """
    if isinstance(batch_out, (list, tuple)):
        loss = _to_scalar(batch_out[0])
        acc = _to_scalar(batch_out[1]) if len(batch_out) > 1 else None
        return loss, acc
    return _to_scalar(batch_out), None


def _ensure_dir(p: Path) -> None:
    """mkdir -p"""
    p.mkdir(parents=True, exist_ok=True)


def _cfg_get(cfg: dict, dotted: str, default=None):
    """
    Nested dict getter using dotted keys, e.g. "paths.artifacts".
    Returns default if missing.
    """
    cur = cfg
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _pick_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """
    Choose which generator weights to load.

    Priority:
      1) G_best.weights.h5
      2) G_last.weights.h5
      3) latest epoch checkpoint G_epoch_*.weights.h5
      4) legacy *.h5 equivalents (if any)
    """
    preferred = [
        ckpt_dir / "G_best.weights.h5",
        ckpt_dir / "G_last.weights.h5",
    ]
    for p in preferred:
        if p.exists():
            return p

    # Latest epoch checkpoint (Keras 3 convention)
    epoch_ckpts = sorted(ckpt_dir.glob("G_epoch_*.weights.h5"))
    if epoch_ckpts:
        return max(epoch_ckpts, key=lambda x: x.stat().st_mtime)

    # Legacy fallbacks (avoid breaking old runs)
    legacy = [
        ckpt_dir / "G_best.h5",
        ckpt_dir / "G_last.h5",
    ]
    for p in legacy:
        if p.exists():
            return p

    legacy_epochs = glob.glob(str(ckpt_dir / "G_epoch_*.h5"))
    if legacy_epochs:
        latest = max(legacy_epochs, key=os.path.getmtime)
        return Path(latest)

    return None


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------
class ConditionalDCGANPipeline:
    """
    Training + synthesis orchestrator for the Conditional DCGAN variant.

    Inputs (expected by train)
    --------------------------
    x_train: float images in [-1, 1], shape (N,H,W,C)
    y_train: one-hot labels, shape (N,num_classes)

    Outputs (artifacts)
    -------------------
    - Generator checkpoints (*.weights.h5)
    - Synthetic dataset dumps (*.npy)

    Variant identity is baked into output folders via:
        artifacts/models/gan/dcgan/...
        artifacts/synth/gan/dcgan/...
    """

    DEFAULTS = {
        "IMG_SHAPE": (40, 40, 1),
        "NUM_CLASSES": 9,
        "LATENT_DIM": 100,
        "EPOCHS": 2000,
        "BATCH_SIZE": 256,
        "LR": 2e-4,
        "BETA_1": 0.5,
        "NOISE_AFTER": 200,
        "SAMPLES_PER_CLASS": 1000,
        # Legacy compatibility if cfg["paths"] not provided:
        "ARTIFACTS": {
            "checkpoints": "artifacts/gan/checkpoints",
            "synthetic": "artifacts/gan/synthetic",
        },
    }

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg or {}
        d = self.DEFAULTS

        # -----------------------------
        # Core hyperparameters (cfg first, then defaults)
        # -----------------------------
        self.img_shape: Tuple[int, int, int] = tuple(self.cfg.get("IMG_SHAPE", d["IMG_SHAPE"]))
        self.num_classes: int = int(self.cfg.get("NUM_CLASSES", d["NUM_CLASSES"]))
        self.latent_dim: int = int(self.cfg.get("LATENT_DIM", d["LATENT_DIM"]))
        self.epochs: int = int(self.cfg.get("EPOCHS", d["EPOCHS"]))
        self.batch_size: int = int(self.cfg.get("BATCH_SIZE", d["BATCH_SIZE"]))
        self.lr: float = float(self.cfg.get("LR", d["LR"]))
        self.beta_1: float = float(self.cfg.get("BETA_1", d["BETA_1"]))
        self.noise_after: int = int(self.cfg.get("NOISE_AFTER", d["NOISE_AFTER"]))

        # Roughly 40 logs across the run, but never less than 50 epochs between saves
        self.log_every: int = int(self.cfg.get("LOG_EVERY", max(50, self.epochs // 40)))
        self.samples_per_class: int = int(self.cfg.get("SAMPLES_PER_CLASS", d["SAMPLES_PER_CLASS"]))

        # Optional run_id helps keep artifacts auditable per run
        self.run_id: Optional[str] = self.cfg.get("run_id", _cfg_get(self.cfg, "run.run_id", None))

        # -----------------------------
        # Artifact path resolution (preferred: cfg["paths"])
        # -----------------------------
        artifacts_root = Path(_cfg_get(self.cfg, "paths.artifacts", "artifacts"))

        # Canonical recommended layout
        # (1) checkpoints: artifacts/models/<family>/<variant>/checkpoints[/<run_id>]
        ckpt_dir_canonical = artifacts_root / "models" / FAMILY / VARIANT / "checkpoints"
        if self.run_id:
            ckpt_dir_canonical = ckpt_dir_canonical / self.run_id

        # (2) synth: artifacts/synth[/<run_id>]/<family>/<variant>/
        synth_dir_canonical = artifacts_root / "synth"
        if self.run_id:
            synth_dir_canonical = synth_dir_canonical / self.run_id
        synth_dir_canonical = synth_dir_canonical / FAMILY / VARIANT

        # Allow explicit overrides if provided
        ckpt_override = _cfg_get(self.cfg, "paths.gan_checkpoints", None)
        synth_override = _cfg_get(self.cfg, "paths.synthetic", None)

        if ckpt_override:
            self.ckpt_dir = Path(ckpt_override)
        else:
            # Legacy compatibility: cfg["ARTIFACTS"]["checkpoints"]
            arts = self.cfg.get("ARTIFACTS", d["ARTIFACTS"])
            self.ckpt_dir = Path(arts.get("checkpoints", str(ckpt_dir_canonical)))

            # If legacy path is used, we still prefer canonical if it's not explicitly set
            # by the user, so we only keep legacy when it exists in cfg.
            if "ARTIFACTS" not in self.cfg:
                self.ckpt_dir = ckpt_dir_canonical

        if synth_override:
            self.synth_dir = Path(synth_override)
        else:
            arts = self.cfg.get("ARTIFACTS", d["ARTIFACTS"])
            self.synth_dir = Path(arts.get("synthetic", str(synth_dir_canonical)))
            if "ARTIFACTS" not in self.cfg:
                self.synth_dir = synth_dir_canonical

        _ensure_dir(self.ckpt_dir)
        _ensure_dir(self.synth_dir)

        # Optional logging callback: cb(epoch:int, d_loss or (d_loss,d_acc), g_loss)
        self.log_cb = self.cfg.get("LOG_CB", None)

        # -----------------------------
        # Build the models (compiled inside build_models)
        # -----------------------------
        models_dict = build_models(
            latent_dim=self.latent_dim,
            num_classes=self.num_classes,
            img_shape=self.img_shape,
            lr=self.lr,
            beta_1=self.beta_1,
        )
        self.G: tf.keras.Model = models_dict["generator"]
        self.D: tf.keras.Model = models_dict["discriminator"]
        self.GAN: tf.keras.Model = models_dict["gan"]

        self._log(
            f"[{FAMILY}.{VARIANT}] Initialized pipeline | "
            f"ckpt_dir={self.ckpt_dir} | synth_dir={self.synth_dir} | run_id={self.run_id}"
        )

    # -------- small print logger --------
    def _log(self, msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {msg}")

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------
    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> Tuple[tf.keras.Model, tf.keras.Model]:
        """
        Train the conditional DCGAN.

        Parameters
        ----------
        x_train:
            Images normalized to [-1, 1], shape (N,H,W,C).
        y_train:
            One-hot labels, shape (N,num_classes).

        Returns
        -------
        (G, D):
            Trained generator and discriminator.
        """
        H, W, C = self.img_shape
        assert x_train.shape[1:] == (H, W, C), (
            f"[{FAMILY}.{VARIANT}] Expected x_train shape (*,{H},{W},{C}), got {x_train.shape}"
        )
        assert y_train.shape[1] == self.num_classes, (
            f"[{FAMILY}.{VARIANT}] y_train must be one-hot with {self.num_classes} columns"
        )

        steps_per_epoch = max(1, math.ceil(len(x_train) / self.batch_size))
        best_g_loss = float("inf")

        self._log(f"[{FAMILY}.{VARIANT}] Training start | epochs={self.epochs}, steps/epoch={steps_per_epoch}")

        for epoch in range(self.epochs):
            perm = np.random.permutation(len(x_train))
            d_losses, g_losses = [], []
            d_acc_epoch = None  # last observed batch accuracy (if D reports it)

            for step in range(steps_per_epoch):
                sl = slice(step * self.batch_size, (step + 1) * self.batch_size)
                idx = perm[sl]

                # Real samples (images + their real labels)
                real_imgs = x_train[idx].astype(np.float32)
                real_lbls = y_train[idx].astype(np.float32)

                n = real_imgs.shape[0]

                # Fake samples (noise + randomly chosen class labels)
                z = np.random.normal(0, 1, (n, self.latent_dim)).astype(np.float32)
                fake_cls = np.random.randint(0, self.num_classes, size=(n, 1))
                fake_lbls = tf.keras.utils.to_categorical(fake_cls, self.num_classes).astype(np.float32)

                gen_imgs = self.G.predict([z, fake_lbls], verbose=0)

                # Stabilization trick: add light noise after warmup
                if epoch + 1 > self.noise_after:
                    real_imgs = real_imgs + np.random.normal(0, 0.01, real_imgs.shape)
                    gen_imgs = gen_imgs + np.random.normal(0, 0.01, gen_imgs.shape)

                # Label smoothing / noisy labels
                real_y = np.random.uniform(0.9, 1.0, size=(n, 1)).astype(np.float32)
                fake_y = np.random.uniform(0.0, 0.1, size=(n, 1)).astype(np.float32)

                # -------------------------
                # Discriminator step
                # -------------------------
                self.D.trainable = True
                d_out_real = self.D.train_on_batch([real_imgs, real_lbls], real_y)
                d_out_fake = self.D.train_on_batch([gen_imgs, fake_lbls], fake_y)

                d_real_loss, d_real_acc = _loss_and_acc(d_out_real)
                d_fake_loss, d_fake_acc = _loss_and_acc(d_out_fake)

                d_loss = 0.5 * (d_real_loss + d_fake_loss)
                d_losses.append(d_loss)

                if (d_real_acc is not None) and (d_fake_acc is not None):
                    d_acc_epoch = 0.5 * (d_real_acc + d_fake_acc)

                # -------------------------
                # Generator step
                # -------------------------
                self.D.trainable = False
                g_out = self.GAN.train_on_batch([z, fake_lbls], np.ones((n, 1), dtype=np.float32))
                g_loss = _to_scalar(g_out)
                g_losses.append(g_loss)

            # Epoch summaries
            d_mean = float(np.mean(d_losses)) if d_losses else float("nan")
            g_mean = float(np.mean(g_losses)) if g_losses else float("nan")

            # Optional external logger (app/main.py can add TB logging, etc.)
            if self.log_cb:
                self.log_cb(epoch + 1, (d_mean, d_acc_epoch) if d_acc_epoch is not None else d_mean, g_mean)

            # Periodic checkpointing (Keras 3 requires *.weights.h5 for save_weights)
            if (epoch + 1) % self.log_every == 0 or epoch == 0:
                ckpt_path = self.ckpt_dir / f"G_epoch_{epoch+1:04d}.weights.h5"
                self.G.save_weights(str(ckpt_path))

            # Track best generator by mean g_loss (simple heuristic)
            if g_mean < best_g_loss:
                best_g_loss = g_mean
                self.G.save_weights(str(self.ckpt_dir / "G_best.weights.h5"))

        # Always save final generator checkpoint
        self.G.save_weights(str(self.ckpt_dir / "G_last.weights.h5"))

        self._log(f"[{FAMILY}.{VARIANT}] Training complete | best_g_loss={best_g_loss:.6f}")
        self._log(f"[{FAMILY}.{VARIANT}] Checkpoints written to: {self.ckpt_dir}")

        return self.G, self.D

    # -----------------------------------------------------------------
    # Synthesis
    # -----------------------------------------------------------------
    def synthesize(self, G: Optional[tf.keras.Model] = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a balanced per-class synthetic dataset and save it under self.synth_dir.

        Returns
        -------
        (x_synth, y_synth):
          x_synth: float32 in [0,1], shape (K*per_class, H, W, C)
          y_synth: one-hot, shape (K*per_class, K)

        Artifacts written (traceability)
        -------------------------------
        Per class:
          - gen_class_<k>.npy         images in [0,1]
          - labels_class_<k>.npy      int labels

        Combined convenience:
          - x_synth.npy
          - y_synth.npy
        """
        H, W, C = self.img_shape
        per_class = int(self.cfg.get("SAMPLES_PER_CLASS", self.samples_per_class))

        # -------------------------
        # Build + load generator if needed
        # -------------------------
        ckpt_used: Optional[Path] = None

        if G is None:
            md = build_models(
                latent_dim=self.latent_dim,
                num_classes=self.num_classes,
                img_shape=self.img_shape,
                lr=self.lr,
                beta_1=self.beta_1,
            )
            G = md["generator"]

            ckpt_used = _pick_checkpoint(self.ckpt_dir)
            if ckpt_used:
                G.load_weights(str(ckpt_used))
                self._log(f"[{FAMILY}.{VARIANT}] Loaded generator weights: {ckpt_used.name}")
            else:
                self._log(f"[{FAMILY}.{VARIANT}][warn] No generator weights found; using untrained generator.")

        # Ensure synth dir exists and is variant-scoped (already in __init__)
        _ensure_dir(self.synth_dir)

        # -------------------------
        # Generate per-class data
        # -------------------------
        xs, ys = [], []

        for cls in range(self.num_classes):
            # Latent noise
            z = np.random.normal(0, 1, (per_class, self.latent_dim)).astype(np.float32)

            # One-hot labels
            y = tf.keras.utils.to_categorical(
                np.full((per_class, 1), cls),
                self.num_classes
            ).astype(np.float32)

            # Predict -> [-1,1], then rescale -> [0,1] for storage/use
            g = G.predict([z, y], verbose=0)
            g01 = np.clip((g + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)

            xs.append(g01.reshape(-1, H, W, C))
            ys.append(y)

            # Per-class dumps (good for debugging imbalance and reproducibility)
            np.save(self.synth_dir / f"gen_class_{cls}.npy", g01)
            np.save(self.synth_dir / f"labels_class_{cls}.npy", np.full((per_class,), cls, dtype=np.int32))

        x_synth = np.concatenate(xs, axis=0)
        y_synth = np.concatenate(ys, axis=0)

        # Combined dumps (convenience, not required but helpful)
        np.save(self.synth_dir / "x_synth.npy", x_synth)
        np.save(self.synth_dir / "y_synth.npy", y_synth)

        self._log(
            f"[{FAMILY}.{VARIANT}] Synthesized {x_synth.shape[0]} samples "
            f"({per_class} per class) -> {self.synth_dir}"
        )
        if ckpt_used:
            self._log(f"[{FAMILY}.{VARIANT}] Synthesis used checkpoint: {ckpt_used}")

        return x_synth, y_synth
