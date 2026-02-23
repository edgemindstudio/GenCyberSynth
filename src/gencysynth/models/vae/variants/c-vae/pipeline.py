# src/gencysynth/models/vae/variants/c-vae/pipeline.py
"""
Rule A — Conditional VAE (cVAE) pipeline: training + optional synthesis (.npy).

This module is the *training* side of the c-vae variant. Sampling PNGs for the
unified orchestrator lives in `sample.py` via:
    synth(cfg, output_root, seed) -> manifest

Why this file exists
--------------------
- Build encoder/decoder via variant-local `models.build_models(...)`.
- Train with a custom loop (MSE recon + beta_KL * KL).
- Write *run-scoped*, *dataset-scoped* artifacts (Rule A) using normalized paths:
    artifacts/
      runs/<dataset_id>/vae/c-vae/<run_id>/
        checkpoints/
          E_epoch_XXXX.weights.h5, D_epoch_XXXX.weights.h5
          E_best.weights.h5,       D_best.weights.h5
          E_last.weights.h5,       D_last.weights.h5
        summaries/   (optional)
        synthetic/   (optional: evaluator-friendly .npy dumps)

Rule A conventions
------------------
- Never hardcode "artifacts/vae/..."; always honor `paths.artifacts` + dataset/run scoping.
- This pipeline writes to:
    cfgN["artifacts"]["checkpoints"]
    cfgN["artifacts"]["synthetic"]
  where cfgN = normalize_cfg(cfg) from variant config.py.

Data conventions
----------------
- train(): expects images in [-1, 1] (tanh decoder convention)
- labels: one-hot (N, K)
- synthesize(): (optional) writes evaluator-friendly .npy in [0, 1]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from .models import build_models
from .config import normalize as normalize_cfg


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _ensure_dir(p: Union[str, Path]) -> Path:
    pp = p if isinstance(p, Path) else Path(str(p))
    pp.mkdir(parents=True, exist_ok=True)
    return pp


def _to_float(x) -> float:
    try:
        return float(x)
    except Exception:
        arr = np.asarray(x)
        return float(arr.reshape(-1)[0])


def _cfg_get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    np.random.seed(int(seed))
    tf.random.set_seed(int(seed))
    try:
        tf.keras.utils.set_random_seed(int(seed))
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Config (dataclass is optional, but makes the pipeline easier to reason about)
# -----------------------------------------------------------------------------
@dataclass
class VAEPipelineConfig:
    # model
    img_shape: Tuple[int, int, int] = (40, 40, 1)
    num_classes: int = 9
    latent_dim: int = 100

    # training
    epochs: int = 200
    batch_size: int = 256
    lr: float = 2e-4
    beta_1: float = 0.5
    beta_kl: float = 1.0
    log_every: int = 20
    early_stopping_patience: int = 10

    # synthesis (optional .npy dumps)
    samples_per_class: int = 1000

    # reproducibility
    seed: Optional[int] = 42

    # artifacts (Rule A)
    ckpt_dir: Path = Path("artifacts")
    synth_dir: Path = Path("artifacts")
    summaries_dir: Optional[Path] = None


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------
class VAEPipeline:
    """
    Training + optional .npy synthesis orchestration for a conditional VAE.

    Note: PNG synthesis for the unified orchestrator is handled by sample.py,
    because the orchestrator passes an explicit `output_root` and expects a
    manifest. This pipeline focuses on training and stable checkpointing.
    """

    def __init__(self, cfg: Dict[str, Any]):
        # Normalize (Rule A): ensures dataset/run-scoped artifacts paths exist.
        cfgN = normalize_cfg(cfg or {})
        self.cfgN = cfgN

        # Resolve core settings with robust fallbacks (supports legacy keys).
        img_shape = tuple(_cfg_get(cfgN, "model.img_shape", _cfg_get(cfgN, "IMG_SHAPE", (40, 40, 1))))
        num_classes = int(_cfg_get(cfgN, "model.num_classes", _cfg_get(cfgN, "NUM_CLASSES", 9)))
        latent_dim = int(_cfg_get(cfgN, "model.latent_dim", _cfg_get(cfgN, "LATENT_DIM", 100)))

        epochs = int(_cfg_get(cfgN, "train.epochs", _cfg_get(cfgN, "EPOCHS", 200)))
        batch_size = int(_cfg_get(cfgN, "train.batch_size", _cfg_get(cfgN, "BATCH_SIZE", 256)))
        lr = float(_cfg_get(cfgN, "train.lr", _cfg_get(cfgN, "LR", 2e-4)))
        beta_1 = float(_cfg_get(cfgN, "train.beta_1", _cfg_get(cfgN, "BETA_1", 0.5)))
        beta_kl = float(_cfg_get(cfgN, "train.beta_kl", _cfg_get(cfgN, "BETA_KL", 1.0)))

        log_every = int(_cfg_get(cfgN, "train.log_every", _cfg_get(cfgN, "LOG_EVERY", max(20, epochs // 40))))
        patience = int(_cfg_get(cfgN, "train.early_stopping_patience", _cfg_get(cfgN, "EARLY_STOPPING_PATIENCE", 10)))

        samples_per_class = int(_cfg_get(cfgN, "synth.samples_per_class", _cfg_get(cfgN, "SAMPLES_PER_CLASS", 1000)))

        seed = _cfg_get(cfgN, "seed", _cfg_get(cfgN, "SEED", 42))
        seed = None if seed is None else int(seed)

        # Artifacts (Rule A)
        ckpt_dir = Path(_cfg_get(cfgN, "artifacts.checkpoints", _cfg_get(cfgN, "ARTIFACTS.checkpoints", "artifacts")))
        synth_dir = Path(_cfg_get(cfgN, "artifacts.synthetic", _cfg_get(cfgN, "ARTIFACTS.synthetic", "artifacts")))
        summaries_dir_val = _cfg_get(cfgN, "artifacts.summaries", _cfg_get(cfgN, "ARTIFACTS.summaries", None))
        summaries_dir = Path(summaries_dir_val) if summaries_dir_val else None

        _ensure_dir(ckpt_dir)
        _ensure_dir(synth_dir)
        if summaries_dir is not None:
            _ensure_dir(summaries_dir)

        self.cfg = VAEPipelineConfig(
            img_shape=img_shape,
            num_classes=num_classes,
            latent_dim=latent_dim,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            beta_1=beta_1,
            beta_kl=beta_kl,
            log_every=log_every,
            early_stopping_patience=patience,
            samples_per_class=samples_per_class,
            seed=seed,
            ckpt_dir=ckpt_dir,
            synth_dir=synth_dir,
            summaries_dir=summaries_dir,
        )

        # Optional external logger callback:
        #   cb(epoch, train_total, recon, kl, val_total)
        self.log_cb = cfgN.get("LOG_CB", None)

        # Seed everything (best-effort)
        _set_seed(self.cfg.seed)

        # Build models
        mdict = build_models(
            img_shape=self.cfg.img_shape,
            latent_dim=self.cfg.latent_dim,
            num_classes=self.cfg.num_classes,
            lr=self.cfg.lr,
            beta_1=self.cfg.beta_1,
            beta_kl=self.cfg.beta_kl,
        )
        self.encoder: tf.keras.Model = mdict["encoder"]
        self.decoder: tf.keras.Model = mdict["decoder"]

        # One optimizer for both networks
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.cfg.lr, beta_1=self.cfg.beta_1)

    # -------------------------------------------------------------------------
    # Training steps
    # -------------------------------------------------------------------------
    @tf.function(reduce_retracing=True)
    def _train_step(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        One gradient update.

        Returns
        -------
        total_loss, recon_loss, kl_loss
        """
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([x_batch, y_batch], training=True)
            x_recon = self.decoder([z, y_batch], training=True)

            # Reconstruction: mean over pixels, mean over batch
            recon = tf.reduce_mean(tf.reduce_mean(tf.square(x_batch - x_recon), axis=[1, 2, 3]))

            # KL(q(z|x)||N(0,I)) per sample then mean over batch
            kl = tf.reduce_mean(
                -0.5 * tf.reduce_sum(1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )

            total = recon + self.cfg.beta_kl * kl

        vars_ = self.encoder.trainable_variables + self.decoder.trainable_variables
        grads = tape.gradient(total, vars_)

        # Guard against rare None grads (should not happen, but better error messages)
        grads_vars = [(g, v) for (g, v) in zip(grads, vars_) if g is not None]
        self.opt.apply_gradients(grads_vars)

        return total, recon, kl

    @tf.function(reduce_retracing=True)
    def _val_step(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Validation forward pass (no gradients).
        """
        z_mean, z_log_var, z = self.encoder([x_batch, y_batch], training=False)
        x_recon = self.decoder([z, y_batch], training=False)

        recon = tf.reduce_mean(tf.reduce_mean(tf.square(x_batch - x_recon), axis=[1, 2, 3]))
        kl = tf.reduce_mean(
            -0.5 * tf.reduce_sum(1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )
        total = recon + self.cfg.beta_kl * kl
        return total, recon, kl

    # -------------------------------------------------------------------------
    # Checkpoint I/O (Keras 3 friendly)
    # -------------------------------------------------------------------------
    def _save_epoch_ckpt(self, epoch: int) -> None:
        """
        Snapshot weights periodically for debugging and reproducibility.
        """
        e_path = self.cfg.ckpt_dir / f"E_epoch_{epoch:04d}.weights.h5"
        d_path = self.cfg.ckpt_dir / f"D_epoch_{epoch:04d}.weights.h5"
        self.encoder.save_weights(str(e_path))
        self.decoder.save_weights(str(d_path))

    def _save_best_ckpt(self) -> None:
        """
        Save best weights under stable names.
        """
        self.encoder.save_weights(str(self.cfg.ckpt_dir / "E_best.weights.h5"))
        self.decoder.save_weights(str(self.cfg.ckpt_dir / "D_best.weights.h5"))

    def _save_last_ckpt(self) -> None:
        """
        Save last weights (always written at end of training).
        """
        self.encoder.save_weights(str(self.cfg.ckpt_dir / "E_last.weights.h5"))
        self.decoder.save_weights(str(self.cfg.ckpt_dir / "D_last.weights.h5"))

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def train(
        self,
        *,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Tuple[tf.keras.Model, tf.keras.Model]:
        """
        Train the cVAE.

        Parameters
        ----------
        x_train : np.ndarray
            float32 images in [-1, 1], shape (N, H, W, C)
        y_train : np.ndarray
            one-hot labels, shape (N, K)
        x_val, y_val : optional validation split (same conventions)

        Returns
        -------
        (encoder, decoder)
        """
        H, W, C = self.cfg.img_shape
        K = self.cfg.num_classes

        if x_train.shape[1:] != (H, W, C):
            raise ValueError(f"Expected x_train shape (*,{H},{W},{C}), got {x_train.shape}")
        if y_train.ndim != 2 or y_train.shape[1] != K:
            raise ValueError(f"Expected y_train one-hot shape (N,{K}), got {y_train.shape}")

        have_val = (x_val is not None) and (y_val is not None)
        if have_val:
            if x_val.shape[1:] != (H, W, C):
                raise ValueError(f"Expected x_val shape (*,{H},{W},{C}), got {x_val.shape}")
            if y_val.ndim != 2 or y_val.shape[1] != K:
                raise ValueError(f"Expected y_val one-hot shape (N,{K}), got {y_val.shape}")

        steps_per_epoch = max(1, math.ceil(len(x_train) / self.cfg.batch_size))

        best_val = float("inf")
        patience_ctr = 0

        for epoch in range(1, self.cfg.epochs + 1):
            # Shuffle each epoch
            perm = np.random.permutation(len(x_train))

            train_totals, train_recons, train_kls = [], [], []

            # -------------------------
            # Training pass
            # -------------------------
            for step in range(steps_per_epoch):
                sl = slice(step * self.cfg.batch_size, (step + 1) * self.cfg.batch_size)
                idx = perm[sl]

                xb = x_train[idx].astype(np.float32, copy=False)
                yb = y_train[idx].astype(np.float32, copy=False)

                t, r, k = self._train_step(xb, yb)
                train_totals.append(_to_float(t))
                train_recons.append(_to_float(r))
                train_kls.append(_to_float(k))

            train_total = float(np.mean(train_totals)) if train_totals else float("nan")
            train_recon = float(np.mean(train_recons)) if train_recons else float("nan")
            train_kl = float(np.mean(train_kls)) if train_kls else float("nan")

            # -------------------------
            # Validation pass
            # -------------------------
            val_total = float("nan")
            if have_val:
                v_losses = []
                bs = self.cfg.batch_size
                for i in range(0, len(x_val), bs):
                    xv = x_val[i : i + bs].astype(np.float32, copy=False)
                    yv = y_val[i : i + bs].astype(np.float32, copy=False)
                    tv, _, _ = self._val_step(xv, yv)
                    v_losses.append(_to_float(tv))
                val_total = float(np.mean(v_losses)) if v_losses else float("nan")

            # -------------------------
            # Logging
            # -------------------------
            if self.log_cb:
                # Contract: cb(epoch, train_total, recon, kl, val_total)
                try:
                    self.log_cb(epoch, train_total, train_recon, train_kl, val_total)
                except Exception:
                    # Fallback console if callback fails
                    print(
                        f"[epoch {epoch:05d}] train={train_total:.4f} recon={train_recon:.4f} "
                        f"kl={train_kl:.4f} val={val_total:.4f}"
                    )
            else:
                print(
                    f"[epoch {epoch:05d}] train={train_total:.4f} recon={train_recon:.4f} "
                    f"kl={train_kl:.4f} val={val_total:.4f}"
                )

            # -------------------------
            # Periodic snapshot (debug/repro)
            # -------------------------
            if epoch == 1 or (epoch % max(1, int(self.cfg.log_every)) == 0):
                self._save_epoch_ckpt(epoch)

            # -------------------------
            # Best / early stopping
            # -------------------------
            if have_val:
                improved = val_total < best_val - 1e-6
                if improved:
                    best_val = val_total
                    patience_ctr = 0
                    self._save_best_ckpt()
                else:
                    patience_ctr += 1
                    if patience_ctr >= int(self.cfg.early_stopping_patience):
                        print(f"[c-vae] Early stopping at epoch {epoch} (best val={best_val:.4f}).")
                        break

        # Always write last snapshot
        self._save_last_ckpt()
        return self.encoder, self.decoder

    # -------------------------------------------------------------------------
    # Optional: evaluator-friendly .npy synthesis under artifacts.synthetic
    # -------------------------------------------------------------------------
    def synthesize(self, decoder: Optional[tf.keras.Model] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate class-balanced synthetic dataset and write evaluator-friendly .npy files.

        This is NOT the unified orchestrator PNG path. This writes:
          artifacts.synthetic/
            gen_class_<k>.npy
            labels_class_<k>.npy
            x_synth.npy
            y_synth.npy

        Returns
        -------
        (x_synth, y_synth)
          x_synth: float32 in [0,1], shape (K*n_pc, H, W, C)
          y_synth: one-hot labels,     shape (K*n_pc, K)
        """
        H, W, C = self.cfg.img_shape
        K = self.cfg.num_classes
        n_pc = int(self.cfg.samples_per_class)

        G = decoder if decoder is not None else self.decoder

        xs, ys = [], []
        _ensure_dir(self.cfg.synth_dir)

        # Use the pipeline seed if present (keeps synthesis reproducible per run)
        _set_seed(self.cfg.seed)

        for cls in range(K):
            z = np.random.normal(0.0, 1.0, size=(n_pc, self.cfg.latent_dim)).astype(np.float32)
            y = tf.keras.utils.to_categorical([cls] * n_pc, K).astype(np.float32)

            g = G.predict([z, y], verbose=0)                 # [-1, 1]
            g01 = np.clip((g + 1.0) / 2.0, 0.0, 1.0)         # -> [0, 1]
            g01 = g01.reshape((-1, H, W, C)).astype(np.float32, copy=False)

            xs.append(g01)
            ys.append(y)

            # Per-class traceability (matches evaluator expectations)
            np.save(self.cfg.synth_dir / f"gen_class_{cls}.npy", g01)
            np.save(self.cfg.synth_dir / f"labels_class_{cls}.npy", np.full((n_pc,), cls, dtype=np.int32))

        x_synth = np.concatenate(xs, axis=0) if xs else np.empty((0, H, W, C), dtype=np.float32)
        y_synth = np.concatenate(ys, axis=0) if ys else np.empty((0, K), dtype=np.float32)

        np.save(self.cfg.synth_dir / "x_synth.npy", x_synth)
        np.save(self.cfg.synth_dir / "y_synth.npy", y_synth)

        print(f"[c-vae][synthesize] {x_synth.shape[0]} samples ({n_pc}/class) -> {self.cfg.synth_dir}")
        return x_synth, y_synth


__all__ = ["VAEPipeline",
           "VAEPipelineConfig"]
