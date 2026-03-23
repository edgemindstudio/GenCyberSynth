# src/gencysynth/models/maskedautoflow/variants/maf_affine/pipeline.py

"""
GenCyberSynth — MaskedAutoFlow — maf_affine — Pipeline (Rule A)

This pipeline trains and samples an *unconditional* Masked Autoregressive Flow (MAF).

Key behaviors
-------------
1) Train
   - Fits ONE global (unconditional) density model on flattened images in [0,1].
   - Saves Keras 3–friendly checkpoints:
       MAF_best.weights.h5
       MAF_last.weights.h5

2) Synthesize
   - Because the model is unconditional, it cannot truly condition on class labels.
   - For evaluator compatibility, we generate K*S samples and *assign* them to
     classes by slicing the batch into K equal blocks.
   - Writes evaluator_friendly artifacts:
       gen_class_{k}.npy
       labels_class_{k}.npy
       x_synth.npy, y_synth.npy

Rule A (scalable, dataset_aware artifacts)
-----------------------------------------
All artifacts are dataset_aware and variant_aware:

  <paths.artifacts>/<data.name>/<model.family>/<model.variant>/
      checkpoints/
      summaries/
      synthetic/

The pipeline reads checkpoints from checkpoints/ and writes synthesis dumps to synthetic/.
(Preview images, if added later, should go to summaries/.)

Config compatibility
--------------------
Supports BOTH:
- NEW keys:
    data.name, data.root
    model.family, model.variant
    model.img_shape, model.num_classes
    model.num_flows, model.hidden_dims
    train.{epochs,batch_size,lr,clip_grad,patience,seed}
    synth.{samples_per_class,temperature}
    paths.artifacts
    artifacts.{checkpoints,summaries,synthetic}  (optional overrides)
- LEGACY keys:
    IMG_SHAPE, NUM_CLASSES, NUM_FLOWS, HIDDEN_DIMS
    EPOCHS, BATCH_SIZE, LR, CLIP_GRAD, PATIENCE, SEED
    SAMPLES_PER_CLASS, SAMPLE_TEMPERATURE
    ARTIFACTS: {checkpoints: ..., synthetic: ...}
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
import tensorflow as tf

from maskedautoflow.models import (
    MAF,
    MAFConfig,
    build_maf_model,
    flatten_images,
    reshape_to_images,
)

# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def _ensure_dir(p: Path) -> None:
    """mkdir -p."""
    p.mkdir(parents=True, exist_ok=True)


def _cfg_get(cfg: Dict[str, Any], dotted: str, default=None):
    """Read nested dict values using dot notation (e.g., 'train.epochs')."""
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _one_hot(ids: np.ndarray, num_classes: int) -> np.ndarray:
    """Minimal one_hot encoder for 1_D int labels."""
    ids = np.asarray(ids).reshape(-1).astype(np.int32)
    out = np.zeros((len(ids), int(num_classes)), dtype=np.float32)
    out[np.arange(len(ids)), ids] = 1.0
    return out


# -----------------------------------------------------------------------------
# Rule A: dataset_aware artifact paths
# -----------------------------------------------------------------------------
def _resolve_artifact_paths(cfg: Dict[str, Any]) -> tuple[Path, Path, Path]:
    """
    Resolve artifact directories following Rule A.

    Base:
      <paths.artifacts>/<data.name>/<model.family>/<model.variant>/

    Subdirs:
      checkpoints/  (read/write model weights)
      summaries/    (write logs/preview)
      synthetic/    (write evaluator_friendly .npy dumps)

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
# Pipeline config
# -----------------------------------------------------------------------------
@dataclass
class MAFPipelineConfig:
    # Data / shapes
    IMG_SHAPE: Tuple[int, int, int] = (40, 40, 1)
    NUM_CLASSES: int = 9

    # Model
    NUM_FLOWS: int = 5
    HIDDEN_DIMS: Tuple[int, ...] = (128, 128)

    # Training
    EPOCHS: int = 5
    BATCH_SIZE: int = 256
    LR: float = 2e_4
    CLIP_GRAD: float = 1.0
    PATIENCE: int = 10
    SEED: int = 42

    # Synthesis
    SAMPLES_PER_CLASS: int = 25
    SAMPLE_TEMPERATURE: float = 1.0  # τ=1.0 default; <1 sharper, >1 more diverse

    # Artifacts (Rule A)
    CKPT_DIR: Path = Path("artifacts/dataset/maskedautoflow/maf_affine/checkpoints")
    SUMS_DIR: Path = Path("artifacts/dataset/maskedautoflow/maf_affine/summaries")
    SYNTH_DIR: Path = Path("artifacts/dataset/maskedautoflow/maf_affine/synthetic")


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------
class MAFPipeline:
    """
    Train + synth wrapper for an unconditional Masked Autoregressive Flow (MAF).

    Notes
    -----
    - y_* inputs are accepted for API parity but unused (model is unconditional).
    - Synthesis assigns samples to classes by equal slicing for evaluator parity.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self._raw_cfg: Dict[str, Any] = cfg or {}

        # ---- Resolve artifacts first (Rule A) ----
        ckpt_dir, sums_dir, synth_dir = _resolve_artifact_paths(self._raw_cfg)
        _ensure_dir(ckpt_dir)
        _ensure_dir(sums_dir)
        _ensure_dir(synth_dir)

        # ---- Resolve knobs (NEW → LEGACY fallbacks) ----
        img_shape = tuple(
            _cfg_get(self._raw_cfg, "model.img_shape", _cfg_get(self._raw_cfg, "IMG_SHAPE", _cfg_get(self._raw_cfg, "img.shape", (40, 40, 1))))
        )
        num_classes = int(_cfg_get(self._raw_cfg, "model.num_classes", _cfg_get(self._raw_cfg, "NUM_CLASSES", _cfg_get(self._raw_cfg, "num_classes", 9))))
        num_flows = int(_cfg_get(self._raw_cfg, "model.num_flows", _cfg_get(self._raw_cfg, "NUM_FLOWS", 5)))
        hidden_dims_raw = _cfg_get(self._raw_cfg, "model.hidden_dims", _cfg_get(self._raw_cfg, "HIDDEN_DIMS", (128, 128)))
        hidden_dims = tuple(int(h) for h in hidden_dims_raw)

        epochs = int(_cfg_get(self._raw_cfg, "train.epochs", _cfg_get(self._raw_cfg, "EPOCHS", 5)))
        batch_size = int(_cfg_get(self._raw_cfg, "train.batch_size", _cfg_get(self._raw_cfg, "BATCH_SIZE", 256)))
        lr = float(_cfg_get(self._raw_cfg, "train.lr", _cfg_get(self._raw_cfg, "LR", 2e_4)))
        clip = float(_cfg_get(self._raw_cfg, "train.clip_grad", _cfg_get(self._raw_cfg, "CLIP_GRAD", 1.0)))
        patience = int(_cfg_get(self._raw_cfg, "train.patience", _cfg_get(self._raw_cfg, "PATIENCE", _cfg_get(self._raw_cfg, "patience", 10))))
        seed = int(_cfg_get(self._raw_cfg, "train.seed", _cfg_get(self._raw_cfg, "SEED", 42)))

        samples_per_class = int(_cfg_get(self._raw_cfg, "synth.samples_per_class", _cfg_get(self._raw_cfg, "SAMPLES_PER_CLASS", _cfg_get(self._raw_cfg, "samples_per_class", 25))))
        temperature = float(_cfg_get(self._raw_cfg, "synth.temperature", _cfg_get(self._raw_cfg, "SAMPLE_TEMPERATURE", 1.0)))

        # ---- Store typed config ----
        self.cfg = MAFPipelineConfig(
            IMG_SHAPE=img_shape,
            NUM_CLASSES=num_classes,
            NUM_FLOWS=num_flows,
            HIDDEN_DIMS=hidden_dims,
            EPOCHS=epochs,
            BATCH_SIZE=batch_size,
            LR=lr,
            CLIP_GRAD=clip,
            PATIENCE=patience,
            SEED=seed,
            SAMPLES_PER_CLASS=samples_per_class,
            SAMPLE_TEMPERATURE=temperature,
            CKPT_DIR=ckpt_dir,
            SUMS_DIR=sums_dir,
            SYNTH_DIR=synth_dir,
        )

        # Optional external logger callback: cb(stage: str, message: str)
        self.log_cb = self._raw_cfg.get("LOG_CB", None)

        # Model handle + bookkeeping
        self.model: Optional[MAF] = None
        self.best_ckpt_path: Optional[Path] = None

        # Reproducibility: seed once (don’t reseed inside tight loops)
        tf.keras.utils.set_random_seed(self.cfg.SEED)
        np.random.seed(self.cfg.SEED)

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    def _log(self, stage: str, msg: str) -> None:
        """Route logs to optional callback or stdout."""
        if self.log_cb is not None:
            try:
                self.log_cb(stage, msg)
                return
            except Exception:
                pass
        print(f"[{stage}] {msg}")

    # -------------------------------------------------------------------------
    # Data pipeline
    # -------------------------------------------------------------------------
    def _make_dataset(self, x_flat: Optional[np.ndarray], *, shuffle: bool) -> Optional[tf.data.Dataset]:
        """
        Wrap (N, D) float32 arrays into a tf.data pipeline.
        Returns None if x_flat is None.
        """
        if x_flat is None:
            return None

        x = np.asarray(x_flat, dtype=np.float32, order="C")
        ds = tf.data.Dataset.from_tensor_slices((x,))
        if shuffle:
            ds = ds.shuffle(
                buffer_size=min(len(x), 10_000),
                seed=self.cfg.SEED,
                reshuffle_each_iteration=True,
            )
        return ds.batch(self.cfg.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    def train(
        self,
        x_train: np.ndarray,
        y_train: Optional[np.ndarray] = None,  # unused (unconditional)
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,    # unused (unconditional)
        *,
        save_checkpoints: bool = True,
    ) -> Dict[str, Any]:
        """
        Fit a global MAF on images in [0,1].

        Parameters
        ----------
        x_train : (N,H,W,C) or (N,D)
        x_val   : optional validation set (same format)
        save_checkpoints : if True, write best/last checkpoints under Rule A ckpt_dir.

        Returns
        -------
        bundle : dict
          Minimal training bundle (useful for orchestrators):
            {"input_dim": D, "best_ckpt": "...", "ckpt_dir": "..."}
        """
        H, W, C = self.cfg.IMG_SHAPE
        D = H * W * C

        # --- Flatten & sanitize inputs to (N, D) float32 in [0,1] ---
        Xtr = flatten_images(x_train, img_shape=(H, W, C), assume_01=True, clip=True)
        Xva = flatten_images(x_val,   img_shape=(H, W, C), assume_01=True, clip=True) if x_val is not None else None

        train_ds = self._make_dataset(Xtr, shuffle=True)
        val_ds = self._make_dataset(Xva, shuffle=False)

        # --- Build model and variables (Keras requires first call) ---
        self.model = build_maf_model(
            MAFConfig(
                IMG_SHAPE=(H, W, C),
                NUM_FLOWS=self.cfg.NUM_FLOWS,
                HIDDEN_DIMS=self.cfg.HIDDEN_DIMS,
            )
        )
        _ = self.model(tf.zeros((1, D), dtype=tf.float32))

        opt = tf.keras.optimizers.Adam(learning_rate=self.cfg.LR)

        best_val = float("inf")
        bad_epochs = 0

        self._log(
            "train",
            f"MAF train | HWC={(H,W,C)} D={D} flows={self.cfg.NUM_FLOWS} hidden={self.cfg.HIDDEN_DIMS} "
            f"epochs={self.cfg.EPOCHS} bs={self.cfg.BATCH_SIZE} lr={self.cfg.LR} "
            f"ckpt_dir={self.cfg.CKPT_DIR}"
        )

        for epoch in range(1, self.cfg.EPOCHS + 1):
            # ---- Train epoch ----
            m_train = tf.keras.metrics.Mean()
            for (xb,) in train_ds:  # type: ignore[misc]
                with tf.GradientTape() as tape:
                    nll = -tf.reduce_mean(self.model.log_prob(xb))
                grads = tape.gradient(nll, self.model.trainable_variables)
                grads, _ = tf.clip_by_global_norm(grads, self.cfg.CLIP_GRAD)
                opt.apply_gradients(zip(grads, self.model.trainable_variables))
                m_train.update_state(nll)

            train_nll = float(m_train.result().numpy())

            # ---- Validate epoch (if provided) ----
            val_nll = None
            if val_ds is not None:
                m_val = tf.keras.metrics.Mean()
                for (xb,) in val_ds:  # type: ignore[misc]
                    m_val.update_state(-tf.reduce_mean(self.model.log_prob(xb)))
                val_nll = float(m_val.result().numpy())

            # ---- Log ----
            if val_nll is None:
                self._log("train", f"epoch {epoch:03d}: train_nll={train_nll:.4f}")
            else:
                self._log("train", f"epoch {epoch:03d}: train_nll={train_nll:.4f} | val_nll={val_nll:.4f}")

            # ---- Save last each epoch (helps debugging/resume) ----
            if save_checkpoints:
                last_path = self.cfg.CKPT_DIR / "MAF_last.weights.h5"
                self.model.save_weights(str(last_path), overwrite=True)

            # ---- Best checkpoint selection ----
            # If we have validation: use val_nll.
            # If no validation: treat the final epoch as "best" (deterministic).
            improved = False
            if val_nll is None:
                improved = (epoch == self.cfg.EPOCHS)
            else:
                improved = (val_nll < best_val - 1e_6)

            if improved:
                bad_epochs = 0
                if val_nll is not None:
                    best_val = val_nll

                if save_checkpoints:
                    best_path = self.cfg.CKPT_DIR / "MAF_best.weights.h5"
                    self.model.save_weights(str(best_path), overwrite=True)
                    self.best_ckpt_path = best_path
                    self._log("ckpt", f"saved {best_path.name}")
            else:
                # Early stopping is only meaningful when we have validation
                if val_nll is not None:
                    bad_epochs += 1
                    if bad_epochs >= self.cfg.PATIENCE:
                        self._log("train", f"early stop: no val improvement for {self.cfg.PATIENCE} epochs.")
                        break

        # Marker for orchestrators
        if save_checkpoints:
            (self.cfg.CKPT_DIR / "MAF_LAST_OK").write_text("ok", encoding="utf_8")

        return {
            "input_dim": D,
            "best_ckpt": str(self.best_ckpt_path) if self.best_ckpt_path else None,
            "ckpt_dir": str(self.cfg.CKPT_DIR),
        }

    # Back_compat: some runners call fit()
    def fit(self, *args, **kwargs):
        """Alias for `train`."""
        return self.train(*args, **kwargs)

    # -------------------------------------------------------------------------
    # Checkpoint loading (for synthesis)
    # -------------------------------------------------------------------------
    def _load_checkpoint(self, D: int) -> MAF:
        """
        Build the model (if needed) and load weights for synthesis.

        Load order:
          1) MAF_best.weights.h5
          2) MAF_last.weights.h5
        """
        if self.model is None:
            self.model = build_maf_model(
                MAFConfig(
                    IMG_SHAPE=self.cfg.IMG_SHAPE,
                    NUM_FLOWS=self.cfg.NUM_FLOWS,
                    HIDDEN_DIMS=self.cfg.HIDDEN_DIMS,
                )
            )
            _ = self.model(tf.zeros((1, D), dtype=tf.float32))

        best = self.cfg.CKPT_DIR / "MAF_best.weights.h5"
        last = self.cfg.CKPT_DIR / "MAF_last.weights.h5"
        to_load = best if best.exists() else last

        if not to_load.exists():
            raise FileNotFoundError(f"No MAF checkpoint found in: {self.cfg.CKPT_DIR}")

        self.model.load_weights(str(to_load))
        self._log("ckpt", f"loaded {to_load.name}")
        return self.model

    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------
    def _sample_latent(self, n: int, D: int, temperature: float) -> tf.Tensor:
        """
        Sample base noise z ~ N(0, I) and scale by temperature τ.

        τ < 1.0 → lower variance (often sharper / less diverse)
        τ > 1.0 → higher variance (often more diverse / noisier)
        """
        z = tf.random.normal(shape=(int(n), int(D)), dtype=tf.float32)
        return z * float(temperature)

    def _sample_unconditional(self, n_total: int) -> np.ndarray:
        """Sample `n_total` unconditional images, returning (N,H,W,C) in [0,1]."""
        H, W, C = self.cfg.IMG_SHAPE
        D = H * W * C

        model = self._load_checkpoint(D)
        z = self._sample_latent(n_total, D, temperature=self.cfg.SAMPLE_TEMPERATURE)

        x_flat = model.inverse(z).numpy().astype(np.float32, copy=False)
        x_flat = np.clip(x_flat, 0.0, 1.0)

        return reshape_to_images(x_flat, (H, W, C), clip=True)

    # -------------------------------------------------------------------------
    # Artifact emission (evaluator dumps)
    # -------------------------------------------------------------------------
    def _emit_per_class_files(self, x: np.ndarray, *, per_class: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split unconditional samples into K equal blocks and write evaluator files.

        Files written to Rule A synthetic/:
          gen_class_{k}.npy
          labels_class_{k}.npy
          x_synth.npy
          y_synth.npy
        """
        _ensure_dir(self.cfg.SYNTH_DIR)

        K = int(self.cfg.NUM_CLASSES)
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []

        for k in range(K):
            start, end = k * int(per_class), (k + 1) * int(per_class)
            xk = x[start:end]

            np.save(self.cfg.SYNTH_DIR / f"gen_class_{k}.npy", xk)
            np.save(self.cfg.SYNTH_DIR / f"labels_class_{k}.npy", np.full((len(xk),), k, dtype=np.int32))

            xs.append(xk)
            ys.append(_one_hot(np.full((len(xk),), k, dtype=np.int32), K))

        x_synth = np.concatenate(xs, axis=0).astype(np.float32, copy=False)
        y_synth = np.concatenate(ys, axis=0).astype(np.float32, copy=False)

        np.save(self.cfg.SYNTH_DIR / "x_synth.npy", x_synth)
        np.save(self.cfg.SYNTH_DIR / "y_synth.npy", y_synth)

        return x_synth, y_synth

    # -------------------------------------------------------------------------
    # Public synthesize
    # -------------------------------------------------------------------------
    def synthesize(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a class_balanced synthetic set and write evaluator dumps.

        Returns
        -------
        x_synth : (K*S, H, W, C) float32 in [0,1]
        y_synth : (K*S, K) one_hot
        """
        K = int(self.cfg.NUM_CLASSES)
        per_class = int(self.cfg.SAMPLES_PER_CLASS)
        n_total = K * per_class

        self._log(
            "synthesize",
            f"MAF synth | K={K} per_class={per_class} total={n_total} "
            f"temperature={self.cfg.SAMPLE_TEMPERATURE} -> {self.cfg.SYNTH_DIR}",
        )

        x = self._sample_unconditional(n_total)

        # Safety: drop any non_finite rows (rare, but protects downstream metrics)
        finite = np.isfinite(x).all(axis=(1, 2, 3))
        if not finite.all():
            dropped = int((~finite).sum())
            self._log("warn", f"dropping {dropped} non_finite synthetic samples")
            x = x[finite]

            # Best_effort top_up to keep exact count
            deficit = n_total - int(len(x))
            if deficit > 0:
                x_more = self._sample_unconditional(deficit)
                x = np.concatenate([x, x_more], axis=0)

        x_s, y_s = self._emit_per_class_files(x[:n_total], per_class=per_class)
        self._log("synthesize", f"wrote {x_s.shape[0]} samples -> {self.cfg.SYNTH_DIR}")
        return x_s, y_s

    # -------------------------------------------------------------------------
    # Preview helper (does not write to disk)
    # -------------------------------------------------------------------------
    def _sample_batch(self, *_, n_per_class: int = 1, **__) -> tuple[np.ndarray, np.ndarray]:
        """
        Quick sampling helper for visualization.

        Accepts **kwargs so callers passing extra keys (e.g. bundle=...) won't break.
        Returns (x, y) where y is a dummy one_hot label array for display parity.
        """
        K = int(self.cfg.NUM_CLASSES)
        n_total = max(1, int(n_per_class)) * K

        x = self._sample_unconditional(n_total)
        ids = np.repeat(np.arange(K, dtype=np.int32), max(1, int(n_per_class)))
        y = _one_hot(ids, K)

        return x[: len(ids)], y


# Back_compat alias
Pipeline = MAFPipeline

__all__ = ["MAFPipeline",
           "Pipeline",
           "MAFPipelineConfig"]
