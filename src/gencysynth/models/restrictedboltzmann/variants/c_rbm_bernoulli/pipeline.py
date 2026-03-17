# src/gencysynth/models/restrictedboltzmann/variants/c_rbm_bernoulli/pipeline.py
"""
Training + synthesis pipeline for the Restricted Boltzmann Machine (RBM) baseline.

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
      rbm_train_preview.png (optional)
      rbm_synth_preview.png (optional)

Where <dataset_id> is resolved from config (same logic as in sample.py):
  1) cfg['data']['id']
  2) cfg['dataset']['id']
  3) basename(cfg['data']['root'] or cfg['DATA_DIR'])
  4) "default_dataset"

Overrides (optional)
--------------------
You may override specific artifact dirs via:
  cfg['artifacts']['restrictedboltzmann']['c_rbm_bernoulli']['checkpoints'|'synthetic'|'summaries']

This module is designed to be called by the unified orchestrator (app/main.py),
and to produce evaluator_friendly .npy outputs (not just PNGs).

Notes
-----
- RBM is trained per_class (one RBM per label).
- Inputs are expected in [0,1] (0..255 allowed; will be normalized).
- If a class has too few samples, we emit a stub marker instead of crashing,
  so the run is reproducible and the missingness is explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

# IMPORTANT: This pipeline lives under:
#   src/gencysynth/models/restrictedboltzmann/variants/c_rbm_bernoulli/
# so we prefer relative imports within the variant package.
from .models import (
    RBMConfig,
    RBM,
    build_rbm,
    to_float01,
    binarize01,
    sample_gibbs,
)

# Reuse Rule A path resolution from the sibling sample.py implementation.
# This avoids duplicating dataset_id parsing logic and keeps paths consistent
# across train/synth/preview.
from .sample import _resolve_artifact_paths, save_grid_from_checkpoints  # type: ignore


# =============================================================================
# Small helpers
# =============================================================================
def _ensure_dir(p: Path) -> None:
    """Create directory (parents included) if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)


def _labels_to_int(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Accept one_hot (N,K) or integer (N,) labels and return int labels (N,).
    """
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == num_classes:
        return np.argmax(y, axis=1).astype(np.int64)
    if y.ndim == 1:
        return y.astype(np.int64)
    raise ValueError(f"Labels must be (N,) ints or (N,{num_classes}) one_hot; got {y.shape}")


def _is_built(rbm: RBM) -> bool:
    """
    Keras models are "built" once variables exist. For RBM we treat presence
    of W as the indicator.
    """
    try:
        _ = rbm.W.shape  # type: ignore[attr_defined]
        return True
    except Exception:
        return False


def _one_hot(ids: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer labels -> one_hot float32."""
    ids = np.asarray(ids, dtype=np.int32).ravel()
    out = np.zeros((len(ids), int(num_classes)), dtype=np.float32)
    out[np.arange(len(ids)), ids] = 1.0
    return out


# =============================================================================
# Pipeline config
# =============================================================================
@dataclass
class RBMPipelineConfig:
    # Data / shapes
    IMG_SHAPE: Tuple[int, int, int] = (40, 40, 1)
    NUM_CLASSES: int = 9

    # RBM hyperparameters (per class)
    HIDDEN_UNITS: int = 256
    CD_K: int = 1
    LR: float = 1e_3
    WEIGHT_DECAY: float = 0.0
    TRAIN_MODE: str = "cd"  # {"cd", "mse"}

    # Optimization (MSE mode only; CD mode uses manual SGD_style updates)
    BATCH_SIZE: int = 256
    EPOCHS: int = 10
    PATIENCE: int = 10
    SAVE_EVERY: int = 10  # periodic epoch snapshots (CD and MSE)

    # Data preprocessing
    BINARIZE: bool = True
    BIN_THRESHOLD: float = 0.5
    VAL_SPLIT: float = 0.1  # used if val set not provided

    # Synthesis
    SAMPLES_PER_CLASS: int = 1000
    SAMPLE_K_STEPS: Optional[int] = None  # defaults to CD_K if None
    SAMPLE_BURN_IN: int = 0

    # Reproducibility
    SEED: Optional[int] = 42

    # Artifact override (Rule A resolver also supports nested overrides)
    ARTIFACTS: Dict[str, Any] = None


# =============================================================================
# Pipeline
# =============================================================================
class RBMPipeline:
    """
    Orchestrates training and synthesis for class_conditional RBMs.

    Responsibilities
    ----------------
    • Train one RBM per class and save checkpoints (BEST/LAST/periodic).
    • Synthesize class_balanced samples and write evaluator_friendly .npy files.
    • Keep all artifacts dataset_scoped under Rule A.
    """

    def __init__(self, cfg: Dict[str, Any]):
        """
        Parameters
        ----------
        cfg : dict
            Unified config dict passed from app/main. Should include dataset info
            (data.id or data.root) and optionally artifact overrides.
        """
        # ---- Map loose dict -> strongly typed config with defaults ----
        self.cfg = RBMPipelineConfig(
            IMG_SHAPE=tuple(cfg.get("IMG_SHAPE", (40, 40, 1))),
            NUM_CLASSES=int(cfg.get("NUM_CLASSES", cfg.get("num_classes", 9))),
            HIDDEN_UNITS=int(cfg.get("HIDDEN_UNITS", cfg.get("RBM_HIDDEN", 256))),
            CD_K=int(cfg.get("CD_K", 1)),
            LR=float(cfg.get("LR", cfg.get("RBM_LR", 1e_3))),
            WEIGHT_DECAY=float(cfg.get("WEIGHT_DECAY", 0.0)),
            TRAIN_MODE=str(cfg.get("TRAIN_MODE", "cd")),
            BATCH_SIZE=int(cfg.get("BATCH_SIZE", cfg.get("RBM_BATCH", 256))),
            EPOCHS=int(cfg.get("EPOCHS", cfg.get("RBM_EPOCHS", 10))),
            PATIENCE=int(cfg.get("PATIENCE", cfg.get("patience", 10))),
            SAVE_EVERY=int(cfg.get("SAVE_EVERY", 10)),
            BINARIZE=bool(cfg.get("BINARIZE", True)),
            BIN_THRESHOLD=float(cfg.get("BIN_THRESHOLD", 0.5)),
            VAL_SPLIT=float(cfg.get("VAL_SPLIT", 0.1)),
            SAMPLES_PER_CLASS=int(cfg.get("SAMPLES_PER_CLASS", cfg.get("samples_per_class", 1000))),
            SAMPLE_K_STEPS=cfg.get("SAMPLE_K_STEPS", None),
            SAMPLE_BURN_IN=int(cfg.get("SAMPLE_BURN_IN", 0)),
            SEED=cfg.get("SEED", cfg.get("SEED", 42)),
            ARTIFACTS=cfg.get("ARTIFACTS", cfg.get("artifacts", {})),
        )

        # ---- Rule A artifacts ----
        # We delegate dataset_id resolution and override handling to sample.py
        # so train and synth always agree on paths.
        self._raw_cfg = dict(cfg)  # keep original config for path resolver
        self.paths = _resolve_artifact_paths(self._raw_cfg)

        _ensure_dir(self.paths.checkpoints)
        _ensure_dir(self.paths.synthetic)
        _ensure_dir(self.paths.summaries)

        # Optional external logger callback: cb(stage: str, message: str)
        self.log_cb = cfg.get("LOG_CB", None)

        # In_memory RBM handles (one per class). We still load from disk for synth.
        self.models: List[Optional[RBM]] = [None] * int(self.cfg.NUM_CLASSES)

        # Best_effort reproducibility
        if self.cfg.SEED is not None:
            np.random.seed(int(self.cfg.SEED))
            tf.random.set_seed(int(self.cfg.SEED))

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log(self, stage: str, msg: str) -> None:
        """Route logs to optional callback or stdout."""
        if self.log_cb:
            try:
                self.log_cb(stage, msg)
                return
            except Exception:
                pass
        print(f"[{stage}] {msg}")

    # ------------------------------------------------------------------
    # tf.data helpers
    # ------------------------------------------------------------------
    def _make_dataset(self, x_flat: np.ndarray, batch_size: int, shuffle: bool) -> tf.data.Dataset:
        """
        Create a dataset of visible vectors only: yields v0 batches (float32).
        """
        ds = tf.data.Dataset.from_tensor_slices(x_flat.astype("float32", copy=False))
        if shuffle:
            ds = ds.shuffle(buffer_size=min(len(x_flat), 8192), reshuffle_each_iteration=True)
        ds = ds.batch(int(batch_size), drop_remainder=False).prefetch(tf.data.AUTOTUNE)
        return ds

    # ------------------------------------------------------------------
    # Checkpoint paths (Rule A)
    # ------------------------------------------------------------------
    def _class_ckpt_dir(self, class_id: int) -> Path:
        """
        Per_class checkpoint folder under Rule A:
          .../checkpoints/class_<k>/
        """
        return self.paths.checkpoints / f"class_{int(class_id)}"

    def _best_ckpt_path(self, class_id: int) -> Path:
        return self._class_ckpt_dir(class_id) / "RBM_best.weights.h5"

    def _last_ckpt_path(self, class_id: int) -> Path:
        return self._class_ckpt_dir(class_id) / "RBM_last.weights.h5"

    def _epoch_ckpt_path(self, class_id: int, epoch: int) -> Path:
        return self._class_ckpt_dir(class_id) / f"RBM_epoch_{int(epoch):04d}.weights.h5"

    # ------------------------------------------------------------------
    # Training (per class)
    # ------------------------------------------------------------------
    def _train_single_class(
        self,
        Xk: np.ndarray,
        *,
        class_id: int,
        x_val: Optional[np.ndarray] = None,
    ) -> Optional[RBM]:
        """
        Train one RBM on a single class.

        Inputs
        ------
        Xk : (N,V) float32, values in {0,1} if BINARIZE else [0,1]
        x_val : optional validation set (same format)

        Outputs
        -------
        RBM instance (loaded with best weights) OR None if too few samples.
        """
        V = int(np.prod(self.cfg.IMG_SHAPE))
        H = int(self.cfg.HIDDEN_UNITS)

        # If a class is too small, emit stub markers (explicit missingness),
        # and return None. This keeps the run stable and reproducible.
        if Xk is None or len(Xk) < 2:
            class_dir = self._class_ckpt_dir(class_id)
            _ensure_dir(class_dir)
            (class_dir / f"RBM_STUB_CLASS_{int(class_id)}.txt").write_text(
                f"Too few samples for class {class_id}; n={0 if Xk is None else len(Xk)}\n",
                encoding="utf_8",
            )
            # Touch markers so downstream code can detect existence.
            self._best_ckpt_path(class_id).touch()
            self._last_ckpt_path(class_id).touch()
            self._log("warn", f"[class {class_id}] too few samples (n={0 if Xk is None else len(Xk)}); wrote stub markers.")
            return None

        # ---- Build RBM ----
        rbm_cfg = RBMConfig(
            visible_units=V,
            hidden_units=H,
            cd_k=int(self.cfg.CD_K),
            learning_rate=float(self.cfg.LR),
            weight_decay=float(self.cfg.WEIGHT_DECAY),
            train_mode=str(self.cfg.TRAIN_MODE),
            seed=int(self.cfg.SEED) if self.cfg.SEED is not None else None,
        )
        rbm = build_rbm(rbm_cfg)

        # ---- Optional train/val split if no explicit val ----
        if x_val is None and float(self.cfg.VAL_SPLIT) > 0.0 and len(Xk) > 1:
            n_val = max(1, int(len(Xk) * float(self.cfg.VAL_SPLIT)))
            x_val = Xk[:n_val]
            Xk = Xk[n_val:]

        train_ds = self._make_dataset(Xk, batch_size=self.cfg.BATCH_SIZE, shuffle=True)
        val_ds = self._make_dataset(x_val, batch_size=self.cfg.BATCH_SIZE, shuffle=False) if x_val is not None else None

        # ---- Checkpoint dirs ----
        class_dir = self._class_ckpt_dir(class_id)
        _ensure_dir(class_dir)

        best_path = self._best_ckpt_path(class_id)
        last_path = self._last_ckpt_path(class_id)

        patience = int(self.cfg.PATIENCE)
        best_val = float("inf")
        best_epoch = 0
        wait = 0

        self._log(
            "train",
            f"[class {class_id}] RBM(H={H}) mode={self.cfg.TRAIN_MODE} cd_k={self.cfg.CD_K} "
            f"lr={self.cfg.LR:g} wd={self.cfg.WEIGHT_DECAY:g} "
            f"epochs={self.cfg.EPOCHS} bs={self.cfg.BATCH_SIZE} -> {class_dir}",
        )

        # Optimizer used only in "mse" mode (CD mode does manual updates)
        optimizer = None
        if str(self.cfg.TRAIN_MODE).lower() == "mse":
            optimizer = tf.keras.optimizers.Adam(learning_rate=float(self.cfg.LR))

        # ---- Epoch loop ----
        for epoch in range(1, int(self.cfg.EPOCHS) + 1):
            # ---- Train ----
            losses: List[float] = []
            for v0 in train_ds:
                if str(self.cfg.TRAIN_MODE).lower() == "cd":
                    loss = rbm.train_step_cd(
                        v0,
                        k=int(self.cfg.CD_K),
                        lr=float(self.cfg.LR),
                        weight_decay=float(self.cfg.WEIGHT_DECAY),
                    )
                else:
                    # mse: use optimizer
                    loss = rbm.train_step_mse(v0, optimizer=optimizer, k=int(self.cfg.CD_K))  # type: ignore[arg_type]
                losses.append(float(loss))

            train_loss = float(np.mean(losses)) if losses else float("nan")

            # ---- Validate ----
            if val_ds is not None:
                vlosses: List[float] = []
                for v in val_ds:
                    v_prob = rbm(v, training=False)
                    vloss = tf.reduce_mean(tf.square(v - v_prob))
                    vlosses.append(float(vloss))
                val_loss = float(np.mean(vlosses)) if vlosses else float("nan")
            else:
                # If no val set provided, use train loss as best_effort monitor.
                val_loss = train_loss

            self._log("train", f"[class {class_id}] epoch {epoch:04d}: train={train_loss:.5f} | val={val_loss:.5f}")

            # ---- Periodic snapshot ----
            if epoch == 1 or (epoch % max(1, int(self.cfg.SAVE_EVERY)) == 0):
                try:
                    rbm.save_weights(str(self._epoch_ckpt_path(class_id, epoch)))
                except Exception as e:
                    self._log("warn", f"[class {class_id}] failed to save epoch snapshot: {e}")

            # ---- Early stopping on val_loss ----
            if np.isfinite(val_loss) and (val_loss < best_val - 1e_6):
                best_val = val_loss
                best_epoch = epoch
                wait = 0
                rbm.save_weights(str(best_path))
            else:
                wait += 1
                if wait >= patience:
                    self._log("train", f"[class {class_id}] early stop at epoch {epoch} (best={best_val:.5f} @ {best_epoch}).")
                    break

        # ---- Always write LAST ----
        rbm.save_weights(str(last_path))

        # ---- Reload BEST if available ----
        if best_path.exists() and best_path.stat().st_size > 0:
            try:
                rbm.load_weights(str(best_path))
                self._log("ckpt", f"[class {class_id}] saved best={best_path.name}, last={last_path.name}")
            except Exception:
                # If best is a stub/invalid file, keep last.
                rbm.load_weights(str(last_path))
                self._log("ckpt", f"[class {class_id}] best invalid; using last={last_path.name}")
        else:
            # No "best" ever written -> keep last.
            self._log("ckpt", f"[class {class_id}] no best checkpoint; using last={last_path.name}")

        return rbm

    # ------------------------------------------------------------------
    # Public training API
    # ------------------------------------------------------------------
    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> List[RBM]:
        """
        Fit one RBM per class.

        Inputs
        ------
        x_train : (N,H,W,C) or (N,V) float in [0,1] (0..255 allowed; normalized)
        y_train : (N,) ints or (N,K) one_hot
        x_val/y_val : optional explicit validation data (used per_class)

        Outputs
        -------
        list of fitted RBM instances (may be fewer than NUM_CLASSES if some classes are empty)
        """
        H, W, C = self.cfg.IMG_SHAPE
        K = int(self.cfg.NUM_CLASSES)
        V = int(H) * int(W) * int(C)

        # ---- Prepare training data (normalize + optional binarize) ----
        x = to_float01(np.asarray(x_train).reshape((-1, int(H), int(W), int(C))))
        if bool(self.cfg.BINARIZE):
            x = binarize01(x, thresh=float(self.cfg.BIN_THRESHOLD))
        X = x.reshape((-1, V)).astype("float32", copy=False)

        y_ids = _labels_to_int(y_train, K)

        # ---- Optional validation ----
        Xv: Optional[np.ndarray] = None
        yv_ids: Optional[np.ndarray] = None
        if x_val is not None:
            xv = to_float01(np.asarray(x_val).reshape((-1, int(H), int(W), int(C))))
            if bool(self.cfg.BINARIZE):
                xv = binarize01(xv, thresh=float(self.cfg.BIN_THRESHOLD))
            Xv = xv.reshape((-1, V)).astype("float32", copy=False)
            yv_ids = _labels_to_int(y_val, K) if y_val is not None else None

        self._log(
            "train",
            f"Training RBMs (per_class) on {len(X)} samples | dim={V} | classes={K} "
            f"| artifacts={self.paths.root}",
        )

        # ---- Train each class ----
        for k in range(K):
            idx = (y_ids == k)
            Xk = X[idx]

            if Xk.size == 0:
                self._log("warn", f"[class {k}] no training samples; skipping.")
                self.models[k] = None
                continue

            # Optional per_class val set if explicit val provided
            Xk_val = None
            if Xv is not None and yv_ids is not None:
                Xk_val = Xv[yv_ids == k]
                if Xk_val.size == 0:
                    Xk_val = None

            rbm_k = self._train_single_class(Xk, class_id=k, x_val=Xk_val)
            self.models[k] = rbm_k

        # Marker for downstream orchestration/debugging
        (self.paths.checkpoints / "RBM_LAST_OK").write_text("ok", encoding="utf_8")

        # Optional: save a small training preview grid (best_effort)
        try:
            out = self.paths.summaries / "rbm_train_preview.png"
            save_grid_from_checkpoints(
                ckpt_root=self.paths.checkpoints,
                img_shape=(int(H), int(W), int(C)),
                num_classes=K,
                path=out,
                per_class=1,
                hidden_dim=int(self.cfg.HIDDEN_UNITS),
                gibbs_k=int(self.cfg.CD_K),
                burn_in=0,
                seed=int(self.cfg.SEED or 42),
            )
            self._log("summaries", f"train preview -> {out}")
        except Exception as e:
            self._log("warn", f"train preview grid failed: {e}")

        # Return all non_None models
        return [m for m in self.models if m is not None]

    # Backwards_friendly alias (some runners may call .fit)
    def fit(self, *args, **kwargs):
        return self.train(*args, **kwargs)

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------
    def _load_model_for_class(self, k: int) -> Optional[RBM]:
        """
        Load a class RBM from disk if not already in memory.

        We prefer BEST -> LAST -> latest EPOCH snapshot (Rule A layout).
        """
        if 0 <= k < len(self.models) and self.models[k] is not None and _is_built(self.models[k]):  # type: ignore[arg_type]
            return self.models[k]

        V = int(np.prod(self.cfg.IMG_SHAPE))
        rbm = build_rbm(
            RBMConfig(
                visible_units=V,
                hidden_units=int(self.cfg.HIDDEN_UNITS),
                cd_k=int(self.cfg.CD_K),
                learning_rate=float(self.cfg.LR),
                weight_decay=float(self.cfg.WEIGHT_DECAY),
                train_mode=str(self.cfg.TRAIN_MODE),
                seed=int(self.cfg.SEED) if self.cfg.SEED is not None else None,
            )
        )

        class_dir = self._class_ckpt_dir(k)
        best = self._best_ckpt_path(k)
        last = self._last_ckpt_path(k)
        epochs = sorted(class_dir.glob("RBM_epoch_*.weights.h5")) if class_dir.exists() else []

        # Select checkpoint in preferred order
        cand = best if (best.exists() and best.stat().st_size > 0) else None
        if cand is None and (last.exists() and last.stat().st_size > 0):
            cand = last
        if cand is None and epochs:
            cand = epochs[-1]

        if cand is None:
            return None

        try:
            rbm.load_weights(str(cand))
        except Exception:
            # If a stub/empty file exists, treat as missing.
            return None

        self.models[k] = rbm
        return rbm

    # ------------------------------------------------------------------
    # Synthesis (Rule A)
    # ------------------------------------------------------------------
    def synthesize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a class_balanced synthetic dataset and write evaluator files.

        Writes
        ------
        {paths.synthetic}/
          gen_class_<k>.npy
          labels_class_<k>.npy
          x_synth.npy
          y_synth.npy

        Returns
        -------
        x_synth : float32, shape (N_total, H, W, C), values in {0,1} (or [0,1] if RBM returns probs)
        y_synth : float32, shape (N_total, K), one_hot labels
        """
        H, W, C = self.cfg.IMG_SHAPE
        K = int(self.cfg.NUM_CLASSES)
        per_class = int(self.cfg.SAMPLES_PER_CLASS)
        k_steps = int(self.cfg.SAMPLE_K_STEPS or self.cfg.CD_K)
        burn_in = int(self.cfg.SAMPLE_BURN_IN)

        _ensure_dir(self.paths.synthetic)

        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []

        for k in range(K):
            rbm = self._load_model_for_class(k)
            if rbm is None:
                raise FileNotFoundError(
                    f"No RBM checkpoint for class {k} found under {self.paths.checkpoints}. "
                    f"Train first."
                )

            # Draw samples via Gibbs
            imgs = sample_gibbs(
                rbm,
                num_samples=int(per_class),
                k=int(k_steps),
                init=None,
                img_shape=(int(H), int(W), int(C)),
                binarize_init=False,
                burn_in=int(burn_in),
                seed=int(self.cfg.SEED or 42) + int(k),
            ).astype("float32", copy=False)

            # Per_class dumps (evaluator contract used across families)
            np.save(self.paths.synthetic / f"gen_class_{k}.npy", imgs)
            labels = np.full((imgs.shape[0],), int(k), dtype=np.int32)
            np.save(self.paths.synthetic / f"labels_class_{k}.npy", labels)

            xs.append(imgs)
            ys.append(_one_hot(labels, K))

        x_synth = np.concatenate(xs, axis=0).astype(np.float32, copy=False)
        y_synth = np.concatenate(ys, axis=0).astype(np.float32, copy=False)

        # Sanity: drop any non_finite rows (extremely unlikely)
        finite = np.isfinite(x_synth).all(axis=(1, 2, 3))
        if not finite.all():
            dropped = int((~finite).sum())
            self._log("warn", f"Dropping {dropped} non_finite synthetic samples.")
            x_synth = x_synth[finite]
            y_synth = y_synth[finite]

        # Combined convenience dumps
        np.save(self.paths.synthetic / "x_synth.npy", x_synth)
        np.save(self.paths.synthetic / "y_synth.npy", y_synth)

        # Optional: synth preview grid (best_effort)
        try:
            out = self.paths.summaries / "rbm_synth_preview.png"
            save_grid_from_checkpoints(
                ckpt_root=self.paths.checkpoints,
                img_shape=(int(H), int(W), int(C)),
                num_classes=K,
                path=out,
                per_class=1,
                hidden_dim=int(self.cfg.HIDDEN_UNITS),
                gibbs_k=int(k_steps),
                burn_in=int(burn_in),
                seed=int(self.cfg.SEED or 42),
            )
            self._log("summaries", f"synth preview -> {out}")
        except Exception as e:
            self._log("warn", f"synth preview grid failed: {e}")

        self._log(
            "synthesize",
            f"{x_synth.shape[0]} samples ({per_class} per class, gibbs_k={k_steps}, burn_in={burn_in}) -> {self.paths.synthetic}",
        )
        return x_synth, y_synth


__all__ = ["RBMPipeline",
           "RBMPipelineConfig"]