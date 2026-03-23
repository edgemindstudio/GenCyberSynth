# src/gencysynth/models/gaussianmixture/variants/c_gmm_full/pipeline.py
"""
GenCyberSynth — GaussianMixture — c_gmm_full — Train + Synthesize Pipeline
========================================================================

This pipeline mirrors the **GAN pipeline contract** and follows **Rule A** for
artifact layout so we can scale across many datasets and runs.

Rule A (Artifacts Contract)
---------------------------
All outputs are scoped by:

  (dataset_id, model_tag, run_id)

and MUST live under:

  artifacts/runs/<dataset_id>/<model_tag>/<run_id>/
    checkpoints/                 # training outputs (joblib)
    synthetic/                   # evaluator_friendly .npy dumps
    samples/                     # optional previews (png)
    manifest.json                # run manifest (written by orchestrator / sampler)

This pipeline is responsible for:
- Training class_conditional GMMs (one per class), with optional global fallback.
- Optional PCA front_end (fit once, train GMMs in PCA space).
- Writing checkpoints into ctx.run_dir/checkpoints.
- Writing synthetic .npy dumps into ctx.run_dir/synthetic using the evaluator contract:
    gen_class_{k}.npy
    labels_class_{k}.npy
    x_synth.npy, y_synth.npy

NOT responsible for:
- Dataset loading (handled by shared data utilities / orchestrator).
- Evaluation (handled by gencysynth/eval/*).

Expected config keys (with safe defaults)
----------------------------------------
IMG_SHAPE: [H, W, C]              # channels_last; C=1 for grayscale
NUM_CLASSES: 9

# GMM hyperparameters
GMM_COMPONENTS: 10                # default per_class components
GMM_COMPONENTS_BY_CLASS: {4: 16}  # optional per_class override
COVARIANCE_TYPE: "full"           # {"full","tied","diag","spherical"}
REG_COVAR: 1e_6                   # numeric stabilizer; retries will increase if needed
MAX_ITER: 300
N_INIT: 1
INIT_PARAMS: "kmeans"
RANDOM_STATE: 42
VERBOSE: 0
TOL: 1e_3

# PCA (optional)
USE_PCA: false
PCA_DIM: 128                      # if USE_PCA true and not set -> min(128, D)
PCA_WHITEN: true
PCA_SVDSOLVER: "auto"             # {"auto","full","randomized","arpack"}

# Synthesis
SAMPLES_PER_CLASS: 1000
SAMPLES_PER_CLASS_BY_CLASS: {4: 400, 7: 400}

# Rule_A paths (preferred, used by resolve_run_context):
paths:
  artifacts: artifacts

model:
  tag: gaussianmixture/c_gmm_full

run_meta:
  run_id: <optional>
  seed: 42
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import warnings

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

from gencysynth.models.base_types import RunContext
from gencysynth.orchestration.context import resolve_run_context
from gencysynth.orchestration.logger import get_run_logger
from gencysynth.utils.paths import ensure_dir

from gaussianmixture.models import (
    GMMConfig,
    build_gmm_model,
    flatten_images,
    reshape_to_images,
)

# -----------------------------------------------------------------------------
# Variant identity (must match registry tag and folder structure)
# -----------------------------------------------------------------------------
FAMILY: str = "gaussianmixture"
VARIANT: str = "c_gmm_full"
MODEL_TAG: str = "gaussianmixture/c_gmm_full"


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _cfg_get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _labels_to_int(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Accept one_hot (N,K) or int (N,) → int labels (N,)."""
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == num_classes:
        return np.argmax(y, axis=1).astype(np.int64)
    if y.ndim == 1:
        return y.astype(np.int64)
    raise ValueError(f"Labels must be (N,) ints or (N,{num_classes}) one_hot; got {y.shape}")


def _is_fitted_gmm(gmm: GaussianMixture) -> bool:
    """Lightweight check for fitted sklearn GMM objects."""
    return hasattr(gmm, "weights_") and getattr(gmm, "weights_", None) is not None


def _resolve_int(cfg: Dict[str, Any], key: str, default: int) -> int:
    v = cfg.get(key, default)
    return int(v)


def _resolve_float(cfg: Dict[str, Any], key: str, default: float) -> float:
    v = cfg.get(key, default)
    return float(v)


# -----------------------------------------------------------------------------
# Config dataclass (local to this pipeline)
# -----------------------------------------------------------------------------
@dataclass
class GMMPipelineConfig:
    # Data / shapes
    IMG_SHAPE: Tuple[int, int, int] = (40, 40, 1)
    NUM_CLASSES: int = 9

    # GMM hyperparameters
    GMM_COMPONENTS: int = 10
    COVARIANCE_TYPE: str = "full"
    TOL: float = 1e_3
    REG_COVAR: float = 1e_6
    MAX_ITER: int = 300
    N_INIT: int = 1
    INIT_PARAMS: str = "kmeans"
    RANDOM_STATE: Optional[int] = 42
    VERBOSE: int = 0

    # PCA front_end (optional)
    USE_PCA: bool = False
    PCA_DIM: Optional[int] = None
    PCA_WHITEN: bool = True
    PCA_SVDSOLVER: str = "auto"

    # Synthesis
    SAMPLES_PER_CLASS: int = 1000


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------
class GaussianMixturePipeline:
    """
    Orchestrates training + synthesis for class_conditional GMMs under Rule A.

    IMPORTANT:
    - All artifact IO goes through `ctx.run_dir` (Rule A).
    - This class assumes the orchestrator already loaded and preprocessed data.
    """

    def __init__(self, cfg: Dict[str, Any], ctx: RunContext):
        self.cfg_raw = cfg or {}
        self.ctx = ctx

        if self.ctx.run_dir is None or self.ctx.logs_dir is None:
            raise ValueError("RunContext must have run_dir and logs_dir resolved (Rule A).")

        # Logger (run_scoped)
        self.logger = get_run_logger(
            name=f"{MODEL_TAG}:{self.ctx.run_id}:pipeline",
            log_dir=Path(self.ctx.logs_dir),
        )

        # Normalize config into dataclass
        self.cfg = GMMPipelineConfig(
            IMG_SHAPE=tuple(self.cfg_raw.get("IMG_SHAPE", (40, 40, 1))),
            NUM_CLASSES=_resolve_int(self.cfg_raw, "NUM_CLASSES", 9),
            GMM_COMPONENTS=_resolve_int(self.cfg_raw, "GMM_COMPONENTS", 10),
            COVARIANCE_TYPE=str(self.cfg_raw.get("COVARIANCE_TYPE", "full")),
            TOL=_resolve_float(self.cfg_raw, "TOL", 1e_3),
            REG_COVAR=_resolve_float(self.cfg_raw, "REG_COVAR", 1e_6),
            MAX_ITER=_resolve_int(self.cfg_raw, "MAX_ITER", 300),
            N_INIT=_resolve_int(self.cfg_raw, "N_INIT", 1),
            INIT_PARAMS=str(self.cfg_raw.get("INIT_PARAMS", "kmeans")),
            RANDOM_STATE=self.cfg_raw.get("RANDOM_STATE", self.ctx.seed),
            VERBOSE=_resolve_int(self.cfg_raw, "VERBOSE", 0),
            USE_PCA=bool(self.cfg_raw.get("USE_PCA", False)),
            PCA_DIM=self.cfg_raw.get("PCA_DIM", None),
            PCA_WHITEN=bool(self.cfg_raw.get("PCA_WHITEN", True)),
            PCA_SVDSOLVER=str(self.cfg_raw.get("PCA_SVDSOLVER", "auto")),
            SAMPLES_PER_CLASS=_resolve_int(self.cfg_raw, "SAMPLES_PER_CLASS", 1000),
        )

        # Per_class overrides (optional)
        self.samples_per_class_by_class: Dict[int, int] = self.cfg_raw.get("SAMPLES_PER_CLASS_BY_CLASS", {}) or {}
        self.components_by_class: Dict[int, int] = self.cfg_raw.get("GMM_COMPONENTS_BY_CLASS", {}) or {}

        # Rule A paths (canonical)
        run_dir = Path(self.ctx.run_dir)
        self.ckpt_dir = ensure_dir(run_dir / "checkpoints")
        self.synth_dir = ensure_dir(run_dir / "synthetic")
        self.samples_dir = ensure_dir(run_dir / "samples")

        # In_memory state
        self.models: List[Optional[GaussianMixture]] = [None] * self.cfg.NUM_CLASSES
        self.global_fallback_: Optional[GaussianMixture] = None
        self.pca_: Optional[PCA] = None
        self.trained_in_pca_: bool = False

        self.logger.info("Initialized GMM pipeline (Rule A).")
        self.logger.info(f"run_dir={run_dir}")
        self.logger.info(f"ckpt_dir={self.ckpt_dir}")
        self.logger.info(f"synth_dir={self.synth_dir}")

    # -------------------------------------------------------------------------
    # PCA utilities
    # -------------------------------------------------------------------------
    def _fit_pca(self, X64: np.ndarray) -> Tuple[PCA, np.ndarray]:
        """
        Fit PCA on float64 data and return (pca, Z).

        If PCA_DIM is None, selects min(128, D).
        PCA is saved to: ctx.run_dir/checkpoints/PCA.joblib
        """
        D = int(X64.shape[1])
        n_comp = int(self.cfg.PCA_DIM if self.cfg.PCA_DIM is not None else min(128, D))

        self.logger.info(f"Fitting PCA: n_components={n_comp}, whiten={self.cfg.PCA_WHITEN}, svd={self.cfg.PCA_SVDSOLVER}")

        pca = PCA(
            n_components=n_comp,
            whiten=self.cfg.PCA_WHITEN,
            svd_solver=self.cfg.PCA_SVDSOLVER,
            random_state=self.cfg.RANDOM_STATE,
        )
        Z = pca.fit_transform(X64)

        joblib.dump(pca, self.ckpt_dir / "PCA.joblib")
        self.logger.info("Saved PCA.joblib")

        return pca, Z

    def _load_pca(self) -> Optional[PCA]:
        """
        Load PCA checkpoint if present. We treat presence of PCA.joblib as the
        authoritative signal that training used PCA space.
        """
        p = self.ckpt_dir / "PCA.joblib"
        if p.exists():
            if self.pca_ is None:
                self.pca_ = joblib.load(p)
            return self.pca_
        return None

    # -------------------------------------------------------------------------
    # Robust GMM fitting
    # -------------------------------------------------------------------------
    def _fit_gmm_with_retries(
        self,
        X64: np.ndarray,
        *,
        base_cfg: GMMConfig,
        class_id: int,
        n_components: int,
    ) -> GaussianMixture:
        """
        Fit a GMM with a sequence of increasingly stable settings.

        Why retries:
        - Full_covariance GMM can fail on ill_conditioned data (especially image
          pixels or low_sample classes).
        - Instead of crashing, we progressively increase regularization and/or
          fall back to simpler covariance structures.
        """
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        attempts = [
            # 1) As configured, but guard reg_covar
            dict(covariance_type=base_cfg.covariance_type, reg_covar=max(base_cfg.reg_covar, 1e_6), n_components=n_components),
            # 2) More regularization
            dict(covariance_type=base_cfg.covariance_type, reg_covar=max(base_cfg.reg_covar, 1e_4), n_components=n_components),
            # 3) Even more regularization
            dict(covariance_type=base_cfg.covariance_type, reg_covar=max(base_cfg.reg_covar, 5e_4), n_components=n_components),
            # 4) Switch to diag covariance
            dict(covariance_type="diag", reg_covar=max(base_cfg.reg_covar, 1e_4), n_components=n_components),
            # 5) diag + fewer components
            dict(covariance_type="diag", reg_covar=max(base_cfg.reg_covar, 1e_3), n_components=max(1, n_components // 2)),
            # 6) spherical last resort
            dict(covariance_type="spherical", reg_covar=max(base_cfg.reg_covar, 1e_3), n_components=max(1, n_components // 2)),
        ]

        for i, params in enumerate(attempts, 1):
            cfg_try = GMMConfig(
                n_components=int(params["n_components"]),
                covariance_type=str(params["covariance_type"]),
                tol=base_cfg.tol,
                reg_covar=float(params["reg_covar"]),
                max_iter=base_cfg.max_iter,
                n_init=base_cfg.n_init,
                init_params=base_cfg.init_params,
                random_state=base_cfg.random_state,
                verbose=base_cfg.verbose,
            )

            tag = f"class {class_id}" if class_id >= 0 else "global"
            self.logger.info(
                f"[{tag}] attempt {i}: GMM(n={cfg_try.n_components}, cov='{cfg_try.covariance_type}', reg={cfg_try.reg_covar:g})"
            )

            gmm = build_gmm_model(cfg_try)
            try:
                gmm.fit(X64)
                return gmm
            except Exception as e:
                self.logger.warning(f"[{tag}] attempt {i} failed: {type(e).__name__}: {e}")

        raise RuntimeError(
            f"GMM training failed for class {class_id} after {len(attempts)} attempts. "
            f"Try lowering GMM_COMPONENTS, using COVARIANCE_TYPE='diag', or increasing REG_COVAR."
        )

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        *,
        save_checkpoints: bool = True,
    ) -> None:
        """
        Train one GMM per class.

        Inputs
        ------
        x_train : float32 (N,H,W,C) in [0,1]
        y_train : int (N,) or one_hot (N,K)

        Outputs (Rule A)
        ----------------
        ctx.run_dir/checkpoints/
          GMM_class_{k}.joblib
          GMM_global_fallback.joblib   (if needed)
          PCA.joblib                   (if USE_PCA)
          GMM_LAST_OK                  marker file
        """
        H, W, C = self.cfg.IMG_SHAPE
        K = self.cfg.NUM_CLASSES

        y_ids = _labels_to_int(y_train, K)

        # Flatten to (N, D). The helper can clip + assume [0,1].
        X = flatten_images(x_train, img_shape=(H, W, C), assume_01=True, clip=True)
        X64 = np.asarray(X, dtype=np.float64, order="C")
        N, D = X64.shape

        self.logger.info(f"Training GMMs: N={N}, D={D}, classes={K}, use_pca={self.cfg.USE_PCA}")

        # Optional PCA (fit on all data once)
        Z64 = X64
        if self.cfg.USE_PCA:
            self.pca_, Z64 = self._fit_pca(X64)
            self.trained_in_pca_ = True
        else:
            self.trained_in_pca_ = False

        # Fit per_class models
        for k in range(K):
            idx = (y_ids == k)
            Zk = Z64[idx]
            nk = int(Zk.shape[0])

            if nk == 0:
                self.logger.warning(f"Class {k} has no training samples; skipping (will use global fallback).")
                self.models[k] = None
                continue

            # Components: per_class override > default; never exceed sample count
            n_comp_cfg = int(self.components_by_class.get(k, self.cfg.GMM_COMPONENTS))
            n_comp = int(min(max(1, n_comp_cfg), nk))

            base_cfg = GMMConfig(
                n_components=n_comp,
                covariance_type=self.cfg.COVARIANCE_TYPE,
                tol=self.cfg.TOL,
                reg_covar=self.cfg.REG_COVAR,
                max_iter=self.cfg.MAX_ITER,
                n_init=self.cfg.N_INIT,
                init_params=self.cfg.INIT_PARAMS,
                random_state=self.cfg.RANDOM_STATE,
                verbose=self.cfg.VERBOSE,
            )

            gmm = self._fit_gmm_with_retries(Zk, base_cfg=base_cfg, class_id=k, n_components=n_comp)
            self.models[k] = gmm

            if save_checkpoints:
                p = self.ckpt_dir / f"GMM_class_{k}.joblib"
                joblib.dump(gmm, p)
                self.logger.info(f"Saved {p.name}")

        # Global fallback if any class missing
        if any(m is None for m in self.models):
            self.logger.info("Training global fallback GMM (some classes missing).")

            # A modest default for global components: <= default and <= ~N/10
            n_comp_global = int(min(max(1, self.cfg.GMM_COMPONENTS), max(1, N // 10)))

            base_cfg_global = GMMConfig(
                n_components=n_comp_global,
                covariance_type=self.cfg.COVARIANCE_TYPE,
                tol=self.cfg.TOL,
                reg_covar=self.cfg.REG_COVAR,
                max_iter=self.cfg.MAX_ITER,
                n_init=self.cfg.N_INIT,
                init_params=self.cfg.INIT_PARAMS,
                random_state=self.cfg.RANDOM_STATE,
                verbose=self.cfg.VERBOSE,
            )

            self.global_fallback_ = self._fit_gmm_with_retries(
                Z64, base_cfg=base_cfg_global, class_id=-1, n_components=n_comp_global
            )

            if save_checkpoints:
                p = self.ckpt_dir / "GMM_global_fallback.joblib"
                joblib.dump(self.global_fallback_, p)
                self.logger.info(f"Saved {p.name}")
        else:
            self.global_fallback_ = None

        # Marker file: tells downstream tools "checkpoints look coherent"
        (self.ckpt_dir / "GMM_LAST_OK").write_text("ok", encoding="utf_8")
        self.logger.info("Wrote GMM_LAST_OK marker")

    # -------------------------------------------------------------------------
    # Checkpoint loading (used during synthesis when pipeline is created fresh)
    # -------------------------------------------------------------------------
    def _load_model_for_class(self, k: int) -> Optional[GaussianMixture]:
        if 0 <= k < len(self.models) and self.models[k] is not None and _is_fitted_gmm(self.models[k]):
            return self.models[k]
        p = self.ckpt_dir / f"GMM_class_{k}.joblib"
        if p.exists():
            gmm = joblib.load(p)
            self.models[k] = gmm
            return gmm
        return None

    def _load_global_fallback(self) -> Optional[GaussianMixture]:
        if self.global_fallback_ is not None and _is_fitted_gmm(self.global_fallback_):
            return self.global_fallback_
        p = self.ckpt_dir / "GMM_global_fallback.joblib"
        if p.exists():
            self.global_fallback_ = joblib.load(p)
            return self.global_fallback_
        return None

    # -------------------------------------------------------------------------
    # Synthesis
    # -------------------------------------------------------------------------
    def synthesize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a class_balanced synthetic dataset (float32 in [0,1]).

        Reads
        -----
        ctx.run_dir/checkpoints/  (GMM_class_*.joblib, optional PCA.joblib)

        Writes (Rule A)
        ---------------
        ctx.run_dir/synthetic/
          gen_class_{k}.npy
          labels_class_{k}.npy
          x_synth.npy
          y_synth.npy

        Returns
        -------
        x_synth : float32 (N,H,W,C) in [0,1]
        y_synth : float32 (N,K) one_hot
        """
        H, W, C = self.cfg.IMG_SHAPE
        K = self.cfg.NUM_CLASSES

        # Decide whether we are in PCA space based on checkpoint presence
        pca = self._load_pca()
        using_pca = pca is not None
        if using_pca:
            self.logger.info("PCA.joblib found → sampling in PCA space then inverse_transform() to pixel space.")
        else:
            self.logger.info("No PCA.joblib → sampling directly in pixel space.")

        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []

        ensure_dir(self.synth_dir)

        for k in range(K):
            gmm = self._load_model_for_class(k)
            if gmm is None:
                gmm = self._load_global_fallback()
                if gmm is None:
                    raise FileNotFoundError(
                        f"No GMM checkpoint for class {k} and no global fallback found in {self.ckpt_dir}"
                    )
                self.logger.warning(f"[class {k}] missing class checkpoint → using global fallback for sampling.")

            # Per_class sampling override
            per_class = int(self.samples_per_class_by_class.get(k, self.cfg.SAMPLES_PER_CLASS))

            # Sample from the *training space* (PCA space if PCA was used, else pixel space)
            Z_flat, _ = gmm.sample(per_class)  # (per_class, Dz or D)
            Z_flat = np.asarray(Z_flat, dtype=np.float64)

            # Map back to pixel space if PCA was used
            if using_pca:
                X_flat = pca.inverse_transform(Z_flat)
            else:
                X_flat = Z_flat

            # Clip to [0,1] (GMM can sample outside bounds)
            X_flat = np.clip(X_flat, 0.0, 1.0)

            # Reshape to images (H,W,C)
            imgs = reshape_to_images(X_flat.astype(np.float32, copy=False), (H, W, C), clip=True)

            # Evaluator contract: per_class dumps
            np.save(self.synth_dir / f"gen_class_{k}.npy", imgs)
            np.save(self.synth_dir / f"labels_class_{k}.npy", np.full((imgs.shape[0],), k, dtype=np.int32))

            xs.append(imgs)
            y1h = np.zeros((imgs.shape[0], K), dtype=np.float32)
            y1h[:, k] = 1.0
            ys.append(y1h)

            self.logger.info(f"[class {k}] wrote {imgs.shape[0]} samples")

        x_synth = np.concatenate(xs, axis=0).astype(np.float32)
        y_synth = np.concatenate(ys, axis=0).astype(np.float32)

        # Drop non_finite samples (rare but safe)
        mask = np.isfinite(x_synth).all(axis=(1, 2, 3))
        if not mask.all():
            dropped = int((~mask).sum())
            self.logger.warning(f"Dropping {dropped} non_finite synthetic samples.")
            x_synth = x_synth[mask]
            y_synth = y_synth[mask]

        # Combined convenience dumps
        np.save(self.synth_dir / "x_synth.npy", x_synth)
        np.save(self.synth_dir / "y_synth.npy", y_synth)

        self.logger.info(f"Synthesis complete: N={x_synth.shape[0]} → {self.synth_dir}")
        return x_synth, y_synth


# -----------------------------------------------------------------------------
# Rule_A adapters (so orchestrator can call train(cfg, ctx) / synth(cfg, ctx))
# -----------------------------------------------------------------------------
def train(cfg: Dict[str, Any], ctx: RunContext) -> int:
    """
    Orchestrator entrypoint (Rule A).

    The orchestrator is expected to load dataset and pass x_train/y_train into
    the pipeline's .train() method directly in most setups; however, we keep
    this function for symmetry with other families.

    If your orchestrator calls this function, it must provide data in cfg under:
      cfg["_payload"]["x_train"], cfg["_payload"]["y_train"]
    (This mirrors patterns used in some of your other adapters.)

    Returns 0 on success.
    """
    payload = cfg.get("_payload", {}) or {}
    x_train = payload.get("x_train", None)
    y_train = payload.get("y_train", None)
    if x_train is None or y_train is None:
        raise ValueError("train(cfg, ctx) requires cfg['_payload'] with x_train and y_train.")

    pipe = GaussianMixturePipeline(cfg, ctx)
    pipe.train(np.asarray(x_train), np.asarray(y_train), save_checkpoints=True)
    return 0


def synth(cfg: Dict[str, Any], ctx: RunContext) -> int:
    """
    Orchestrator entrypoint (Rule A): load checkpoints from ctx.run_dir/checkpoints
    and write evaluator_ready .npy dumps to ctx.run_dir/synthetic.

    Returns 0 on success.
    """
    pipe = GaussianMixturePipeline(cfg, ctx)
    pipe.synthesize()
    return 0


# -----------------------------------------------------------------------------
# Legacy_friendly wrapper (if old code instantiates pipeline without ctx)
# -----------------------------------------------------------------------------
def build_pipeline(cfg: Dict[str, Any]) -> GaussianMixturePipeline:
    """
    Convenience builder that resolves a Rule_A RunContext from config.

    This is the safest way to keep older scripts working while still enforcing
    Rule A artifact placement.
    """
    resolved = resolve_run_context(cfg, create_dirs=True)
    return GaussianMixturePipeline(resolved.cfg, resolved.ctx)


# Back_compat alias so `from gaussianmixture.pipeline import GMMPipeline` works.
GMMPipeline = GaussianMixturePipeline

__all__ = [
    "GaussianMixturePipeline",
    "GMMPipeline",
    "build_pipeline",
    "train",
    "synth",
]