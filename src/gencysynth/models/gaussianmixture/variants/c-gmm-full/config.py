# src/gencysynth/models/gaussianmixture/variants/c-gmm-full/config.py
"""
GenCyberSynth — GaussianMixture — c-gmm-full — Config & Defaults
===============================================================

This module defines **variant-scoped** defaults and small helpers used by:
- train.py / pipeline.py (training)
- samply.py / sample.py  (sampling)

Rule A (Scalability rule)
-------------------------
All artifact I/O is keyed by:
    (dataset_id, model_tag, run_id)

This config module does **not** write artifacts; it only:
- supplies defaults
- normalizes config keys (new + legacy compatibility)
- provides dotted-key accessors

The orchestrator is expected to resolve a RunContext and pass the resolved cfg.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple


# -----------------------------------------------------------------------------
# Variant identity (must match folder structure and registry tag)
# -----------------------------------------------------------------------------
FAMILY: str = "gaussianmixture"
VARIANT: str = "c-gmm-full"
MODEL_TAG: str = "gaussianmixture/c-gmm-full"


# -----------------------------------------------------------------------------
# Dotted-key config helper (shared pattern across variants)
# -----------------------------------------------------------------------------
def cfg_get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    """
    Read nested config values using a dotted key, e.g. "paths.artifacts".
    Returns `default` if any part of the path is missing.
    """
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


# -----------------------------------------------------------------------------
# Canonical defaults (mirrors defaults.yaml)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class CfgDefaults:
    # Data / shape
    img_shape: Tuple[int, int, int] = (40, 40, 1)
    num_classes: int = 9

    # Training: per-class GMM
    gmm_components: int = 10
    covariance_type: str = "full"   # {"full","tied","diag","spherical"}
    tol: float = 1e-3
    reg_covar: float = 1e-6
    max_iter: int = 300
    n_init: int = 1
    init_params: str = "kmeans"
    random_state: int = 42
    verbose: int = 0
    train_global_fallback: bool = True

    # Optional PCA front-end
    use_pca: bool = False
    pca_dim: Optional[int] = None     # if None, choose min(128, D) at runtime
    pca_whiten: bool = True
    pca_svdsolver: str = "auto"

    # Synthesis budget (used by sampler)
    synth_n_per_class: int = 25

    # Run meta
    seed: int = 42
    dataset_id: str = "USTC-TFC2016_40x40_gray"


DEFAULTS = CfgDefaults()


# -----------------------------------------------------------------------------
# Legacy key normalization
# -----------------------------------------------------------------------------
def normalize_cfg(cfg_in: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a shallow-normalized cfg dict.

    Goals:
    - Provide **canonical keys** used by Rule-A-aware orchestrators:
        cfg["model"]["tag"]
        cfg["run_meta"]["seed"]
        cfg["dataset"]["id"]  (or data.dataset_id)
        cfg["img"]["shape"]
        cfg["data"]["num_classes"]
        cfg["synth"]["n_per_class"]
    - Keep compatibility with legacy flat keys used in older scripts:
        IMG_SHAPE, NUM_CLASSES, SAMPLES_PER_CLASS, SEED, DATA_DIR, etc.

    This function should be safe to call multiple times.
    """
    cfg = dict(cfg_in or {})

    # ---- model tag ----
    cfg.setdefault("model", {})
    if isinstance(cfg["model"], dict):
        cfg["model"].setdefault("tag", MODEL_TAG)

    # ---- dataset id (Rule A key component) ----
    # We support: dataset.id, data.dataset_id, DATASET_ID, or fall back to DEFAULTS.dataset_id
    cfg.setdefault("dataset", {})
    if isinstance(cfg["dataset"], dict):
        cfg["dataset"].setdefault("id", cfg_get(cfg, "data.dataset_id", cfg.get("DATASET_ID", DEFAULTS.dataset_id)))

    # ---- run meta (seed) ----
    cfg.setdefault("run_meta", {})
    if isinstance(cfg["run_meta"], dict):
        cfg["run_meta"].setdefault("seed", int(cfg.get("SEED", cfg_get(cfg, "seed", DEFAULTS.seed))))

    # ---- image shape ----
    cfg.setdefault("img", {})
    if isinstance(cfg["img"], dict):
        cfg["img"].setdefault(
            "shape",
            tuple(cfg.get("IMG_SHAPE", cfg_get(cfg, "img.shape", DEFAULTS.img_shape))),
        )

    # ---- class count ----
    cfg.setdefault("data", {})
    if isinstance(cfg["data"], dict):
        cfg["data"].setdefault(
            "num_classes",
            int(cfg.get("NUM_CLASSES", cfg.get("num_classes", cfg_get(cfg, "data.num_classes", DEFAULTS.num_classes)))),
        )

    # ---- synthesis budget ----
    cfg.setdefault("synth", {})
    if isinstance(cfg["synth"], dict):
        cfg["synth"].setdefault(
            "n_per_class",
            int(cfg_get(cfg, "synth.n_per_class", cfg.get("SAMPLES_PER_CLASS", cfg.get("samples_per_class", DEFAULTS.synth_n_per_class)))),
        )

    # ---- gmm hyperparams ----
    cfg.setdefault("gmm", {})
    if isinstance(cfg["gmm"], dict):
        cfg["gmm"].setdefault("components", int(cfg.get("GMM_COMPONENTS", cfg_get(cfg, "gmm.components", DEFAULTS.gmm_components))))
        cfg["gmm"].setdefault("covariance_type", str(cfg.get("COVARIANCE_TYPE", cfg_get(cfg, "gmm.covariance_type", DEFAULTS.covariance_type))))
        cfg["gmm"].setdefault("tol", float(cfg.get("TOL", cfg_get(cfg, "gmm.tol", DEFAULTS.tol))))
        cfg["gmm"].setdefault("reg_covar", float(cfg.get("REG_COVAR", cfg_get(cfg, "gmm.reg_covar", DEFAULTS.reg_covar))))
        cfg["gmm"].setdefault("max_iter", int(cfg.get("MAX_ITER", cfg_get(cfg, "gmm.max_iter", DEFAULTS.max_iter))))
        cfg["gmm"].setdefault("n_init", int(cfg.get("N_INIT", cfg_get(cfg, "gmm.n_init", DEFAULTS.n_init))))
        cfg["gmm"].setdefault("init_params", str(cfg.get("INIT_PARAMS", cfg_get(cfg, "gmm.init_params", DEFAULTS.init_params))))
        cfg["gmm"].setdefault("verbose", int(cfg.get("VERBOSE", cfg_get(cfg, "gmm.verbose", DEFAULTS.verbose))))
        cfg["gmm"].setdefault("train_global_fallback", bool(cfg.get("GMM_TRAIN_GLOBAL", cfg_get(cfg, "gmm.train_global_fallback", DEFAULTS.train_global_fallback))))

        # Optional per-class override maps (kept as dict[int,int] downstream)
        cfg["gmm"].setdefault("components_by_class", cfg.get("GMM_COMPONENTS_BY_CLASS", cfg_get(cfg, "gmm.components_by_class", {})) or {})

    # ---- optional PCA ----
    cfg.setdefault("pca", {})
    if isinstance(cfg["pca"], dict):
        cfg["pca"].setdefault("use", bool(cfg.get("USE_PCA", cfg_get(cfg, "pca.use", DEFAULTS.use_pca))))
        cfg["pca"].setdefault("dim", cfg.get("PCA_DIM", cfg_get(cfg, "pca.dim", DEFAULTS.pca_dim)))
        cfg["pca"].setdefault("whiten", bool(cfg.get("PCA_WHITEN", cfg_get(cfg, "pca.whiten", DEFAULTS.pca_whiten))))
        cfg["pca"].setdefault("svd_solver", str(cfg.get("PCA_SVDSOLVER", cfg_get(cfg, "pca.svd_solver", DEFAULTS.pca_svdsolver))))

    return cfg


__all__ = [
    "FAMILY",
    "VARIANT",
    "MODEL_TAG",
    "CfgDefaults",
    "DEFAULTS",
    "cfg_get",
    "normalize_cfg",
]