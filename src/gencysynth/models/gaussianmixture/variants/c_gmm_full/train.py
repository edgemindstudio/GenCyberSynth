# src/gencysynth/models/gaussianmixture/variants/c_gmm_full/train.py
"""
GenCyberSynth — GaussianMixture family — c_gmm_full variant (per_class, full covariance) — Training
===============================================================================================

RULE A (Scalable artifact policy)
---------------------------------
Everything is keyed by:
    (dataset_id, model_tag, run_id)

Therefore this training module MUST write ONLY under the resolved run context:

  artifacts/
    runs/<dataset_id>/<model_tag>/<run_id>/
      checkpoints/            # GMM_class_{k}.joblib, GMM_global_fallback.joblib
      samples/                # preview grids sampled from the fitted models
      tensorboard/            # unused here (CPU sklearn), but folder is allowed
      run_meta_snapshot.json  # config snapshot for audit/debug

and logs MUST go to:

  artifacts/logs/<dataset_id>/<model_tag>/<run_id>/
    run.log

What this module does
---------------------
- Loads YAML config and resolves RunContext (dataset_id/model_tag/run_id/seed).
- Loads dataset via shared loader (common.data.load_dataset_npy) when available.
- Trains one full_covariance GaussianMixture model per class.
- Optionally trains a global fallback GMM on the full dataset.
- Saves joblib checkpoints under ctx.run_dir/checkpoints/.
- Writes a small preview PNG under ctx.run_dir/samples/.

What this module MUST NOT do
----------------------------
- Invent custom artifact layouts (no artifacts/gaussianmixture/...).
- Write outside Rule A run directories.
- Perform evaluation (belongs in gencysynth/eval).

CLI entry
---------
python -m gencysynth.models.gaussianmixture.variants.c_gmm_full.train --config <path/to/config.yaml>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.mixture import GaussianMixture

# -----------------------------------------------------------------------------
# Prefer shared loader; keep fallback support if import fails
# -----------------------------------------------------------------------------
try:
    from common.data import load_dataset_npy  # type: ignore
except Exception:
    load_dataset_npy = None  # fallback mode used if import fails

# -----------------------------------------------------------------------------
# GenCyberSynth shared plumbing (Rule A)
# -----------------------------------------------------------------------------
from gencysynth.orchestration.context import resolve_run_context
from gencysynth.orchestration.logger import get_run_logger
from gencysynth.utils.io import write_json
from gencysynth.utils.paths import ensure_dir
from gencysynth.utils.reproducibility import now_iso


# -----------------------------------------------------------------------------
# Variant identity
# -----------------------------------------------------------------------------
FAMILY: str = "gaussianmixture"
VARIANT: str = "c_gmm_full"
MODEL_TAG: str = "gaussianmixture/c_gmm_full"


# =============================================================================
# Small helpers
# =============================================================================
def _cfg_get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    """Read nested config values using dotted keys (e.g., 'paths.data_root')."""
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _to_int_labels(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Accept labels as:
      - (N,) integer class ids
      - (N,K) one_hot
    Return:
      - (N,) integer class ids
    """
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == num_classes:
        return np.argmax(y, axis=1).astype(np.int32)
    return y.astype(np.int32)


def _flatten_images(x: np.ndarray, img_shape: Tuple[int, int, int]) -> np.ndarray:
    """(N,H,W,C) -> (N,D) float32."""
    H, W, C = img_shape
    x = np.asarray(x, dtype=np.float32)
    return x.reshape((x.shape[0], H * W * C))


def _save_preview_grid(
    *,
    models_dir: Path,
    out_path: Path,
    img_shape: Tuple[int, int, int],
    num_classes: int,
    seed: int,
    per_class: int = 1,
) -> None:
    """
    Sample a small number of images from each fitted per_class GMM checkpoint,
    then save a simple grid PNG under the run samples folder.

    This is ONLY a lightweight visual sanity check; it is not evaluation.
    """
    import matplotlib.pyplot as plt

    H, W, C = img_shape
    rng = np.random.default_rng(int(seed))

    # Load global fallback if present (used when a per_class model is missing)
    fallback_path = models_dir / "GMM_global_fallback.joblib"
    fallback = joblib.load(fallback_path) if fallback_path.exists() else None

    # Collect images: per class, sample `per_class`
    imgs: list[np.ndarray] = []
    titles: list[str] = []

    for k in range(num_classes):
        ckpt = models_dir / f"GMM_class_{k}.joblib"
        gmm = joblib.load(ckpt) if ckpt.exists() else fallback
        if gmm is None:
            # If truly nothing exists, just skip
            continue

        # sklearn sample returns (X, y)
        Xk, _ = gmm.sample(per_class, random_state=rng.integers(0, 2**31 - 1))
        Xk = np.asarray(Xk, dtype=np.float32)

        # Convert to images in [0,1] and clamp
        Xk = np.clip(Xk, 0.0, 1.0)
        Xk_img = Xk.reshape((per_class, H, W, C))

        for i in range(per_class):
            imgs.append(Xk_img[i])
            titles.append(f"C{k}")

    if not imgs:
        return

    # Layout: rows=num_classes, cols=per_class (but handle missing classes gracefully)
    cols = max(1, per_class)
    rows = int(np.ceil(len(imgs) / cols))

    plt.figure(figsize=(cols * 1.6, rows * 1.6))
    for i, im in enumerate(imgs):
        ax = plt.subplot(rows, cols, i + 1)
        if C == 1:
            ax.imshow(im[:, :, 0], cmap="gray", vmin=0.0, vmax=1.0)
        else:
            ax.imshow(np.clip(im, 0.0, 1.0))
        ax.set_title(titles[i], fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=200)
    plt.close()


def _snapshot_run_meta(cfg: dict, run_dir: Path) -> Path:
    """
    Write a stable config snapshot for reproducibility / debugging.
    Stored directly in the run directory by Rule A.
    """
    out = Path(run_dir) / "run_meta_snapshot.json"
    payload = {
        "timestamp": now_iso(),
        "model_tag": cfg.get("model", {}).get("tag"),
        "dataset_id": cfg.get("dataset", {}).get("id"),
        "run_meta": cfg.get("run_meta", {}),
        "paths": cfg.get("paths", {}),
        "cfg": cfg,
    }
    return write_json(out, payload, indent=2, sort_keys=True, atomic=True)


# =============================================================================
# Core training: per_class GMMs (full covariance)
# =============================================================================
def train_per_class_gmms(
    *,
    x_train01: np.ndarray,                 # (N,H,W,C) float in [0,1]
    y_train: np.ndarray,                   # (N,) ints or (N,K) one_hot
    img_shape: Tuple[int, int, int],
    num_classes: int,
    ckpt_dir: Path,
    n_components: int = 10,
    reg_covar: float = 1e_6,
    max_iter: int = 200,
    random_state: int = 42,
    train_global_fallback: bool = True,
    verbose: bool = True,
    logger=None,
) -> None:
    """
    Fit one GaussianMixture model per class and save under ckpt_dir.

    Important guardrails
    --------------------
    - GMM expects flattened vectors, so images are flattened to D=H*W*C.
    - We cap components per class to <= number of samples in that class.
    - If a class has <2 samples, we skip that class; sampler can use global fallback.
    """
    H, W, C = img_shape
    D = H * W * C

    ensure_dir(ckpt_dir)

    y_int = _to_int_labels(y_train, num_classes)
    X_flat = _flatten_images(x_train01, img_shape)

    # ----------------- Optional: global fallback -----------------
    if train_global_fallback:
        if verbose:
            msg = f"[GMM] Training global fallback on {X_flat.shape[0]} samples (D={D})"
            print(msg)
            if logger:
                logger.info(msg)

        k_components = min(int(n_components), max(1, int(X_flat.shape[0])))
        fallback = GaussianMixture(
            n_components=k_components,
            covariance_type="full",
            reg_covar=float(reg_covar),
            max_iter=int(max_iter),
            random_state=int(random_state),
            verbose=1 if verbose else 0,
        )
        fallback.fit(X_flat)
        joblib.dump(fallback, ckpt_dir / "GMM_global_fallback.joblib")

    # ----------------- Per_class models -----------------
    for k in range(num_classes):
        idx = (y_int == k)
        count = int(idx.sum())

        if verbose:
            msg = f"[GMM] Class {k}: n={count}"
            print(msg)
            if logger:
                logger.info(msg)

        if count < 2:
            warn = f"[warn] Class {k} has <2 samples; skipping per_class GMM (will rely on fallback at sampling)."
            print(warn)
            if logger:
                logger.warning(warn)
            continue

        Xk = X_flat[idx]
        k_components = min(int(n_components), count)

        gmm = GaussianMixture(
            n_components=k_components,
            covariance_type="full",
            reg_covar=float(reg_covar),
            max_iter=int(max_iter),
            random_state=int(random_state),
            verbose=1 if verbose else 0,
        )
        gmm.fit(Xk)
        joblib.dump(gmm, ckpt_dir / f"GMM_class_{k}.joblib")

    if verbose:
        msg = f"[GMM] Saved checkpoints → {ckpt_dir}"
        print(msg)
        if logger:
            logger.info(msg)


# =============================================================================
# High_level runner (Rule A)
# =============================================================================
def run_from_file(cfg_path: Path) -> int:
    """
    Main entrypoint that:
      - loads YAML
      - resolves RunContext (Rule A)
      - loads dataset
      - trains GMMs
      - writes preview
    """
    import yaml

    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    if not isinstance(cfg, dict):
        raise TypeError("Config YAML must parse to a dict/object.")

    # -------------------------------------------------------------------------
    # Enforce variant identity in config (helps avoid path collisions)
    # -------------------------------------------------------------------------
    cfg.setdefault("model", {})
    if isinstance(cfg["model"], dict):
        cfg["model"].setdefault("tag", MODEL_TAG)
        cfg["model"].setdefault("family", FAMILY)
        cfg["model"].setdefault("variant", VARIANT)

    cfg.setdefault("dataset", {})
    if isinstance(cfg["dataset"], dict):
        cfg["dataset"].setdefault("id", cfg.get("dataset_id", "unknown_dataset"))

    # Resolve ctx + inject run_meta/paths (Rule A)
    resolved = resolve_run_context(cfg, create_dirs=True)
    ctx = resolved.ctx
    cfg = resolved.cfg

    assert ctx.run_dir is not None and ctx.logs_dir is not None

    run_dir = Path(ctx.run_dir)
    log_dir = Path(ctx.logs_dir)

    # Standard run subfolders (Rule A)
    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    samples_dir = ensure_dir(run_dir / "samples")

    # Logger for this run
    logger = get_run_logger(name=f"{MODEL_TAG}:{ctx.run_id}", log_dir=log_dir)

    logger.info("=== c_gmm_full TRAIN START ===")
    logger.info(f"dataset_id={ctx.dataset_id} model_tag={ctx.model_tag} run_id={ctx.run_id} seed={ctx.seed}")
    logger.info(f"run_dir={run_dir}")
    logger.info(f"logs_dir={log_dir}")

    # Snapshot config for reproducibility
    _snapshot_run_meta(cfg, run_dir)

    # -------------------------------------------------------------------------
    # Hyperparameters
    # -------------------------------------------------------------------------
    seed = int(_cfg_get(cfg, "run_meta.seed", _cfg_get(cfg, "SEED", 42)))
    np.random.seed(seed)

    img_shape = tuple(_cfg_get(cfg, "IMG_SHAPE", (40, 40, 1)))
    img_shape = (int(img_shape[0]), int(img_shape[1]), int(img_shape[2]))
    num_classes = int(_cfg_get(cfg, "NUM_CLASSES", _cfg_get(cfg, "num_classes", 9)))
    val_fraction = float(_cfg_get(cfg, "VAL_FRACTION", 0.5))

    n_components = int(_cfg_get(cfg, "GMM_COMPONENTS", 10))
    reg_covar = float(_cfg_get(cfg, "GMM_REG_COVAR", 1e_6))
    max_iter = int(_cfg_get(cfg, "GMM_MAX_ITER", 200))
    train_global = bool(_cfg_get(cfg, "GMM_TRAIN_GLOBAL", True))

    # Resolve dataset root (prefer Rule A style paths.data_root)
    data_root = _cfg_get(cfg, "paths.data_root", None) or _cfg_get(cfg, "DATA_DIR", None) or _cfg_get(cfg, "data.root", None)
    if data_root is None:
        # conservative fallback: relative to config location
        data_root = (cfg_path.resolve().parents[1] / "USTC_TFC2016_malware").resolve()
    data_dir = Path(data_root)

    logger.info(
        f"Config: img_shape={img_shape} K={num_classes} seed={seed} "
        f"components={n_components} reg_covar={reg_covar} max_iter={max_iter} "
        f"train_global={train_global} val_fraction={val_fraction}"
    )
    logger.info(f"DATA_DIR={data_dir}")

    # -------------------------------------------------------------------------
    # Load dataset (expects [0,1] images)
    # -------------------------------------------------------------------------
    if load_dataset_npy is None:
        raise ImportError(
            "common.data.load_dataset_npy is required for this variant under the new repo layout. "
            "Please ensure common.data is available."
        )

    x_train01, y_train, x_val01, y_val, x_test01, y_test = load_dataset_npy(
        data_dir, img_shape, num_classes, val_fraction=val_fraction
    )

    # -------------------------------------------------------------------------
    # Train per_class GMMs and save under run checkpoints
    # -------------------------------------------------------------------------
    train_per_class_gmms(
        x_train01=x_train01,
        y_train=y_train,
        img_shape=img_shape,
        num_classes=num_classes,
        ckpt_dir=ckpt_dir,
        n_components=n_components,
        reg_covar=reg_covar,
        max_iter=max_iter,
        random_state=seed,
        train_global_fallback=train_global,
        verbose=True,
        logger=logger,
    )

    # -------------------------------------------------------------------------
    # Preview grid (1 sample per class) under run samples folder
    # -------------------------------------------------------------------------
    preview_path = samples_dir / "gmm_preview.png"
    try:
        _save_preview_grid(
            models_dir=ckpt_dir,
            out_path=preview_path,
            img_shape=img_shape,
            num_classes=num_classes,
            seed=seed,
            per_class=1,
        )
        logger.info(f"Saved preview grid → {preview_path}")
    except Exception as e:
        logger.warning(f"Could not write preview grid: {e}")

    logger.info("=== c_gmm_full TRAIN END ===")
    return 0


# =============================================================================
# Orchestrator adapter + CLI
# =============================================================================
def train(cfg_or_argv) -> int:
    """
    Orchestrator entrypoint.

    Accepts:
      - argv_like list/tuple  -> CLI args
      - dict config           -> written to a temp YAML then executed

    Returns 0 on success.
    """
    if isinstance(cfg_or_argv, (list, tuple)):
        return main(list(cfg_or_argv))

    if isinstance(cfg_or_argv, dict):
        import tempfile
        import yaml

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(cfg_or_argv, f)
            tmp = f.name
        return main(["--config", tmp])

    raise TypeError(f"Unsupported config payload type: {type(cfg_or_argv)}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=f"Train per_class GMMs ({MODEL_TAG})")
    p.add_argument("--config", type=Path, default=Path("configs/config.yaml"), help="Path to YAML config")
    args = p.parse_args(argv)
    return run_from_file(args.config)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "train_per_class_gmms",
    "run_from_file",
    "train",
    "main",
]