# src/gencysynth/models/gaussianmixture/variants/c_gmm_full/model.py
"""
GenCyberSynth — GaussianMixture — c_gmm_full — Model Utilities
=============================================================

This module is **stateless** and contains only:
- sklearn GaussianMixture construction
- image flatten/reshape utilities
- a small sampling helper (given a *fitted* GMM)

Rule A note
-----------
**No artifact I/O happens here.** All reading/writing of checkpoints, synthetic
arrays, manifests, and previews is handled by the pipeline/sampler modules under
Rule A directories:

  artifacts/runs/<dataset_id>/<model_tag>/<run_id>/

This file should remain import_safe (NumPy + sklearn only) and reusable across
datasets and variants.

Conventions
-----------
- Images are channels_last: (N, H, W, C)
- Pixel range is expected to be [0, 1] float32 for training.
  If legacy arrays are 0..255, we auto_normalize when `assume_01=True`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Any

import numpy as np
from sklearn.mixture import GaussianMixture


# -----------------------------------------------------------------------------
# Public config for sklearn GaussianMixture (nice defaults and IDE support)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class GMMConfig:
    """
    Configuration for building a scikit_learn GaussianMixture.

    Parameters mirror sklearn.mixture.GaussianMixture with sensible defaults.
    """
    n_components: int = 10
    covariance_type: str = "full"      # {"full","tied","diag","spherical"}
    tol: float = 1e_3
    reg_covar: float = 1e_6
    max_iter: int = 300
    n_init: int = 1
    init_params: str = "kmeans"        # {"kmeans","random"}
    random_state: Optional[int] = 42
    warm_start: bool = False
    verbose: int = 0


# -----------------------------------------------------------------------------
# Builders
# -----------------------------------------------------------------------------
def build_gmm_model(cfg: Optional[GMMConfig] = None, **overrides: Any) -> GaussianMixture:
    """
    Construct an (unfitted) GaussianMixture instance.

    Usage
    -----
    - With config:
        gmm = build_gmm_model(GMMConfig(n_components=20, covariance_type="diag"))
    - With overrides:
        gmm = build_gmm_model(n_components=20, covariance_type="diag")

    Returns
    -------
    sklearn.mixture.GaussianMixture (UNFITTED)
    """
    cfg = cfg or GMMConfig()
    params = dict(
        n_components=int(cfg.n_components),
        covariance_type=str(cfg.covariance_type),
        tol=float(cfg.tol),
        reg_covar=float(cfg.reg_covar),
        max_iter=int(cfg.max_iter),
        n_init=int(cfg.n_init),
        init_params=str(cfg.init_params),
        random_state=cfg.random_state,
        warm_start=bool(cfg.warm_start),
        verbose=int(cfg.verbose),
    )
    params.update(overrides or {})
    return GaussianMixture(**params)


def create_gmm(**kwargs: Any) -> GaussianMixture:
    """
    Thin convenience alias used by legacy callers.
    Prefer build_gmm_model() for typed defaults.
    """
    return GaussianMixture(**kwargs)


# -----------------------------------------------------------------------------
# Data utilities
# -----------------------------------------------------------------------------
def flatten_images(
    x: np.ndarray,
    img_shape: Optional[Tuple[int, int, int]] = None,
    *,
    assume_01: bool = True,
    clip: bool = True,
) -> np.ndarray:
    """
    Convert HWC images to a flat 2D array (N, D).

    Accepted inputs
    ---------------
    - x: (N, H, W, C) images
    - x: (N, D) already flattened

    Normalization behavior
    ----------------------
    If assume_01=True:
      - If max(x) > 1.5, treat as 0..255 and divide by 255.
      - Otherwise assume already in [0,1].
    Then (optionally) clip to [0,1].

    Parameters
    ----------
    x : np.ndarray
        Input images or already_flattened array.
    img_shape : Optional[Tuple[int,int,int]]
        If provided and x is 4D, validates/reshapes to this shape.
        If provided and x is not 4D, attempts reshape to (-1,*img_shape).
    assume_01 : bool
        Whether to auto_normalize from 0..255 to 0..1 when needed.
    clip : bool
        Clamp output to [0,1] (safety for sampling / legacy arrays).

    Returns
    -------
    flat : (N, D) float32 in [0,1] (if clip=True)
    """
    x = np.asarray(x)

    # Already flattened: just normalize/clip safely
    if x.ndim == 2:
        flat = x.astype(np.float32, copy=False)
        if assume_01 and float(np.nanmax(flat)) > 1.5:
            flat = flat / 255.0
        if clip:
            flat = np.clip(flat, 0.0, 1.0)
        return flat

    # Image tensors
    if x.ndim != 4:
        # Allow reshaping if img_shape is provided (legacy callers)
        if img_shape is None:
            raise ValueError(f"Expected (N,H,W,C) or (N,D); got shape {x.shape}")
        try:
            x = x.reshape((-1, *img_shape))
        except Exception as e:
            raise ValueError(f"Could not reshape {x.shape} to (-1, {img_shape})") from e

    if img_shape is not None and tuple(x.shape[1:]) != tuple(img_shape):
        # Try to coerce shape (useful when legacy arrays are (N, D) stored as 4D incorrectly)
        try:
            x = x.reshape((-1, *img_shape))
        except Exception as e:
            raise ValueError(
                f"Input image shape {tuple(x.shape[1:])} does not match img_shape={img_shape} "
                f"and cannot be reshaped safely."
            ) from e

    x = x.astype(np.float32, copy=False)
    if assume_01 and float(np.nanmax(x)) > 1.5:
        x = x / 255.0
    if clip:
        x = np.clip(x, 0.0, 1.0)

    n = int(x.shape[0])
    return x.reshape(n, -1)


def reshape_to_images(
    flat: np.ndarray,
    img_shape: Tuple[int, int, int],
    *,
    clip: bool = True,
) -> np.ndarray:
    """
    Reshape flattened samples back to HWC images.

    Parameters
    ----------
    flat : (N, D) array
    img_shape : (H, W, C)
    clip : clamp to [0,1]

    Returns
    -------
    imgs : (N, H, W, C) float32
    """
    flat = np.asarray(flat, dtype=np.float32)
    H, W, C = img_shape
    expected = int(H * W * C)

    if flat.ndim != 2 or int(flat.shape[1]) != expected:
        raise ValueError(f"Flat array must have shape (N, {expected}); got {flat.shape}")

    imgs = flat.reshape(-1, H, W, C)
    if clip:
        imgs = np.clip(imgs, 0.0, 1.0)
    return imgs


# -----------------------------------------------------------------------------
# Sampling helper
# -----------------------------------------------------------------------------
def sample_gmm_images(
    gmm: GaussianMixture,
    n: int,
    img_shape: Tuple[int, int, int],
    *,
    clip: bool = True,
) -> np.ndarray:
    """
    Draw `n` samples from a *fitted* GMM and return images.

    Notes
    -----
    - Assumes the GMM was trained on flattened images normalized to [0,1].
    - GMM sampling can produce small out_of_range values; clipping is a safety guard.

    Returns
    -------
    imgs : (n, H, W, C) float32 in [0,1] if clip=True
    """
    if not hasattr(gmm, "weights_") or getattr(gmm, "weights_", None) is None:
        raise ValueError("The provided GMM appears to be unfitted. Call gmm.fit(...) first.")

    flat, _ = gmm.sample(int(n))  # (n, D)
    flat = np.asarray(flat, dtype=np.float32)

    if clip:
        flat = np.clip(flat, 0.0, 1.0)

    return reshape_to_images(flat, img_shape, clip=clip)


__all__ = [
    "GMMConfig",
    "build_gmm_model",
    "create_gmm",
    "flatten_images",
    "reshape_to_images",
    "sample_gmm_images",
]