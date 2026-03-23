# src/gencysynth/metrics/sanity/shape_checks.py
"""
Sanity checks: shape, dtype, and basic validity for image datasets and labels.

Why this exists
---------------
For end_to_end "smoke tests" (e.g., <5 epochs across multiple model families),
we want fast checks that catch the most common pipeline failures early:

- wrong image shape (missing channel dim, wrong H/W)
- dtype issues (uint8 vs float, float64, etc.)
- invalid ranges (e.g., model expects [-1,1] but gets [0,1], or vice_versa)
- NaN/Inf contamination
- label shape mismatches (int vs one_hot confusion)
- class id out of range

Design goals
------------
- Zero ML framework dependencies (NumPy only).
- Safe on huge arrays: supports optional subsampling.
- Returns a structured dict that can be merged into run_level eval summaries
  (Rule A friendly). No file I/O here; a separate writer can persist results.

Typical usage
-------------
from gencysynth.metrics.sanity.shape_checks import check_images, check_labels, check_pair

img_report = check_images(x_real, expected_shape=(40,40,1), range_hint="01")
lbl_report = check_labels(y_real, num_classes=9)

pair_report = check_pair(real=(x_real,y_real), synth=(x_syn,y_syn), num_classes=9)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np


ArrayLike = Union[np.ndarray, Sequence[Any]]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _as_np(x: ArrayLike) -> np.ndarray:
    return np.asarray(x)


def _safe_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _finite_summary(x: np.ndarray) -> Dict[str, Any]:
    finite = np.isfinite(x)
    n_total = int(x.size)
    n_finite = int(finite.sum())
    return {
        "n_total": n_total,
        "n_finite": n_finite,
        "n_nonfinite": int(n_total - n_finite),
        "all_finite": bool(n_total == n_finite),
    }


def _maybe_subsample_rows(x: np.ndarray, max_rows: Optional[int], seed: int = 0) -> np.ndarray:
    """
    Subsample the first axis if max_rows is set and x is large.
    Keeps behavior deterministic with seed.
    """
    if max_rows is None:
        return x
    if x.ndim == 0:
        return x
    if x.shape[0] <= int(max_rows):
        return x
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(x.shape[0], size=int(max_rows), replace=False)
    return x[idx]


def _infer_range_hint(x: np.ndarray) -> str:
    """
    Heuristic: infer whether array looks like [0,1], [-1,1], or "other".
    """
    if x.size == 0:
        return "empty"
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))

    # Loose tolerances to avoid false alarms from minor numeric drift.
    if x_min >= -0.05 and x_max <= 1.05:
        return "01"
    if x_min >= -1.05 and x_max <= 1.05:
        # if it dips below ~-0.05, likely [-1,1]
        if x_min < -0.05:
            return "m11"
        return "01"
    return "other"


def _shape_str(x: np.ndarray) -> str:
    try:
        return str(tuple(x.shape))
    except Exception:
        return "<unknown>"


# -----------------------------------------------------------------------------
# Image checks
# -----------------------------------------------------------------------------
def check_images(
    x: np.ndarray,
    *,
    expected_shape: Optional[Tuple[int, int, int]] = None,
    # range_hint: "01" for [0,1], "m11" for [-1,1], or None to infer
    range_hint: Optional[str] = None,
    allow_uint8: bool = True,
    max_rows: Optional[int] = 2048,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Validate an image array.

    Expected conventions in this repo
    ---------------------------------
    - Most metrics expect images shaped (N,H,W,C) float32.
    - Some generators (like VAE tanh) may train on [-1,1], but metrics typically
      evaluate on [0,1]. We therefore allow both and report what we detect.

    Returns
    -------
    dict with:
      - ok: bool
      - errors: [str]
      - warnings: [str]
      - observed: {shape, dtype, inferred_range, min, max, mean, std, finite...}
      - expected: {shape, range_hint}
    """
    errors = []
    warnings = []

    x = _as_np(x)

    observed: Dict[str, Any] = {
        "shape": _shape_str(x),
        "ndim": int(getattr(x, "ndim", -1)),
        "dtype": str(getattr(x, "dtype", "unknown")),
    }

    # Basic dimensional expectations
    if x.ndim not in (2, 4):
        errors.append(f"images must be 4D (N,H,W,C) or 2D (N,D). got ndim={x.ndim}, shape={x.shape}")

    # If 4D, check H/W/C if expected provided
    if x.ndim == 4 and expected_shape is not None:
        eh, ew, ec = expected_shape
        oh, ow, oc = x.shape[1], x.shape[2], x.shape[3]
        if (oh, ow, oc) != (eh, ew, ec):
            errors.append(
                f"image spatial shape mismatch: expected (H,W,C)={(eh,ew,ec)} but got {(oh,ow,oc)}"
            )

    # dtype handling
    if x.dtype == np.uint8:
        if not allow_uint8:
            errors.append("uint8 images not allowed here (expected float array).")
        else:
            warnings.append("images are uint8; downstream steps may require float32 scaling to [0,1].")
    elif x.dtype in (np.float16, np.float32, np.float64):
        if x.dtype == np.float64:
            warnings.append("images are float64; consider casting to float32 for speed/consistency.")
    else:
        warnings.append(f"unusual dtype for images: {x.dtype} (expected float32 or uint8).")

    # Subsample for stats
    xs = _maybe_subsample_rows(x, max_rows=max_rows, seed=seed)

    # Range + finiteness (only if numeric)
    if np.issubdtype(xs.dtype, np.number) and xs.size > 0:
        fin = _finite_summary(xs)
        observed.update({"finite": fin})
        if not fin["all_finite"]:
            errors.append(f"images contain NaN/Inf: nonfinite={fin['n_nonfinite']}/{fin['n_total']}")

        # Compute basic stats robustly (ignore NaNs)
        x_min = float(np.nanmin(xs))
        x_max = float(np.nanmax(xs))
        x_mean = float(np.nanmean(xs))
        x_std = float(np.nanstd(xs))

        observed.update({"min": x_min, "max": x_max, "mean": x_mean, "std": x_std})

        inferred = _infer_range_hint(xs.astype(np.float32, copy=False))
        observed["inferred_range"] = inferred

        desired = range_hint or None
        if desired is not None and desired not in ("01", "m11"):
            warnings.append(f"unknown range_hint='{desired}'. expected '01' or 'm11'.")

        if range_hint is not None:
            # Check against hint with loose tolerances
            if range_hint == "01" and (x_min < -0.05 or x_max > 1.05):
                warnings.append(f"images appear out of [0,1] range: min={x_min:.4f}, max={x_max:.4f}")
            if range_hint == "m11" and (x_min < -1.05 or x_max > 1.05):
                warnings.append(f"images appear out of [-1,1] range: min={x_min:.4f}, max={x_max:.4f}")
        else:
            # No hint provided; still warn if range is very unusual
            if inferred == "other":
                warnings.append(f"images range looks unusual: min={x_min:.4f}, max={x_max:.4f}")

    expected: Dict[str, Any] = {}
    if expected_shape is not None:
        expected["shape"] = (None, *expected_shape)  # N,H,W,C
    if range_hint is not None:
        expected["range_hint"] = range_hint

    ok = (len(errors) == 0)
    return {"ok": ok, "errors": errors, "warnings": warnings, "observed": observed, "expected": expected}


# -----------------------------------------------------------------------------
# Label checks
# -----------------------------------------------------------------------------
def check_labels(
    y: np.ndarray,
    *,
    num_classes: Optional[int] = None,
    allow_onehot: bool = True,
    allow_int: bool = True,
    max_rows: Optional[int] = 4096,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Validate labels.

    Supported forms
    ---------------
    - int labels: shape (N,), values in [0, K_1]
    - one_hot:    shape (N,K) with rows summing to 1 (tolerant)

    Returns a dict: {ok, errors, warnings, observed, expected}
    """
    errors = []
    warnings = []

    y = _as_np(y)

    observed: Dict[str, Any] = {
        "shape": _shape_str(y),
        "ndim": int(getattr(y, "ndim", -1)),
        "dtype": str(getattr(y, "dtype", "unknown")),
    }

    ys = _maybe_subsample_rows(y, max_rows=max_rows, seed=seed)

    if y.ndim == 1:
        if not allow_int:
            errors.append("integer labels are not allowed here (expected one_hot).")
        if not np.issubdtype(y.dtype, np.integer):
            # could be float ints; warn not fail
            warnings.append(f"labels are 1D but dtype is {y.dtype}; expected integer ids.")
        if num_classes is not None and ys.size > 0:
            y_min = int(np.min(ys))
            y_max = int(np.max(ys))
            observed.update({"min": y_min, "max": y_max})
            if y_min < 0 or y_max >= int(num_classes):
                errors.append(f"label ids out of range: min={y_min}, max={y_max}, expected [0,{num_classes_1}]")

    elif y.ndim == 2:
        if not allow_onehot:
            errors.append("one_hot labels are not allowed here (expected integer ids).")
        if num_classes is not None and y.shape[1] != int(num_classes):
            errors.append(f"one_hot width mismatch: got K={y.shape[1]} but expected K={num_classes}")

        if np.issubdtype(y.dtype, np.number) and ys.size > 0:
            # check row sums ~ 1 and values ~ {0,1}
            row_sums = np.sum(ys, axis=1)
            rs_min = float(np.min(row_sums))
            rs_max = float(np.max(row_sums))
            observed.update({"row_sum_min": rs_min, "row_sum_max": rs_max})

            if rs_min < 0.5 or rs_max > 1.5:
                warnings.append("one_hot rows do not appear to sum to ~1. (Are these soft labels?)")

            # check argmax range if K provided
            if num_classes is not None:
                ids = np.argmax(ys, axis=1)
                y_min = int(np.min(ids)) if ids.size else 0
                y_max = int(np.max(ids)) if ids.size else 0
                if y_min < 0 or y_max >= int(num_classes):
                    errors.append(f"argmax ids out of range: min={y_min}, max={y_max}, expected [0,{num_classes_1}]")
    else:
        errors.append(f"labels must be 1D (N,) or 2D (N,K). got ndim={y.ndim}, shape={y.shape}")

    expected: Dict[str, Any] = {}
    if num_classes is not None:
        expected["num_classes"] = int(num_classes)

    ok = (len(errors) == 0)
    return {"ok": ok, "errors": errors, "warnings": warnings, "observed": observed, "expected": expected}


# -----------------------------------------------------------------------------
# Pair checks (real vs synth)
# -----------------------------------------------------------------------------
def check_pair(
    *,
    real: Tuple[np.ndarray, Optional[np.ndarray]],
    synth: Tuple[np.ndarray, Optional[np.ndarray]],
    expected_shape: Optional[Tuple[int, int, int]] = None,
    num_classes: Optional[int] = None,
    real_range_hint: Optional[str] = None,
    synth_range_hint: Optional[str] = None,
    max_rows: Optional[int] = 2048,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Validate a (real, synth) pair consistently.

    Returns
    -------
    {
      "ok": bool,
      "errors": [...],
      "warnings": [...],
      "real": {"images":..., "labels":...},
      "synth": {"images":..., "labels":...}
    }
    """
    xr, yr = real
    xs, ys = synth

    r_img = check_images(xr, expected_shape=expected_shape, range_hint=real_range_hint, max_rows=max_rows, seed=seed)
    s_img = check_images(xs, expected_shape=expected_shape, range_hint=synth_range_hint, max_rows=max_rows, seed=seed)

    r_lab = check_labels(yr, num_classes=num_classes, max_rows=max_rows, seed=seed) if yr is not None else None
    s_lab = check_labels(ys, num_classes=num_classes, max_rows=max_rows, seed=seed) if ys is not None else None

    errors = []
    warnings = []

    errors.extend(r_img["errors"])
    errors.extend(s_img["errors"])
    warnings.extend(r_img["warnings"])
    warnings.extend(s_img["warnings"])

    if r_lab is not None:
        errors.extend(r_lab["errors"])
        warnings.extend(r_lab["warnings"])
    if s_lab is not None:
        errors.extend(s_lab["errors"])
        warnings.extend(s_lab["warnings"])

    ok = (len(errors) == 0)
    return {"ok": ok, "errors": errors, "warnings": warnings, "real": {"images": r_img, "labels": r_lab}, "synth": {"images": s_img, "labels": s_lab}}


__all__ = ["check_images", "check_labels", "check_pair"]
