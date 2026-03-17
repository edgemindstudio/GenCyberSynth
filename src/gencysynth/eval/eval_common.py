# src/gencysynth/eval/eval_common.py
"""
GenCyberSynth — Unified Evaluation Utilities
============================================

This module provides the *core* evaluation logic used across GenCyberSynth runs:

1) Generative distribution metrics
   - FID (macro), cFID (per_class + macro), KID, JS/KL (pixel hist), Diversity
   - Optional domain_FID (FID in a caller_provided embedding space)

2) Downstream utility metrics (EvalCNN probe)
   - Train a small CNN on:
       (a) REAL_only
       (b) REAL + SYNTH
     then evaluate both on the REAL test set.

3) Summary shaping + optional writing
   - Map raw metrics into a stable summary schema
   - Optionally write outputs into a dataset_scalable artifacts layout.

Dataset scalability & path policy
---------------------------------
This file intentionally does NOT load datasets from disk and does NOT assume any
single dataset layout. It operates on arrays provided by the caller.

When writing summary outputs, we support a dataset_aware convention:

    {artifacts_root}/eval/{dataset_id}/{model_tag}/{run_id}/summary.jsonl
    {artifacts_root}/eval/{dataset_id}/{model_tag}/{run_id}/summary.txt

Where:
- artifacts_root: cfg["paths"]["artifacts"] or "artifacts"
- dataset_id: a stable dataset identifier (e.g. "USTC_TFC2016_40x40_gray")
- model_tag: variant_aware tag (e.g. "gan/dcgan", "diffusion/ddpm")
- run_id: identifies a run (e.g. "A_seed42" or "seed42")

Design goals
------------
- Robust: missing optional deps or metric failures do not crash the suite.
- Reproducible: controlled seeds + careful normalization.
- Compatible: summary includes both structured blocks and legacy flattened keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, List, Any
from gencysynth.data.transforms import to_01_hwc, onehot_to_int

import numpy as np
try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None

from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    average_precision_score, roc_curve, precision_recall_fscore_support
)
from sklearn.utils.class_weight import compute_class_weight
try:
    from scipy.linalg import sqrtm
except Exception:  # pragma: no cover
    sqrtm = None

from scipy.stats import entropy
from tensorflow.keras import layers, models, optimizers

# -----------------------------------------------------------------------------
# OPTIONAL dependency
# -----------------------------------------------------------------------------
# Keep this optional/non_fatal for HPC + library usage.
try:
    from scripts.eval_write_summary import write_phase2_summary  # noqa: F401
except Exception:  # pragma: no cover
    write_phase2_summary = None  # type: ignore[assignment]


# -----------------------------------------------------------------------------
# Recommended: use shared repo utils for paths and IO
# -----------------------------------------------------------------------------
# If these don't exist yet, you can either:
#   (a) create them under gencysynth/utils/, OR
#   (b) keep using the local fallback resolve_eval_output_paths() below.
try:
    from gencysynth.utils.paths import resolve_eval_paths  # recommended
except Exception:  # pragma: no cover
    resolve_eval_paths = None  # type: ignore[assignment]

try:
    from gencysynth.utils.io import append_jsonl, write_text  # recommended
except Exception:  # pragma: no cover
    append_jsonl = None  # type: ignore[assignment]
    write_text = None    # type: ignore[assignment]


# ======================================================================================
# 0) Repro / basic utilities
# ======================================================================================

def set_global_seed(seed: int = 42) -> None:
    """
    Set RNG seeds for reproducibility across NumPy and TensorFlow.

    Notes:
    - Reduces run_to_run variance.
    - Does not guarantee perfect determinism on all systems (TF kernels may vary).
    """
    if tf is None:
        return None  # or a dict of None metrics

    np.random.seed(int(seed))
    try:
        tf.random.set_seed(int(seed))
    except Exception:
        pass


def onehot_to_int(y: np.ndarray) -> np.ndarray:
    """
    Convert labels to integer class ids.

    Accepted formats:
    - (N,) int labels
    - (N,K) one_hot or probabilities -> argmax(axis=1)
    """
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] > 1:
        return y.argmax(axis=1).astype(int, copy=False)
    return y.astype(int, copy=False)


# ======================================================================================
# 0.5) Path utilities (fallback)
# ======================================================================================
# Prefer gencysynth.utils.paths.resolve_eval_paths(). This fallback exists so that
# eval_common.py still works even before you finish centralizing path policy.

def _safe_slug(s: str, *, max_len: int = 120) -> str:
    """Filesystem_friendly token for dataset_id / model_tag / run_id."""
    if not isinstance(s, str):
        return "unknown"
    s = s.strip().replace(" ", "_")
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", ".", "/"):
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out).strip("_")
    return slug[:max_len] if slug else "unknown"


@dataclass(frozen=True)
class EvalOutputPaths:
    """Resolved output paths for JSONL + console text."""
    jsonl_path: Path
    console_path: Path


def resolve_eval_output_paths(
    *,
    artifacts_root: str | Path = "artifacts",
    dataset_id: str = "unknown_dataset",
    model_tag: str = "unknown_model",
    run_id: str = "run",
) -> EvalOutputPaths:
    """
    Fallback resolver matching the repo convention:

      artifacts/eval/<dataset_id>/<model_tag>/<run_id>/summary.jsonl
      artifacts/eval/<dataset_id>/<model_tag>/<run_id>/summary.txt
    """
    ar = Path(artifacts_root)
    ds = _safe_slug(dataset_id)
    mt = _safe_slug(model_tag)
    rid = _safe_slug(run_id)

    out_dir = ar / "eval" / ds / mt / rid
    return EvalOutputPaths(
        jsonl_path=out_dir / "summary.jsonl",
        console_path=out_dir / "summary.txt",
    )


# ======================================================================================
# 1) Generative metrics: FID / cFID / JS / KL / Diversity / KID (+ optional domain_FID)
# ======================================================================================

_inception_model: Optional[tf.keras.Model] = None
_inception_ok = True


def _get_inception() -> Optional[tf.keras.Model]:
    """
    Lazy_load + cache InceptionV3 pooling backbone for FID/KID.

    Failure behavior:
    - If creation fails once, mark as unavailable to avoid repeated slow failures in logs.
    """
    global _inception_model, _inception_ok
    if not _inception_ok:
        return None

    if _inception_model is None:
        try:
            _inception_model = tf.keras.applications.InceptionV3(
                include_top=False,
                pooling="avg",
                input_shape=(299, 299, 3),
            )
        except Exception:
            _inception_ok = False
            _inception_model = None

    return _inception_model


def _fid_from_activations(a: np.ndarray, b: np.ndarray) -> float:
    """Fréchet distance in feature space (drop tiny imaginary numeric artifacts)."""
    mu1, sig1 = a.mean(axis=0), np.cov(a, rowvar=False)
    mu2, sig2 = b.mean(axis=0), np.cov(b, rowvar=False)

    diff = mu1 - mu2

    if sqrtm is None:
        return None

    covmean = sqrtm(sig1 @ sig2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sig1 + sig2 - 2.0 * covmean))


def fid_keras(real_01: np.ndarray, fake_01: np.ndarray) -> Optional[float]:
    """
    FID using InceptionV3 pooled activations.
    Inputs must be float32 [0,1], shape (N,H,W,C).
    """
    inc = _get_inception()
    if inc is None:
        return None

    real = tf.image.resize(real_01, (299, 299))
    fake = tf.image.resize(fake_01, (299, 299))

    real = tf.image.grayscale_to_rgb(real)
    fake = tf.image.grayscale_to_rgb(fake)

    real = tf.keras.applications.inception_v3.preprocess_input(real * 255.0)
    fake = tf.keras.applications.inception_v3.preprocess_input(fake * 255.0)

    a = inc.predict(real, verbose=0)
    b = inc.predict(fake, verbose=0)
    return _fid_from_activations(a, b)


def fid_in_domain_embedding(
    real_01: np.ndarray,
    fake_01: np.ndarray,
    embed_fn: Callable[[np.ndarray], np.ndarray],
) -> Optional[float]:
    """Optional FID in a caller_provided embedding space."""
    try:
        return _fid_from_activations(embed_fn(real_01), embed_fn(fake_01))
    except Exception:
        return None


def js_kl_on_pixels(
    real01: np.ndarray,
    fake01: np.ndarray,
    bins: int = 256,
    eps: float = 1e_8,
) -> Tuple[Optional[float], Optional[float]]:
    """Histogram_based JS + KL on flattened pixel intensities."""
    try:
        pr, _ = np.histogram(real01.ravel(), bins=bins, range=(0, 1), density=True)
        pf, _ = np.histogram(fake01.ravel(), bins=bins, range=(0, 1), density=True)

        pr = pr + eps
        pf = pf + eps
        m = 0.5 * (pr + pf)

        js = 0.5 * (entropy(pr, m) + entropy(pf, m))
        kl = entropy(pr, pf)

        return float(js), float(kl)
    except Exception:
        return None, None


def diversity_score(x01: np.ndarray) -> Optional[float]:
    """Pixel_variance diversity proxy (quick signal; not semantic diversity)."""
    try:
        flat = x01.reshape((x01.shape[0], -1))
        return float(np.mean(np.var(flat, axis=0)))
    except Exception:
        return None


def _poly_mmd2_unbiased(
    A: np.ndarray,
    B: np.ndarray,
    degree: int = 3,
    gamma: Optional[float] = None,
    coef0: float = 1.0
) -> float:
    """Unbiased polynomial_kernel MMD^2 estimator (KID core)."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    m, d = A.shape
    n, _ = B.shape
    if m < 2 or n < 2:
        return float("nan")
    if gamma is None:
        gamma = 1.0 / d

    Kaa = (gamma * (A @ A.T) + coef0) ** degree
    Kbb = (gamma * (B @ B.T) + coef0) ** degree
    Kab = (gamma * (A @ B.T) + coef0) ** degree

    np.fill_diagonal(Kaa, 0.0)
    np.fill_diagonal(Kbb, 0.0)

    return float(
        Kaa.sum() / (m * (m - 1)) +
        Kbb.sum() / (n * (n - 1)) -
        2.0 * Kab.mean()
    )


def kid_keras(
    real_01: np.ndarray,
    fake_01: np.ndarray,
    subset: int = 200,
    n_subsets: int = 10,
    seed: int = 42,
    degree: int = 3
) -> Optional[float]:
    """
    KID computed as polynomial MMD^2 on Inception pooled activations.
    Returns mean over subsets, or None if Inception is unavailable.
    """
    inc = _get_inception()
    if inc is None:
        return None

    rng = np.random.default_rng(int(seed))
    m = min(len(real_01), len(fake_01), int(subset))
    if m < 2:
        return None

    vals: List[float] = []
    for _ in range(int(n_subsets)):
        r_idx = rng.choice(len(real_01), size=m, replace=False)
        f_idx = rng.choice(len(fake_01), size=m, replace=False)

        r = tf.image.resize(real_01[r_idx], (299, 299))
        f = tf.image.resize(fake_01[f_idx], (299, 299))
        r = tf.image.grayscale_to_rgb(r)
        f = tf.image.grayscale_to_rgb(f)

        r = tf.keras.applications.inception_v3.preprocess_input(r * 255.0)
        f = tf.keras.applications.inception_v3.preprocess_input(f * 255.0)

        a = inc.predict(r, verbose=0)
        b = inc.predict(f, verbose=0)
        vals.append(_poly_mmd2_unbiased(a, b, degree=degree))

    return float(np.mean(vals)) if vals else None


# ======================================================================================
# 2) Evaluator CNN + downstream utility metrics
# ======================================================================================

def _build_eval_cnn(img_shape: tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    """Lightweight downstream probe model for utility evaluation."""
    inp = layers.Input(shape=img_shape, name="x")

    def maybe_pool(x):
        h, w = x.shape[1], x.shape[2]
        if (h is not None and h < 2) or (w is not None and w < 2):
            return x
        return layers.MaxPooling2D(pool_size=2, strides=2, padding="valid")(x)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = maybe_pool(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = maybe_pool(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = maybe_pool(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.1)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    m = models.Model(inp, out, name="EvalCNN")
    m.compile(optimizer=optimizers.Adam(learning_rate=1e_3),
              loss="categorical_crossentropy",
              metrics=["accuracy"])
    return m


def _macro_auprc(y_true_int: np.ndarray, proba: np.ndarray, K: int) -> float:
    """Macro_AUPRC (one_vs_rest)."""
    y_oh = np.eye(K)[y_true_int]
    aps = [average_precision_score(y_oh[:, k], proba[:, k]) for k in range(K)]
    return float(np.mean(aps))


def _recall_at_fpr(y_true_int: np.ndarray, proba: np.ndarray, K: int, target_fpr: float = 0.01) -> float:
    """Macro_average Recall@FPR<=target across classes."""
    y_oh = np.eye(K)[y_true_int]
    recalls: List[float] = []
    for k in range(K):
        try:
            fpr, tpr, _ = roc_curve(y_oh[:, k], proba[:, k])
            idx = np.where(fpr <= float(target_fpr))[0]
            recalls.append(0.0 if len(idx) == 0 else float(np.max(tpr[idx])))
        except Exception:
            recalls.append(0.0)
    return float(np.mean(recalls))


def _ece(proba: np.ndarray, y_true_int: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (ECE)."""
    conf = proba.max(axis=1)
    preds = proba.argmax(axis=1)
    correct = (preds == y_true_int).astype(float)

    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    N = len(conf)

    for i in range(int(n_bins)):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            continue
        ece += abs(correct[m].mean() - conf[m].mean()) * (m.sum() / N)

    return float(ece)


def _brier_multiclass(y_true_oh: np.ndarray, proba: np.ndarray) -> float:
    """Multiclass Brier score."""
    y_true_oh = y_true_oh.astype(np.float32)
    proba = proba.astype(np.float32)
    return float(np.mean(np.sum((proba - y_true_oh) ** 2, axis=1)))


def _fit_eval_cnn(
    x_train: np.ndarray, y_train_oh: np.ndarray,
    x_val: np.ndarray,   y_val_oh: np.ndarray,
    input_shape: Tuple[int, int, int],
    num_classes: int,
    class_weight: Optional[Dict[int, float]] = None,
    seed: int = 42,
    epochs: int = 20,
) -> tf.keras.Model:
    """Train EvalCNN with early stopping (class_weight optional)."""
    set_global_seed(seed)
    m = _build_eval_cnn(input_shape, num_classes)
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
    m.fit(x_train, y_train_oh,
          validation_data=(x_val, y_val_oh),
          epochs=int(epochs),
          batch_size=128,
          verbose=0,
          callbacks=cb,
          class_weight=class_weight)
    return m


def _eval_on_test(model: tf.keras.Model, x_test: np.ndarray, y_test_oh: np.ndarray) -> Dict[str, Any]:
    """Evaluate probe model on REAL test set (includes macro precision/recall)."""
    y_true = onehot_to_int(y_test_oh)
    proba = model.predict(x_test, verbose=0)
    y_pred = proba.argmax(axis=1)
    K = int(y_test_oh.shape[1])

    mp, mr, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    out: Dict[str, Any] = {
        "accuracy":            float(accuracy_score(y_true, y_pred)),
        "macro_f1":            float(f1_score(y_true, y_pred, average="macro")),
        "bal_acc":             float(balanced_accuracy_score(y_true, y_pred)),
        "macro_auprc":         _macro_auprc(y_true, proba, K),
        "recall_at_1pct_fpr":  _recall_at_fpr(y_true, proba, K, target_fpr=0.01),
        "ece":                 _ece(proba, y_true, n_bins=15),
        "brier":               _brier_multiclass(np.eye(K)[y_true], proba),

        # Important for downstream gen_precision/gen_recall reporting:
        "macro_precision":     float(mp),
        "macro_recall":        float(mr),
    }

    prec, rec, f1c, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(K)), average=None, zero_division=0
    )
    out["per_class"] = {
        "precision": prec.astype(float).tolist(),
        "recall":    rec.astype(float).tolist(),
        "f1":        f1c.astype(float).tolist(),
        "support":   sup.astype(int).tolist(),
    }
    return out


# ======================================================================================
# 3) High_level evaluator: compute ALL metrics (core)
# ======================================================================================

def compute_all_metrics(
    *,
    img_shape: Tuple[int, int, int],
    x_train_real: np.ndarray, y_train_real: np.ndarray,
    x_val_real:   np.ndarray, y_val_real:   np.ndarray,
    x_test_real:  np.ndarray, y_test_real:  np.ndarray,
    x_synth: Optional[np.ndarray] = None,
    y_synth: Optional[np.ndarray] = None,
    fid_cap_per_class: int = 200,
    seed: int = 42,
    domain_embed_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    epochs: int = 20,
) -> Dict[str, Any]:
    """
    Compute full Phase_1 metric suite.

    Contract:
    - Images may be in [0,255], [-1,1], or [0,1]; normalized via to_01_hwc().
    - Labels may be int (N,) or one_hot (N,K).
    """
    set_global_seed(seed)
    H, W, C = img_shape

    xr = to_01_hwc(x_train_real, img_shape)
    xv = to_01_hwc(x_val_real,   img_shape)
    xt = to_01_hwc(x_test_real,  img_shape)

    # Labels -> one_hot
    if np.asarray(y_train_real).ndim == 1:
        K = int(np.max(y_train_real) + 1)
        yr = np.eye(K)[np.asarray(y_train_real).astype(int)]
        yv = np.eye(K)[np.asarray(y_val_real).astype(int)]
        yt = np.eye(K)[np.asarray(y_test_real).astype(int)]
    else:
        K = int(np.asarray(y_train_real).shape[1])
        yr, yv, yt = np.asarray(y_train_real), np.asarray(y_val_real), np.asarray(y_test_real)

    has_synth = (x_synth is not None) and (y_synth is not None)
    if has_synth:
        xs = to_01_hwc(np.asarray(x_synth), img_shape)
        ys = np.asarray(y_synth)
        ys = ys if ys.ndim == 2 else np.eye(K)[ys.astype(int)]
    else:
        xs, ys = None, None

    # Generative metrics defaults
    fid_macro = None
    cfid_per_class = None
    cfid_macro = None
    kid = None
    js = None
    kl = None
    diversity = None
    fid_domain = None

    # ---- Generative metrics
    if has_synth and xs is not None and ys is not None:
        yv_int = onehot_to_int(yv)
        ys_int = onehot_to_int(ys)

        per_class: List[Optional[float]] = []
        idx_val_cat: List[np.ndarray] = []
        idx_syn_cat: List[np.ndarray] = []

        cap = int(fid_cap_per_class)

        for k in range(int(K)):
            v_idx = np.where(yv_int == k)[0]
            s_idx = np.where(ys_int == k)[0]

            if len(v_idx) == 0 or len(s_idx) == 0:
                per_class.append(None)
                continue

            n = min(len(v_idx), len(s_idx), cap)
            if n <= 1:
                per_class.append(None)
                continue

            rng = np.random.default_rng(int(seed) + int(k))
            v_sel = rng.choice(v_idx, size=n, replace=False)
            s_sel = rng.choice(s_idx, size=n, replace=False)

            idx_val_cat.append(v_sel)
            idx_syn_cat.append(s_sel)

            per_class.append(fid_keras(xv[v_sel], xs[s_sel]))

        valid = [v for v in per_class if v is not None]
        cfid_per_class = per_class
        cfid_macro = float(np.mean(valid)) if valid else None

        if idx_val_cat:
            idx_val_all = np.concatenate(idx_val_cat) if len(idx_val_cat) > 1 else idx_val_cat[0]
            idx_syn_all = np.concatenate(idx_syn_cat) if len(idx_syn_cat) > 1 else idx_syn_cat[0]

            rv = xv[idx_val_all]
            sv = xs[idx_syn_all]

            fid_macro = fid_keras(rv, sv)
            js, kl = js_kl_on_pixels(rv, sv)
            diversity = diversity_score(sv)
            kid = kid_keras(rv, sv, subset=min(200, len(rv), len(sv)), n_subsets=10, seed=seed)

            if domain_embed_fn is not None:
                fid_domain = fid_in_domain_embedding(rv, sv, domain_embed_fn)

    # ---- Class weights (real train)
    ytr_int = onehot_to_int(yr)
    cls_w_arr = compute_class_weight(class_weight="balanced", classes=np.arange(K), y=ytr_int)
    class_weight = {i: float(w) for i, w in enumerate(cls_w_arr)}

    # ---- Utility: real_only
    clf_R = _fit_eval_cnn(
        xr, yr, xv, yv,
        input_shape=(H, W, C),
        num_classes=K,
        class_weight=class_weight,
        seed=seed,
        epochs=epochs,
    )
    util_R = _eval_on_test(clf_R, xt, yt)

    # ---- Utility: real + synth
    if has_synth and xs is not None and ys is not None:
        x_rs = np.concatenate([xr, xs], axis=0)
        y_rs = np.concatenate([yr, ys], axis=0)

        clf_RS = _fit_eval_cnn(
            x_rs, y_rs, xv, yv,
            input_shape=(H, W, C),
            num_classes=K,
            class_weight=None,  # protocol choice
            seed=seed,
            epochs=epochs,
        )
        util_RS = _eval_on_test(clf_RS, xt, yt)
    else:
        util_RS = {
            "accuracy": None,
            "macro_f1": None,
            "macro_precision": None,
            "macro_recall": None,
            "bal_acc": None,
            "macro_auprc": None,
            "recall_at_1pct_fpr": None,
            "ece": None,
            "brier": None,
            "per_class": None,
        }

    return {
        "fid_macro": fid_macro,
        "cfid_macro": cfid_macro,
        "cfid_per_class": cfid_per_class,
        "js": js,
        "kl": kl,
        "diversity": diversity,
        "fid_domain": fid_domain,
        "kid": kid,

        "real_only": util_R,
        "real_plus_synth": util_RS,
    }


# ======================================================================================
# 4) Summary shaping (stable schema + legacy shim)
# ======================================================================================

def _map_util_names(util_block: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize utility naming into stable schema."""
    if not isinstance(util_block, dict):
        return {}
    return {
        "accuracy":           util_block.get("accuracy"),
        "macro_f1":           util_block.get("macro_f1"),
        "macro_precision":    util_block.get("macro_precision"),
        "macro_recall":       util_block.get("macro_recall"),
        "balanced_accuracy":  util_block.get("bal_acc"),
        "macro_auprc":        util_block.get("macro_auprc"),
        "recall_at_1pct_fpr": util_block.get("recall_at_1pct_fpr"),
        "ece":                util_block.get("ece"),
        "brier":              util_block.get("brier"),
        "per_class":          util_block.get("per_class"),
    }


def evaluate_model_suite(
    *,
    model_name: str,
    img_shape: Tuple[int, int, int],
    x_train_real: np.ndarray, y_train_real: np.ndarray,
    x_val_real:   np.ndarray, y_val_real:   np.ndarray,
    x_test_real:  np.ndarray, y_test_real:  np.ndarray,
    x_synth: Optional[np.ndarray] = None,
    y_synth: Optional[np.ndarray] = None,
    per_class_cap_for_fid: int = 200,
    seed: int = 42,
    epochs: int = 20,
) -> Dict[str, Any]:
    """Compute + map metrics into stable schema and legacy flattened keys."""
    metrics = compute_all_metrics(
        img_shape=img_shape,
        x_train_real=x_train_real, y_train_real=y_train_real,
        x_val_real=x_val_real,     y_val_real=y_val_real,
        x_test_real=x_test_real,   y_test_real=y_test_real,
        x_synth=x_synth,           y_synth=y_synth,
        fid_cap_per_class=per_class_cap_for_fid,
        seed=seed,
        epochs=epochs,
    )

    counts = {
        "train_real": int(len(x_train_real)),
        "val_real":   int(len(x_val_real)),
        "test_real":  int(len(x_test_real)),
        "synthetic":  (int(len(x_synth)) if x_synth is not None else None),
    }

    generative = {
        "fid":            metrics.get("fid_macro"),
        "fid_macro":      metrics.get("fid_macro"),
        "cfid_macro":     metrics.get("cfid_macro"),
        "js":             metrics.get("js"),
        "kl":             metrics.get("kl"),
        "diversity":      metrics.get("diversity"),
        "cfid_per_class": metrics.get("cfid_per_class"),
        "fid_domain":     metrics.get("fid_domain"),
        "kid":            metrics.get("kid"),
    }

    util_R  = _map_util_names(metrics.get("real_only") or {})
    util_RS = _map_util_names(metrics.get("real_plus_synth") or {})

    def _delta(k: str) -> Optional[float]:
        a = util_RS.get(k)
        b = util_R.get(k)
        return None if (a is None or b is None) else float(a - b)

    deltas = {
        "accuracy":           _delta("accuracy"),
        "macro_f1":           _delta("macro_f1"),
        "macro_precision":    _delta("macro_precision"),
        "macro_recall":       _delta("macro_recall"),
        "balanced_accuracy":  _delta("balanced_accuracy"),
        "macro_auprc":        _delta("macro_auprc"),
        "recall_at_1pct_fpr": _delta("recall_at_1pct_fpr"),
        "ece":                _delta("ece"),
        "brier":              _delta("brier"),
    }

    legacy_metrics = {
        "metrics.cfid":                   generative.get("cfid_macro"),
        "metrics.cfid_macro":             generative.get("cfid_macro"),
        "metrics.fid_macro":              generative.get("fid_macro"),
        "metrics.kid":                    generative.get("kid"),
        "metrics.ms_ssim":                generative.get("diversity"),
        "metrics.downstream.macro_f1":    util_RS.get("macro_f1"),
        "counts.num_real":                counts["train_real"],
        "counts.num_fake":                counts["synthetic"],

        # Recommended meaning: downstream macro P/R on REAL TEST for Real+Synth model
        "metrics.gen_precision":          util_RS.get("macro_precision"),
        "metrics.gen_recall":             util_RS.get("macro_recall"),
        "metrics.downstream.precision":   util_RS.get("macro_precision"),
        "metrics.downstream.recall":      util_RS.get("macro_recall"),
    }

    return {
        "model": model_name,
        "seed":  int(seed),

        "images": counts,
        "generative": generative,
        "utility_real_only": util_R,
        "utility_real_plus_synth": util_RS,
        "deltas_RS_minus_R": deltas,

        "metrics_meta": {
            "computed_at": "runtime",
            "metrics_version": "phase1_runtime_v2",
            "kid_mode": "inception_poly_mmd_v1",
            "gen_pr_mode": "downstream_macro_precision_recall",
        },

        **legacy_metrics,
    }


# ======================================================================================
# 5) Writer: dataset_aware outputs + JSONL append
# ======================================================================================

def write_summary(
    *,
    record: Dict[str, Any],
    artifacts_root: str,
    dataset_id: str,
    model_tag: str,
    run_id: str,
) -> Tuple[Path, Path]:
    """
    Write one evaluation record to:
      - summary.txt (human readable)
      - summary.jsonl (append one JSON object per run)

    This is the preferred writer API for the new repo.
    """
    # Use central path policy if available; otherwise use module fallback.
    if resolve_eval_paths is not None:
        paths = resolve_eval_paths(
            artifacts_root=artifacts_root,
            dataset_id=dataset_id,
            model_tag=model_tag,
            run_id=run_id,
        )
        json_path = paths.jsonl_path
        txt_path = paths.console_path
    else:
        paths = resolve_eval_output_paths(
            artifacts_root=artifacts_root,
            dataset_id=dataset_id,
            model_tag=model_tag,
            run_id=run_id,
        )
        json_path = paths.jsonl_path
        txt_path = paths.console_path

    json_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a readable console block (stable, quick to skim in Slurm logs)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        f"[{ts}] model_tag={model_tag} run_id={run_id} dataset_id={dataset_id}",
        f"  outputs: jsonl={json_path}  console={txt_path}",
    ]

    gen = (record.get("generative") or {})
    util_rs = (record.get("utility_real_plus_synth") or {})

    if gen.get("fid") is not None:
        lines.append(f"  FID: {gen['fid']:.4f}")
    if gen.get("cfid_macro") is not None:
        lines.append(f"  cFID (macro): {gen['cfid_macro']:.4f}")
    if gen.get("kid") is not None:
        lines.append(f"  KID: {gen['kid']:.6f}")
    if gen.get("diversity") is not None:
        lines.append(f"  Diversity: {gen['diversity']:.4f}")
    if gen.get("js") is not None or gen.get("kl") is not None:
        lines.append(f"  JS: {gen.get('js')} | KL: {gen.get('kl')}")
    if util_rs.get("macro_f1") is not None:
        lines.append(f"  Macro_F1 (R+S): {util_rs['macro_f1']:.4f}")
    if util_rs.get("macro_precision") is not None and util_rs.get("macro_recall") is not None:
        lines.append(
            f"  Macro_P/R (R+S): {util_rs['macro_precision']:.4f} / {util_rs['macro_recall']:.4f}"
        )
    lines.append("")

    # Prefer shared IO helpers if present
    if write_text is not None:
        write_text(txt_path, "\n".join(lines))
    else:
        txt_path.write_text("\n".join(lines))

    if append_jsonl is not None:
        append_jsonl(json_path, record)
    else:
        import json
        with open(json_path, "a") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")

    return json_path, txt_path
