# src/gencysynth/models/maskedautoflow/variants/maf_affine/train.py

# =============================================================================
# GenCyberSynth — MaskedAutoFlow — maf_affine — Training (Rule A)
#
# Training utilities for a Masked Autoregressive Flow (MAF) density model.
#
# Rule A highlights
# -----------------
# Dataset-aware artifacts (scalable across many datasets):
#    <paths.artifacts>/<data.name>/<model.family>/<model.variant>/{checkpoints,summaries,synthetic}
#
# Backward compatible config:
#    - NEW keys: data.*, model.*, train.*, paths.*
#    - LEGACY keys: IMG_SHAPE, EPOCHS, BATCH_SIZE, DATA_DIR, etc.
#
# Clean I/O:
#    - reads data from `data.root` (or DATA_DIR)
#    - writes weights + TB logs only into Rule-A artifact paths
#
# Notes
# -----
# - MAF is unconditional here (labels ignored) and models flattened images.
# - Inputs are float32 in [0,1], shape (N, D).
# - Optional dequantization noise helps continuous likelihood on near-binary inputs.
# - Checkpoints are Keras 3 weight files: *.weights.h5
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Callable

import numpy as np
import tensorflow as tf

# Preferred shared loader (if present in your repo)
try:
    from common.data import load_dataset_npy  # type: ignore
except Exception:
    load_dataset_npy = None  # fallback used below

from maskedautoflow.models import (
    MAF,
    MAFConfig,
    build_maf_model,
    flatten_images,
)

# GPU memory growth (safe no-op on CPU)
for d in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(d, True)
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Small config helpers
# -----------------------------------------------------------------------------
def _cfg_get(cfg: Dict[str, Any], dotted: str, default=None):
    """Read nested dict values using dot notation (e.g., 'train.epochs')."""
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _coerce_cfg(cfg_or_argv):
    """
    Accept either:
      - dict: already-parsed config
      - argv list/tuple: e.g. ['--config','path/to.yaml']
    Returns a Python dict.
    """
    if isinstance(cfg_or_argv, dict):
        return dict(cfg_or_argv)

    if isinstance(cfg_or_argv, (list, tuple)):
        import yaml

        cfg_path = None
        if "--config" in cfg_or_argv:
            i = cfg_or_argv.index("--config")
            if i + 1 < len(cfg_or_argv):
                cfg_path = Path(cfg_or_argv[i + 1])

        if cfg_path is None:
            cfg_path = Path("configs/config.yaml")

        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    raise TypeError(f"Unsupported config payload type: {type(cfg_or_argv)}")


# -----------------------------------------------------------------------------
# Rule A: dataset-aware artifact paths
# -----------------------------------------------------------------------------
def _resolve_artifact_paths(cfg: Dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    """
    Resolve artifact directories following Rule A.

    Canonical layout:
      <paths.artifacts>/<data.name>/<model.family>/<model.variant>/
        checkpoints/
        summaries/
          tb/
        synthetic/          # reserved (sampling module writes here)

    Overrides (optional, advanced):
      artifacts.checkpoints / artifacts.summaries / artifacts.synthetic
    """
    artifacts_root = Path(_cfg_get(cfg, "paths.artifacts", _cfg_get(cfg, "PATHS.artifacts", "artifacts")))

    # Dataset identity (required for Rule A; fall back to something stable if missing)
    dataset_name = str(
        _cfg_get(cfg, "data.name", _cfg_get(cfg, "dataset", _cfg_get(cfg, "DATASET", "dataset")))
    )

    # Model identity (family/variant)
    family = str(_cfg_get(cfg, "model.family", "maskedautoflow"))
    variant = str(_cfg_get(cfg, "model.variant", "maf_affine"))

    base = artifacts_root / dataset_name / family / variant

    ckpt_dir = Path(_cfg_get(cfg, "artifacts.checkpoints", base / "checkpoints"))
    sums_dir = Path(_cfg_get(cfg, "artifacts.summaries", base / "summaries"))
    synth_dir = Path(_cfg_get(cfg, "artifacts.synthetic", base / "synthetic"))

    # Keep TensorBoard under summaries/tb to match other families
    tb_dir = sums_dir / "tb"
    return ckpt_dir, synth_dir, sums_dir, tb_dir


# -----------------------------------------------------------------------------
# TrainConfig (defaults + readable knobs)
# -----------------------------------------------------------------------------
@dataclass
class TrainConfig:
    # Shapes
    IMG_SHAPE: Tuple[int, int, int] = (40, 40, 1)

    # Input pipeline
    BATCH_SIZE: int = 256

    # Optimization
    EPOCHS: int = 50
    LR: float = 2e-4
    PATIENCE: int = 10
    CLIP_GRAD: float = 1.0
    SEED: int = 42

    # Model
    NUM_FLOWS: int = 5
    HIDDEN_DIMS: Tuple[int, ...] = (128, 128)

    # Dequantization (recommended when inputs are 0/1-ish)
    DEQUANT_NOISE: bool = True
    DEQUANT_EPS: float = 1.0 / 256.0  # uniform noise U(0, eps)


# -----------------------------------------------------------------------------
# Dataset preparation
# -----------------------------------------------------------------------------
def _to_float01(x: np.ndarray) -> np.ndarray:
    """Cast to float32 and map byte-like inputs to [0,1]."""
    x = np.asarray(x).astype("float32", copy=False)
    if float(np.nanmax(x)) > 1.5:  # looks like 0..255
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


def build_datasets(
    x_train: np.ndarray,
    x_val: np.ndarray,
    cfg: TrainConfig,
    *,
    shuffle: bool = True,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Build tf.data pipelines yielding flattened vectors (B, D) in [0,1].

    MAF is unconditional here, so labels are intentionally ignored.
    """
    H, W, C = cfg.IMG_SHAPE

    # 1) Normalize to float32 in [0,1]
    Xtr = _to_float01(x_train)
    Xva = _to_float01(x_val)

    # 2) Optional dequantization noise (keeps values away from exact 0/1)
    if cfg.DEQUANT_NOISE and cfg.DEQUANT_EPS > 0.0:
        rng = np.random.default_rng(int(cfg.SEED))
        Xtr = Xtr + rng.uniform(0.0, cfg.DEQUANT_EPS, size=Xtr.shape).astype("float32")
        Xva = Xva + rng.uniform(0.0, cfg.DEQUANT_EPS, size=Xva.shape).astype("float32")
        Xtr = np.clip(Xtr, 1e-6, 1.0 - 1e-6)
        Xva = np.clip(Xva, 1e-6, 1.0 - 1e-6)

    # 3) Flatten to (N, D)
    Xtr = flatten_images(Xtr, (H, W, C), assume_01=True, clip=True)
    Xva = flatten_images(Xva, (H, W, C), assume_01=True, clip=True)

    def _make(x: np.ndarray, do_shuffle: bool) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices(x.astype("float32", copy=False))
        if do_shuffle:
            ds = ds.shuffle(
                buffer_size=min(10_000, int(x.shape[0])),
                seed=int(cfg.SEED),
                reshuffle_each_iteration=True,
            )
        return ds.batch(int(cfg.BATCH_SIZE)).prefetch(tf.data.AUTOTUNE)

    return _make(Xtr, shuffle), _make(Xva, False)


def _load_train_val_split(cfg: Dict[str, Any], img_shape: Tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Load (x_train, x_val) for unconditional density modeling.

    Priority
    --------
    1) shared loader: common.data.load_dataset_npy(data_root, img_shape, num_classes, ...)
       (we ignore labels; we just want consistent split behavior)
    2) fallback: expects train_data.npy + test_data.npy in data.root (or DATA_DIR)
       and uses test_data as the source for validation slices.
    """
    data_root = Path(_cfg_get(cfg, "data.root", _cfg_get(cfg, "DATA_DIR", "data"))).expanduser()
    val_frac = float(_cfg_get(cfg, "train.val_fraction", _cfg_get(cfg, "VAL_FRACTION", 0.5)))
    num_classes = int(_cfg_get(cfg, "model.num_classes", _cfg_get(cfg, "NUM_CLASSES", _cfg_get(cfg, "num_classes", 1))))

    if load_dataset_npy is not None:
        # load_dataset_npy returns:
        #   x_train, y_train, x_val, y_val, x_test, y_test
        x_tr, _y_tr, x_va, _y_va, _x_te, _y_te = load_dataset_npy(
            data_root, img_shape, num_classes, val_fraction=val_frac
        )
        return x_tr, x_va

    # ---- minimal fallback loader (expects 2 files) ----
    x_tr = np.load(data_root / "train_data.npy").astype("float32")
    x_te = np.load(data_root / "test_data.npy").astype("float32")

    if float(np.nanmax(x_tr)) > 1.5:
        x_tr = x_tr / 255.0
        x_te = x_te / 255.0

    H, W, C = img_shape
    x_tr = x_tr.reshape((-1, H, W, C))
    x_te = x_te.reshape((-1, H, W, C))

    n_val = int(len(x_te) * val_frac)
    x_va = x_te[:n_val]
    return x_tr, x_va


# -----------------------------------------------------------------------------
# Model building
# -----------------------------------------------------------------------------
def build_model(cfg: TrainConfig) -> MAF:
    """
    Build the MAF and create variables once (Keras needs an initial forward pass).
    """
    model = build_maf_model(
        MAFConfig(
            IMG_SHAPE=cfg.IMG_SHAPE,
            NUM_FLOWS=cfg.NUM_FLOWS,
            HIDDEN_DIMS=cfg.HIDDEN_DIMS,
        )
    )
    H, W, C = cfg.IMG_SHAPE
    D = H * W * C
    _ = model(tf.zeros((1, D), dtype=tf.float32))  # variable creation
    return model


# -----------------------------------------------------------------------------
# Training loop (NLL + early stopping)
# -----------------------------------------------------------------------------
def train_maf_model(
    model: MAF,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    *,
    cfg: TrainConfig,
    ckpt_dir: Path,
    writer: Optional[tf.summary.SummaryWriter] = None,
    log_cb: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, Any]:
    """
    Train a MAF by minimizing negative log-likelihood (NLL).

    Writes
    ------
    ckpt_dir/
      MAF_best.weights.h5
      MAF_last.weights.h5
      MAF_LAST_OK
    summaries/tb/
      TensorBoard scalars (if writer is provided)

    Returns
    -------
    dict: best_path, last_path, best_val, epochs_run
    """
    tf.keras.utils.set_random_seed(int(cfg.SEED))

    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_path = ckpt_dir / "MAF_best.weights.h5"
    last_path = ckpt_dir / "MAF_last.weights.h5"

    opt = tf.keras.optimizers.Adam(learning_rate=float(cfg.LR))

    @tf.function(reduce_retracing=True)
    def train_step(x: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            nll = -tf.reduce_mean(model.log_prob(x))
        grads = tape.gradient(nll, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, float(cfg.CLIP_GRAD))
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return nll

    @tf.function(reduce_retracing=True)
    def val_step(x: tf.Tensor) -> tf.Tensor:
        return -tf.reduce_mean(model.log_prob(x))

    def _log(stage: str, msg: str) -> None:
        if log_cb is not None:
            try:
                log_cb(stage, msg)
                return
            except Exception:
                pass
        print(f"[{stage}] {msg}")

    best_val = float("inf")
    bad_epochs = 0
    epochs_run = 0

    for epoch in range(1, int(cfg.EPOCHS) + 1):
        epochs_run = epoch
        tr_mean = tf.keras.metrics.Mean()
        va_mean = tf.keras.metrics.Mean()

        # ---- Train ----
        for xb in train_ds:
            tr_mean.update_state(train_step(xb))

        # ---- Validate ----
        for xb in val_ds:
            va_mean.update_state(val_step(xb))

        tr = float(tr_mean.result().numpy())
        va = float(va_mean.result().numpy())

        if writer is not None:
            with writer.as_default():
                tf.summary.scalar("Loss/train_nll", tr, step=epoch)
                tf.summary.scalar("Loss/val_nll", va, step=epoch)
                writer.flush()

        _log("train", f"epoch {epoch:03d}: train_nll={tr:.4f} | val_nll={va:.4f}")

        # Save "last" every epoch (cheap + robust for crash recovery)
        model.save_weights(str(last_path), overwrite=True)

        # Best checkpoint + early stopping
        if va < best_val - 1e-6:
            best_val = va
            bad_epochs = 0
            model.save_weights(str(best_path), overwrite=True)
            _log("ckpt", f"saved {best_path.name}")
        else:
            bad_epochs += 1
            _log("train", f"patience {bad_epochs}/{int(cfg.PATIENCE)}")
            if bad_epochs >= int(cfg.PATIENCE):
                _log("train", "early stopping")
                break

    (ckpt_dir / "MAF_LAST_OK").write_text("ok", encoding="utf-8")

    return {
        "best_path": str(best_path),
        "last_path": str(last_path),
        "best_val": float(best_val),
        "epochs_run": int(epochs_run),
    }


# -----------------------------------------------------------------------------
# Unified-CLI adapter (Rule A + legacy support)
# -----------------------------------------------------------------------------
def _train_from_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unified entrypoint used by `gencs train`.

    Reads (new → legacy fallback)
    -----------------------------
    model.img_shape  → IMG_SHAPE / img.shape
    train.*          → EPOCHS/BATCH_SIZE/LR/PATIENCE/CLIP_GRAD/SEED/...
    data.root        → DATA_DIR

    Writes (Rule A)
    ---------------
    <artifacts>/<data.name>/maskedautoflow/maf_affine/checkpoints/
    <artifacts>/<data.name>/maskedautoflow/maf_affine/summaries/tb/
    """
    # ---- Shapes (prefer NEW keys; fall back to LEGACY keys) ----
    H, W, C = tuple(
        _cfg_get(cfg, "model.img_shape", _cfg_get(cfg, "IMG_SHAPE", _cfg_get(cfg, "img.shape", (40, 40, 1))))
    )

    # ---- Training knobs ----
    bs = int(_cfg_get(cfg, "train.batch_size", _cfg_get(cfg, "BATCH_SIZE", 256)))
    epochs = int(_cfg_get(cfg, "train.epochs", _cfg_get(cfg, "EPOCHS", 50)))
    lr = float(_cfg_get(cfg, "train.lr", _cfg_get(cfg, "LR", 2e-4)))
    patience = int(_cfg_get(cfg, "train.patience", _cfg_get(cfg, "PATIENCE", 10)))
    clip = float(_cfg_get(cfg, "train.clip_grad", _cfg_get(cfg, "CLIP_GRAD", 1.0)))
    seed = int(_cfg_get(cfg, "train.seed", _cfg_get(cfg, "SEED", 42)))

    # ---- Model knobs ----
    num_flows = int(_cfg_get(cfg, "model.num_flows", _cfg_get(cfg, "NUM_FLOWS", 5)))
    hidden_dims = _cfg_get(cfg, "model.hidden_dims", _cfg_get(cfg, "HIDDEN_DIMS", (128, 128)))
    hidden = tuple(int(h) for h in hidden_dims)

    # ---- Dequantization knobs ----
    deq = bool(_cfg_get(cfg, "train.dequant_noise", _cfg_get(cfg, "DEQUANT_NOISE", True)))
    deq_eps = float(_cfg_get(cfg, "train.dequant_eps", _cfg_get(cfg, "DEQUANT_EPS", 1.0 / 256.0)))

    # ---- Artifacts (Rule A) ----
    ckpt_dir, _synth_dir, sums_dir, tb_dir = _resolve_artifact_paths(cfg)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sums_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build TrainConfig ----
    tcfg = TrainConfig(
        IMG_SHAPE=(H, W, C),
        BATCH_SIZE=bs,
        EPOCHS=epochs,
        LR=lr,
        PATIENCE=patience,
        CLIP_GRAD=clip,
        SEED=seed,
        NUM_FLOWS=num_flows,
        HIDDEN_DIMS=hidden,
        DEQUANT_NOISE=deq,
        DEQUANT_EPS=deq_eps,
    )

    # ---- Load data ----
    x_train, x_val = _load_train_val_split(cfg, (H, W, C))

    # ---- Build tf.data ----
    train_ds, val_ds = build_datasets(x_train, x_val, tcfg, shuffle=True)

    # ---- Build model ----
    model = build_model(tcfg)

    # ---- TensorBoard writer ----
    writer = tf.summary.create_file_writer(str(tb_dir))

    # ---- Print a clear run header (helpful on HPC logs) ----
    dataset_name = str(_cfg_get(cfg, "data.name", _cfg_get(cfg, "DATASET", "dataset")))
    print(
        f"[maf_affine] dataset={dataset_name} | HWC={(H, W, C)} | "
        f"bs={bs} epochs={epochs} lr={lr} seed={seed} | flows={num_flows} hidden={hidden}"
    )
    print(f"[paths] ckpt_dir={ckpt_dir.resolve()}")
    print(f"[paths] summaries={sums_dir.resolve()} (tb={tb_dir.resolve()})")

    # ---- Train ----
    summary = train_maf_model(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        cfg=tcfg,
        ckpt_dir=ckpt_dir,
        writer=writer,
        log_cb=None,
    )

    print(
        f"[maf_affine] done | best_val={summary['best_val']:.4f} | "
        f"epochs_run={summary['epochs_run']} | best={Path(summary['best_path']).name}"
    )

    # A small completion marker under summaries (consistent with other families)
    (sums_dir / "train_done.txt").write_text("ok", encoding="utf-8")

    return summary


def train(cfg_or_argv) -> Dict[str, Any]:
    """Unified-CLI callable: train(dict) or train(['--config','...'])."""
    cfg = _coerce_cfg(cfg_or_argv)
    return _train_from_cfg(cfg)


__all__ = [
    "TrainConfig",
    "build_datasets",
    "build_model",
    "train_maf_model",
    "train",
]


# -----------------------------------------------------------------------------
# Optional local CLI entrypoint:
#   python -m gencysynth.models.maskedautoflow.variants.maf_affine.train --config path/to.yaml
# -----------------------------------------------------------------------------
def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(description="Train Masked Autoregressive Flow (MAF) — maf_affine")
    p.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    args = p.parse_args(argv)

    # Route through the same unified training path
    _train_from_cfg(_coerce_cfg(["--config", args.config]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
