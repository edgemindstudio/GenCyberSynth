# src/gencysynth/models/diffusion/variants/c-ddpm/train.py
"""
GenCyberSynth — Diffusion family — c-DDPM variant (Conditional) — Training
========================================================================

RULE A (Scalable artifact policy)
---------------------------------
Everything is keyed by:
    (dataset_id, model_tag, run_id)

This training module MUST write ONLY under the resolved run context:

  artifacts/
    runs/<dataset_id>/<model_tag>/<run_id>/
      checkpoints/           # weights + best metrics json (if any)
      samples/               # (optional) preview artifacts produced during training
      tensorboard/           # event files
      run_meta_snapshot.json # config snapshot for audit/debug

and logs MUST go to:

  artifacts/logs/<dataset_id>/<model_tag>/<run_id>/
    run.log

This module MUST NOT invent ad-hoc directories like:
  artifacts/diffusion/checkpoints
  artifacts/models/...
  artifacts/runs/<run_id>/...

Why this matters
----------------
We will run:
- multiple datasets
- multiple families and variants
- multiple seeds and configs in parallel on HPC job arrays

So *every* read/write must be dataset-aware and run-aware.

What this module does
---------------------
- Resolves RunContext (dataset_id, model_tag, run_id, seed) via orchestration.
- Loads the dataset via the shared loader (common.data.load_dataset_npy).
- Builds the c-DDPM model (via diffusion.models.build_diffusion_model).
- Trains ε-prediction objective with early stopping.
- Writes checkpoints + tensorboard logs under the run directory.

What this module does NOT do
----------------------------
- Sampling/synthesis (belongs in variants/c-ddpm/sample.py)
- Evaluation metrics/reporting (belongs under gencysynth/eval or reporting/)
- Any global artifact layouts outside Rule A

CLI usage
---------
python -m gencysynth.models.diffusion.variants.c-ddpm.train --config configs/<...>.yaml

Or via orchestration adapter:
train([...]) or train({dict_config})
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import tensorflow as tf

from common.data import load_dataset_npy  # shared loader (expects images in [0,1])
# NOTE: model builder imported lazily inside _train_from_cfg to keep import-safe.

from gencysynth.orchestration.context import resolve_run_context
from gencysynth.orchestration.logger import get_run_logger
from gencysynth.utils.io import write_json
from gencysynth.utils.paths import ensure_dir
from gencysynth.utils.reproducibility import now_iso


# -----------------------------------------------------------------------------
# Variant identity (consistent with registry + folder name)
# -----------------------------------------------------------------------------
FAMILY: str = "diffusion"
VARIANT: str = "c-ddpm"
MODEL_TAG: str = "diffusion/c-ddpm"


# =============================================================================
# Small config helpers
# =============================================================================
def _cfg_get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    """
    Read nested config values using a dotted key, e.g. "paths.artifacts".

    Returns `default` if a key is missing or an intermediate is not a dict.
    """
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _set_seeds(seed: int) -> None:
    """Best-effort determinism for numpy + TF."""
    np.random.seed(int(seed))
    tf.keras.utils.set_random_seed(int(seed))


def _enable_gpu_mem_growth() -> None:
    """
    Avoid TF reserving all VRAM on multi-tenant GPUs.
    Safe no-op on CPU-only nodes.
    """
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass


def _snapshot_run_meta(cfg: Dict[str, Any], run_dir: Path) -> Path:
    """
    Write an audit snapshot of the effective config into the *run directory*.

    Location:
      artifacts/runs/<dataset_id>/<model_tag>/<run_id>/run_meta_snapshot.json
    """
    out = run_dir / "run_meta_snapshot.json"
    payload = {
        "timestamp": now_iso(),
        "model_tag": _cfg_get(cfg, "model.tag", None),
        "dataset_id": _cfg_get(cfg, "dataset.id", None),
        "run_meta": cfg.get("run_meta", {}),
        "paths": cfg.get("paths", {}),
        "cfg": cfg,  # full snapshot
    }
    write_json(out, payload, indent=2, sort_keys=True, atomic=True)
    return out


# =============================================================================
# Noise schedule (ᾱ_t)
# =============================================================================
def build_alpha_hat(
    T: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    schedule: str = "linear",
) -> np.ndarray:
    """
    Build ᾱ (alpha-hat) schedule used by DDPM-style training.

    Parameters
    ----------
    T : int
        Number of diffusion timesteps.
    beta_start, beta_end : float
        Range for β_t.
    schedule : {"linear", "cosine"}
        Supported schedules.

    Returns
    -------
    np.ndarray
        ᾱ_t array of shape (T,), dtype float32 in (0, 1].
    """
    schedule = (schedule or "linear").lower()

    if schedule == "linear":
        betas = np.linspace(beta_start, beta_end, T, dtype=np.float32)
        alphas = 1.0 - betas
        alpha_hat = np.cumprod(alphas, axis=0)
        return alpha_hat.astype("float32")

    if schedule == "cosine":
        # Nichol & Dhariwal cosine schedule (commonly used in DDPM variants)
        s = 0.008
        steps = np.arange(T + 1, dtype=np.float32)
        f = lambda t: np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
        alpha_bar = f(steps) / f(0)
        # convert to incremental alphas then cumulative product again (stabilize)
        alpha_hat = alpha_bar[1:] / alpha_bar[:-1]
        alpha_hat = np.clip(alpha_hat, 1e-5, 1.0)
        return np.cumprod(alpha_hat).astype("float32")

    raise ValueError(f"Unsupported schedule '{schedule}' (use 'linear' or 'cosine')")


# =============================================================================
# Training primitives (tf.function)
# =============================================================================
mse_loss = tf.keras.losses.MeanSquaredError()


@tf.function
def train_step(
    model: tf.keras.Model,
    x0: tf.Tensor,                # clean image batch in [0,1], shape (B,H,W,C)
    y_onehot: tf.Tensor,          # one-hot labels, shape (B,K)
    t_vec: tf.Tensor,             # integer timesteps, shape (B,)
    alpha_hat_tf: tf.Tensor,      # ᾱ_t tensor, shape (T,)
    optimizer: tf.keras.optimizers.Optimizer,
) -> tf.Tensor:
    """
    One optimization step for ε-prediction objective.

    Loss = MSE(ε, ε̂(x_t, t, y)), where:
      x_t = sqrt(ᾱ_t) * x0 + sqrt(1 - ᾱ_t) * ε
    """
    noise = tf.random.normal(shape=tf.shape(x0))

    sqrt_a = tf.sqrt(tf.gather(alpha_hat_tf, t_vec))          # (B,)
    sqrt_oma = tf.sqrt(1.0 - tf.gather(alpha_hat_tf, t_vec))  # (B,)
    sqrt_a = tf.reshape(sqrt_a, (-1, 1, 1, 1))
    sqrt_oma = tf.reshape(sqrt_oma, (-1, 1, 1, 1))

    x_t = sqrt_a * x0 + sqrt_oma * noise

    with tf.GradientTape() as tape:
        eps_pred = model([x_t, y_onehot, t_vec], training=True)
        loss = mse_loss(noise, eps_pred)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


@tf.function
def eval_step(
    model: tf.keras.Model,
    x0: tf.Tensor,
    y_onehot: tf.Tensor,
    t_vec: tf.Tensor,
    alpha_hat_tf: tf.Tensor,
) -> tf.Tensor:
    """Forward-only ε-prediction loss (no gradients)."""
    noise = tf.random.normal(shape=tf.shape(x0))

    sqrt_a = tf.sqrt(tf.gather(alpha_hat_tf, t_vec))
    sqrt_oma = tf.sqrt(1.0 - tf.gather(alpha_hat_tf, t_vec))
    sqrt_a = tf.reshape(sqrt_a, (-1, 1, 1, 1))
    sqrt_oma = tf.reshape(sqrt_oma, (-1, 1, 1, 1))

    x_t = sqrt_a * x0 + sqrt_oma * noise
    eps_pred = model([x_t, y_onehot, t_vec], training=False)
    return mse_loss(noise, eps_pred)


# =============================================================================
# Dataset / batching utilities
# =============================================================================
def _as_float(x: Any) -> float:
    """Convert TF tensors / python scalars into a plain float."""
    try:
        return float(x)
    except Exception:
        arr = tf.convert_to_tensor(x)
        return float(arr.numpy().reshape(-1)[0])


def _make_batches(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """
    Create a tf.data pipeline. Uses a bounded shuffle buffer for practicality.
    """
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(x), 10_000))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# =============================================================================
# Core training loop (pure training logic, path injected from Rule A context)
# =============================================================================
def train_diffusion(
    *,
    model: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epochs: int = 200,
    batch_size: int = 128,
    learning_rate: float = 2e-4,
    beta_1: float = 0.9,
    alpha_hat: Optional[np.ndarray] = None,
    T: int = 1000,
    patience: int = 10,
    ckpt_dir: Path,
    log_cb=None,  # Optional callable(epoch:int, train_loss:float, val_loss:Optional[float])
) -> Dict[str, float]:
    """
    Train c-DDPM with ε-prediction objective, early stopping, and checkpoints.

    IMPORTANT:
    - This function is intentionally "path-agnostic" except for ckpt_dir,
      which MUST be provided by the Rule A run context.
    """
    ensure_dir(ckpt_dir)

    if alpha_hat is None:
        alpha_hat = build_alpha_hat(T)
    alpha_hat_tf = tf.constant(alpha_hat, dtype=tf.float32)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1)

    train_ds = _make_batches(x_train, y_train, batch_size=batch_size, shuffle=True)
    val_ds = None
    if x_val is not None and y_val is not None:
        val_ds = _make_batches(x_val, y_val, batch_size=batch_size, shuffle=False)

    best_val = float("inf")
    patience_ctr = 0
    last_train_loss = float("nan")
    last_val_loss: Optional[float] = None

    # Sample random t per batch (tf.function-safe)
    def _sample_t(n: tf.Tensor) -> tf.Tensor:
        T_dyn = tf.shape(alpha_hat_tf)[0]
        return tf.random.uniform(shape=(n,), minval=0, maxval=T_dyn, dtype=tf.int32)

    for epoch in range(1, int(epochs) + 1):
        # -------------------------
        # Training
        # -------------------------
        train_losses = []
        for xb, yb in train_ds:
            bsz = tf.shape(xb)[0]
            t_vec = _sample_t(bsz)
            loss = train_step(model, xb, yb, t_vec, alpha_hat_tf, optimizer)
            train_losses.append(loss)

        last_train_loss = _as_float(tf.reduce_mean(train_losses))

        # -------------------------
        # Validation
        # -------------------------
        last_val_loss = None
        if val_ds is not None:
            val_losses = []
            for xb, yb in val_ds:
                bsz = tf.shape(xb)[0]
                t_vec = _sample_t(bsz)
                vloss = eval_step(model, xb, yb, t_vec, alpha_hat_tf)
                val_losses.append(vloss)
            last_val_loss = _as_float(tf.reduce_mean(val_losses))

        # -------------------------
        # Optional logging callback (TensorBoard, console, etc.)
        # -------------------------
        if log_cb is not None:
            log_cb(epoch, last_train_loss, last_val_loss)

        # -------------------------
        # Checkpoints (Keras 3 friendly: *.weights.h5)
        # -------------------------
        # Periodic snapshot (and epoch 1 for quick sanity check)
        if epoch == 1 or epoch % 25 == 0:
            model.save_weights(str(ckpt_dir / f"DDPM_epoch_{epoch:04d}.weights.h5"))
            model.save_weights(str(ckpt_dir / f"DIFF_epoch_{epoch:04d}.weights.h5"))  # alias (legacy convenience)

        # Always update "last"
        model.save_weights(str(ckpt_dir / "DDPM_last.weights.h5"))
        model.save_weights(str(ckpt_dir / "DIFF_last.weights.h5"))  # alias

        # Best model tracking (prefer val if available, else train)
        score = last_val_loss if last_val_loss is not None else last_train_loss
        if score < best_val:
            best_val = float(score)
            patience_ctr = 0
            model.save_weights(str(ckpt_dir / "DDPM_best.weights.h5"))
            model.save_weights(str(ckpt_dir / "DIFF_best.weights.h5"))  # alias
        else:
            patience_ctr += 1
            if patience_ctr >= int(patience):
                break

    return {
        "train_loss": float(last_train_loss),
        "val_loss": float(last_val_loss) if last_val_loss is not None else float("nan"),
        "best_val": float(best_val),
        "stopped_early": float(patience_ctr >= int(patience)),
        "epochs_ran": float(epoch),
    }


# =============================================================================
# Rule A adapter: config -> RunContext -> training
# =============================================================================
def _coerce_cfg(cfg_or_argv) -> Dict[str, Any]:
    """
    Accept either:
      - dict config
      - argv-like list/tuple containing optional: --config <path>

    Returns a Python dict config.
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

        return yaml.safe_load(cfg_path.read_text()) or {}

    raise TypeError(f"Unsupported config payload type: {type(cfg_or_argv)}")


def _train_from_cfg(cfg: Dict[str, Any]) -> Dict[str, float]:
    """
    End-to-end training driver that enforces Rule A.

    Responsibilities:
    - ensure cfg has stable model identity
    - resolve run context (dataset_id, model_tag, run_id, seed)
    - create run/log directories (dataset-aware and run-aware)
    - write config snapshot
    - load data
    - build model
    - train and write checkpoints/tensorboard under run_dir
    """
    _enable_gpu_mem_growth()

    # -----------------------------
    # Ensure stable model identity in config
    # -----------------------------
    cfg.setdefault("model", {})
    if isinstance(cfg["model"], dict):
        cfg["model"].setdefault("tag", MODEL_TAG)
        cfg["model"].setdefault("family", FAMILY)
        cfg["model"].setdefault("variant", VARIANT)

    # If dataset.id is missing, orchestration will handle it (e.g., "unknown_dataset")
    cfg.setdefault("dataset", {})
    if isinstance(cfg["dataset"], dict):
        cfg["dataset"].setdefault("id", _cfg_get(cfg, "dataset.id", None))

    # -----------------------------
    # Resolve canonical RunContext + paths (Rule A)
    # -----------------------------
    resolved = resolve_run_context(cfg, create_dirs=True)
    ctx = resolved.ctx
    cfg = resolved.cfg  # cfg now has run_meta injected

    if ctx.run_dir is None or ctx.logs_dir is None:
        raise ValueError("resolve_run_context did not produce run_dir/logs_dir. Check orchestration config.")

    run_dir = Path(ctx.run_dir)
    log_dir = Path(ctx.logs_dir)

    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    tb_root = ensure_dir(run_dir / "tensorboard")
    # Note: samples directory exists for future previews during training; sampling is separate.
    _ = ensure_dir(run_dir / "samples")

    # Run-scoped logger (writes to artifacts/logs/<dataset_id>/<model_tag>/<run_id>/run.log)
    logger = get_run_logger(name=f"{MODEL_TAG}:{ctx.run_id}", log_dir=log_dir)

    logger.info("=== c-DDPM TRAIN START ===")
    logger.info(f"dataset_id={ctx.dataset_id}")
    logger.info(f"model_tag={ctx.model_tag}")
    logger.info(f"run_id={ctx.run_id}")
    logger.info(f"seed={ctx.seed}")
    logger.info(f"run_dir={run_dir}")
    logger.info(f"logs_dir={log_dir}")

    # Snapshot effective config (audit trail)
    _snapshot_run_meta(cfg, run_dir)

    # -----------------------------
    # Resolve training hyperparameters (prefer nested, keep safe fallbacks)
    # -----------------------------
    img_shape = tuple(_cfg_get(cfg, "IMG_SHAPE", _cfg_get(cfg, "img.shape", (40, 40, 1))))
    H, W, C = int(img_shape[0]), int(img_shape[1]), int(img_shape[2])
    K = int(_cfg_get(cfg, "NUM_CLASSES", _cfg_get(cfg, "num_classes", 9)))

    epochs = int(_cfg_get(cfg, "EPOCHS", _cfg_get(cfg, "train.epochs", 200)))
    batch_size = int(_cfg_get(cfg, "BATCH_SIZE", _cfg_get(cfg, "train.batch_size", 256)))
    patience = int(_cfg_get(cfg, "PATIENCE", _cfg_get(cfg, "train.patience", 10)))

    lr = float(_cfg_get(cfg, "LR", _cfg_get(cfg, "diffusion.lr", 2e-4)))
    beta_1 = float(_cfg_get(cfg, "BETA_1", _cfg_get(cfg, "diffusion.beta_1", 0.9)))

    T = int(_cfg_get(cfg, "TIMESTEPS", _cfg_get(cfg, "diffusion.timesteps", 1000)))
    schedule = str(_cfg_get(cfg, "SCHEDULE", _cfg_get(cfg, "diffusion.schedule", "linear")))

    beta_start = float(_cfg_get(cfg, "BETA_START", _cfg_get(cfg, "diffusion.beta_start", 1e-4)))
    beta_end = float(_cfg_get(cfg, "BETA_END", _cfg_get(cfg, "diffusion.beta_end", 2e-2)))

    val_fraction = float(_cfg_get(cfg, "VAL_FRACTION", _cfg_get(cfg, "data.val_fraction", 0.5)))

    # Seed everything (use ctx.seed as source of truth)
    _set_seeds(int(ctx.seed))

    logger.info(
        f"Config: IMG_SHAPE={(H, W, C)}, K={K}, epochs={epochs}, bs={batch_size}, "
        f"lr={lr}, beta1={beta_1}, T={T}, schedule={schedule}, val_frac={val_fraction}"
    )

    # -----------------------------
    # Resolve dataset root (shared convention)
    # -----------------------------
    # Preferred: cfg.data.root or cfg.paths.data_root
    data_root = _cfg_get(cfg, "data.root", None)
    if data_root is None and isinstance(cfg.get("paths"), dict):
        data_root = cfg["paths"].get("data_root", None)

    # Legacy compatibility
    data_root = data_root or cfg.get("DATA_DIR") or cfg.get("DATA_PATH") or "data"
    data_dir = Path(str(data_root))

    logger.info(f"DATA_DIR={data_dir.resolve()}")

    # -----------------------------
    # TensorBoard writer (run-scoped)
    # -----------------------------
    tb_run_dir = tb_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_writer = tf.summary.create_file_writer(str(tb_run_dir))
    logger.info(f"TensorBoard → {tb_run_dir}")

    def _log_cb(epoch: int, train_loss: float, val_loss: Optional[float]) -> None:
        """
        Logging callback used by train_diffusion.
        Writes both to logger and to TensorBoard under the run directory.
        """
        if val_loss is None:
            logger.info(f"epoch={epoch:04d} | train_loss={train_loss:.6f}")
        else:
            logger.info(f"epoch={epoch:04d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        with tb_writer.as_default():
            tf.summary.scalar("loss/train", train_loss, step=epoch)
            if val_loss is not None:
                tf.summary.scalar("loss/val", val_loss, step=epoch)
        tb_writer.flush()

    # -----------------------------
    # Data loading (shared loader returns images in [0,1] + one-hot labels)
    # -----------------------------
    x_tr, y_tr_oh, x_va, y_va_oh, _x_te, _y_te = load_dataset_npy(
        data_dir, (H, W, C), K, val_fraction=val_fraction
    )

    # -----------------------------
    # Build model (local import keeps module import-safe for tooling)
    # -----------------------------
    from diffusion.models import build_diffusion_model  # type: ignore

    model = build_diffusion_model(
        img_shape=(H, W, C),
        num_classes=K,
        base_filters=int(_cfg_get(cfg, "BASE_FILTERS", _cfg_get(cfg, "diffusion.base_filters", 64))),
        depth=int(_cfg_get(cfg, "DEPTH", _cfg_get(cfg, "diffusion.depth", 2))),
        time_emb_dim=int(_cfg_get(cfg, "TIME_EMB_DIM", _cfg_get(cfg, "diffusion.time_emb_dim", 128))),
        learning_rate=lr,
        beta_1=beta_1,
    )

    # -----------------------------
    # Noise schedule
    # -----------------------------
    alpha_hat = build_alpha_hat(
        T=T,
        beta_start=beta_start,
        beta_end=beta_end,
        schedule=schedule,
    )

    # -----------------------------
    # Train (writes checkpoints under run_dir/checkpoints)
    # -----------------------------
    metrics = train_diffusion(
        model=model,
        x_train=x_tr,
        y_train=y_tr_oh,
        x_val=x_va,
        y_val=y_va_oh,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        beta_1=beta_1,
        alpha_hat=alpha_hat,
        T=T,
        patience=patience,
        ckpt_dir=ckpt_dir,
        log_cb=_log_cb,
    )

    # Persist a tiny training summary for quick grepping
    summary_path = run_dir / "train_summary.json"
    write_json(
        summary_path,
        {
            "timestamp": now_iso(),
            "dataset_id": str(ctx.dataset_id),
            "model_tag": str(ctx.model_tag),
            "run_id": str(ctx.run_id),
            "seed": int(ctx.seed),
            "metrics": metrics,
            "checkpoints_dir": str(ckpt_dir),
            "tensorboard_dir": str(tb_run_dir),
        },
        indent=2,
        sort_keys=True,
        atomic=True,
    )
    logger.info(f"Wrote training summary: {summary_path}")

    logger.info("=== c-DDPM TRAIN END ===")
    return metrics


# =============================================================================
# Public adapter expected by orchestration: train(cfg_or_argv) -> metrics
# =============================================================================
def train(cfg_or_argv) -> Dict[str, float]:
    """
    Orchestrator entrypoint.

    Accepts:
      - argv-like list/tuple  -> will parse YAML from --config
      - dict config           -> used directly

    Returns training metrics dict.
    """
    cfg = _coerce_cfg(cfg_or_argv)
    return _train_from_cfg(cfg)


# =============================================================================
# CLI (optional)
# =============================================================================
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=f"Train conditional DDPM ({MODEL_TAG})")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"), help="Path to YAML config")
    args = parser.parse_args(argv)

    # Run through the same adapter pathway for identical behavior
    _ = train(["--config", str(args.config)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_alpha_hat",
    "train_step",
    "eval_step",
    "train_diffusion",
    "train",
]