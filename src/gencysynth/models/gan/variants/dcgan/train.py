# src/gencysynth/models/gan/variants/dcgan/train.py
"""
GenCyberSynth — GAN family — DCGAN variant (Conditional) — Training
==================================================================

IMPORTANT: Scalable artifact policy (Rule A)
--------------------------------------------
Everything is keyed by:
    (dataset_id, model_tag, run_id)

So this training module MUST write only under the resolved run context:

  artifacts/
    runs/<dataset_id>/<model_tag>/<run_id>/
      checkpoints/           # weights + best_fid.json
      samples/               # preview grids, debug samples
      tensorboard/           # event files
      provenance.json        # (written by orchestration if enabled)
      manifest.json          # (written by orchestration / sampling)
      run_meta_snapshot.json # config snapshot for audit/debug

and logs MUST go to:

  artifacts/logs/<dataset_id>/<model_tag>/<run_id>/
    run.log
    events.jsonl (optional)

This file should NOT invent ad-hoc directories like:
  artifacts/models/... or artifacts/synth/... or artifacts/runs/<run_id>/...

Those layouts do not scale once you add:
- multiple datasets
- multiple families and variants
- multiple seeds/runs in parallel (HPC arrays)

What this module does
---------------------
- Loads config (YAML) and resolves RunContext (dataset_id, model_tag, run_id).
- Loads dataset (currently via existing helper loader).
- Builds conditional DCGAN models (generator/discriminator/combined).
- Trains adversarially, writing:
  - checkpoints under ctx.run_dir/checkpoints/
  - preview grids under ctx.run_dir/samples/
  - tensorboard logs under ctx.run_dir/tensorboard/

Dependencies
------------
This module intentionally keeps your existing training logic and expects:
- a dataset loader (currently referenced as `common.data.load_dataset_npy`)
- a model builder (currently referenced as `gan.models.build_models`)

If you later migrate these helpers into gencysynth.data + gencysynth.models,
the ONLY things you should need to change are those imports.

CLI entry
---------
python -m gencysynth.models.gan.variants.dcgan.train --config <path/to/config.yaml>
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import tensorflow as tf

# -----------------------------------------------------------------------------
# KEEP: your existing loader/model builder (adjust later if you relocate them)
# -----------------------------------------------------------------------------
from common.data import load_dataset_npy, to_minus1_1  # type: ignore
from gan.models import build_models  # type: ignore

# Optional FID helper (expects images in [0,1])
try:
    from eval_common import fid_keras as compute_fid_01  # type: ignore
except Exception:
    compute_fid_01 = None

# -----------------------------------------------------------------------------
# GenCyberSynth shared plumbing (Rule A compatible)
# -----------------------------------------------------------------------------
from gencysynth.orchestration.context import resolve_run_context
from gencysynth.orchestration.logger import get_run_logger
from gencysynth.utils.io import write_json
from gencysynth.utils.paths import ensure_dir
from gencysynth.utils.reproducibility import now_iso


# -----------------------------------------------------------------------------
# Variant identity (must match registry tag + folder name)
# -----------------------------------------------------------------------------
FAMILY: str = "gan"
VARIANT: str = "dcgan"
MODEL_TAG: str = "gan/dcgan"


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _set_seeds(seed: int) -> None:
    """Deterministic RNG for numpy + TF (as much as TF allows)."""
    np.random.seed(int(seed))
    tf.keras.utils.set_random_seed(int(seed))


def _enable_gpu_mem_growth() -> None:
    """Avoid TF reserving all VRAM on multi-tenant GPUs."""
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass


def _to_float(x: Any) -> float:
    """Convert Keras/TF outputs into a plain float (handles list/tuple/tensor)."""
    if isinstance(x, (list, tuple)):
        x = x[0]
    try:
        return float(x)
    except Exception:
        arr = np.asarray(x)
        return float(arr.reshape(-1)[0])


def _maybe_add_noise(x: np.ndarray, std: float) -> np.ndarray:
    """Optional stabilization: inject small Gaussian noise after warmup."""
    if std <= 0:
        return x
    return x + np.random.normal(0.0, std, size=x.shape).astype(np.float32)


def _save_grid(
    images01: np.ndarray,
    img_shape: Tuple[int, int, int],
    rows: int,
    cols: int,
    out_path: Path,
) -> None:
    """
    Save a PNG preview grid from images in [0,1].

    NOTE: This is a human-inspection artifact and therefore belongs under:
        ctx.run_dir/samples/
    """
    import matplotlib.pyplot as plt

    H, W, C = img_shape
    n = min(rows * cols, images01.shape[0])

    plt.figure(figsize=(cols * 1.6, rows * 1.6))
    for i in range(n):
        ax = plt.subplot(rows, cols, i + 1)
        im = images01[i].reshape(H, W, C)
        if C == 1:
            plt.imshow(im.squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
        else:
            plt.imshow(np.clip(im, 0.0, 1.0))
        ax.axis("off")

    plt.tight_layout()
    ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=200)
    plt.close()


def _snapshot_run_meta(cfg: dict, run_dir: Path) -> Path:
    """
    Write a stable config snapshot for forensic reproducibility.

    Stored in the run directory:
      artifacts/runs/<dataset_id>/<model_tag>/<run_id>/run_meta_snapshot.json

    Why include full cfg?
    - Later, you can reconstruct exactly what produced this run
    - Useful on HPC when configs are generated/overridden dynamically
    """
    out = Path(run_dir) / "run_meta_snapshot.json"
    payload = {
        "timestamp": now_iso(),
        "run_meta": cfg.get("run_meta", {}),
        "model": cfg.get("model", {}),
        "dataset": cfg.get("dataset", {}),
        "paths": cfg.get("paths", {}),
        "cfg": cfg,  # full config snapshot (can be large; that's okay)
    }
    return write_json(out, payload, indent=2, sort_keys=True, atomic=True)


def _cfg_get(cfg: dict, dotted: str, default=None):
    """Nested dict getter: _cfg_get(cfg, 'paths.data_root', None)."""
    cur: Any = cfg
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# -----------------------------------------------------------------------------
# Core training
# -----------------------------------------------------------------------------
def run_from_file(
    cfg_path: Path,
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    eval_every: int = 25,
    save_every: int = 50,
    label_smooth: tuple[float, float] = (0.9, 1.0),
    fake_label_range: tuple[float, float] = (0.0, 0.1),
    noise_after: int = 200,
    noise_std: float = 0.01,
    grid: tuple[int, int] | None = None,
    g_weights: Path | None = None,
    d_weights: Path | None = None,
    sample_after: bool = False,
    samples_per_class: int = 0,
    seed: int = 42,
) -> int:
    """
    Train Conditional DCGAN using YAML config at cfg_path.

    Returns
    -------
    int
        0 on success (CLI-friendly).
    """
    import yaml

    # -------------------------------------------------------------
    # Reproducibility (seed + GPU behavior)
    # -------------------------------------------------------------
    _set_seeds(seed)
    _enable_gpu_mem_growth()

    # -------------------------------------------------------------
    # Load config and enforce minimum identity fields
    # -------------------------------------------------------------
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    if not isinstance(cfg, dict):
        raise TypeError("Config YAML must parse to a dict/object.")

    # Ensure model identity is stable for this variant.
    # Orchestration uses cfg["model"]["tag"] to form the (dataset_id, model_tag, run_id) key.
    cfg.setdefault("model", {})
    if isinstance(cfg["model"], dict):
        cfg["model"].setdefault("tag", MODEL_TAG)
        cfg["model"].setdefault("family", FAMILY)
        cfg["model"].setdefault("variant", VARIANT)

    # -------------------------------------------------------------
    # Resolve run context (Rule A): creates dataset/model/run scoped dirs
    # -------------------------------------------------------------
    resolved = resolve_run_context(cfg, create_dirs=True)
    ctx = resolved.ctx
    cfg = resolved.cfg  # cfg now has run_meta injected: dataset_id/model_tag/run_id/seed

    assert ctx.run_dir is not None and ctx.logs_dir is not None and ctx.eval_dir is not None
    run_dir = Path(ctx.run_dir)
    log_dir = Path(ctx.logs_dir)

    # Run-local subfolders (canonical; do NOT create any other top-level layouts)
    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    samples_dir = ensure_dir(run_dir / "samples")
    tb_root = ensure_dir(run_dir / "tensorboard")

    # -------------------------------------------------------------
    # Logger (console + artifacts/logs/.../run.log)
    # -------------------------------------------------------------
    logger = get_run_logger(name=f"{MODEL_TAG}:{ctx.run_id}", log_dir=log_dir)

    logger.info("=== DCGAN TRAIN START ===")
    logger.info(f"dataset_id={ctx.dataset_id}")
    logger.info(f"model_tag={ctx.model_tag}")
    logger.info(f"run_id={ctx.run_id}")
    logger.info(f"seed={ctx.seed}")
    logger.info(f"run_dir={run_dir}")
    logger.info(f"logs_dir={log_dir}")
    logger.info(f"eval_dir={ctx.eval_dir}")

    # Config snapshot (audit trail)
    _snapshot_run_meta(cfg, run_dir)

    # -------------------------------------------------------------
    # Hyperparameters (prefer config keys; preserve legacy fallbacks)
    # -------------------------------------------------------------
    IMG_SHAPE = tuple(cfg.get("IMG_SHAPE", (40, 40, 1)))
    NUM_CLASSES = int(cfg.get("NUM_CLASSES", 9))
    LATENT_DIM = int(cfg.get("LATENT_DIM", 100))
    EPOCHS = int(epochs if epochs is not None else cfg.get("EPOCHS", 5000))
    BATCH_SIZE = int(batch_size if batch_size is not None else cfg.get("BATCH_SIZE", 256))
    LR = float(cfg.get("LR", 2e-4))
    BETA_1 = float(cfg.get("BETA_1", 0.5))
    VAL_FRACTION = float(cfg.get("VAL_FRACTION", 0.5))

    # Resolve dataset root directory.
    # Preferred new key: cfg.paths.data_root
    # Legacy keys: DATA_DIR / DATA_PATH
    data_root = _cfg_get(cfg, "paths.data_root", None) or cfg.get("DATA_DIR") or cfg.get("DATA_PATH")

    if data_root is None:
        # Conservative fallback: relative to config file
        data_root = (cfg_path.resolve().parents[1] / "USTC-TFC2016_malware").resolve()

    DATA_DIR = Path(data_root)

    logger.info(
        f"Config: IMG_SHAPE={IMG_SHAPE}, K={NUM_CLASSES}, Z={LATENT_DIM}, "
        f"epochs={EPOCHS}, bs={BATCH_SIZE}, lr={LR}, beta1={BETA_1}, val_frac={VAL_FRACTION}"
    )
    logger.info(f"DATA_DIR={DATA_DIR}")

    # -------------------------------------------------------------
    # TensorBoard logs: run-local
    # -------------------------------------------------------------
    tb_run_dir = tb_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = tf.summary.create_file_writer(str(tb_run_dir))
    logger.info(f"TensorBoard → {tb_run_dir}")

    # -------------------------------------------------------------
    # Data
    # -------------------------------------------------------------
    x_train01, y_train_oh, x_val01, y_val_oh, x_test01, y_test_oh = load_dataset_npy(
        DATA_DIR, IMG_SHAPE, NUM_CLASSES, val_fraction=VAL_FRACTION
    )

    # DCGAN expects [-1, 1] input range
    x_train = to_minus1_1(x_train01).astype(np.float32)

    # -------------------------------------------------------------
    # Models
    # -------------------------------------------------------------
    nets = build_models(
        latent_dim=LATENT_DIM,
        num_classes=NUM_CLASSES,
        img_shape=IMG_SHAPE,
        lr=LR,
        beta_1=BETA_1,
    )
    G: tf.keras.Model = nets["generator"]
    D: tf.keras.Model = nets["discriminator"]
    COMBINED: tf.keras.Model = nets["gan"]  # D is frozen inside combined graph

    # Optional resume (explicit paths only; avoids guessing legacy folders)
    if g_weights and Path(g_weights).exists():
        G.load_weights(str(g_weights))
        logger.info(f"Loaded generator weights from {g_weights}")
    if d_weights and Path(d_weights).exists():
        D.load_weights(str(d_weights))
        logger.info(f"Loaded discriminator weights from {d_weights}")

    # -------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------
    steps_per_epoch = math.ceil(x_train.shape[0] / BATCH_SIZE)
    best_fid = float("inf")
    best_meta_path = ckpt_dir / "best_fid.json"

    logger.info(f"Start training for {EPOCHS} epochs ({steps_per_epoch} steps/epoch).")

    for epoch in range(1, EPOCHS + 1):
        idx = np.random.permutation(x_train.shape[0])
        d_losses, g_losses = [], []
        fid_val: Optional[float] = None

        for step in range(steps_per_epoch):
            # -----------------------------
            # Real batch
            # -----------------------------
            batch_idx = idx[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
            real_imgs = x_train[batch_idx]
            real_lbls = y_train_oh[batch_idx].astype(np.float32)

            # Important: last batch may be smaller than BATCH_SIZE
            n = int(real_imgs.shape[0])

            # Optional noise injection for stability (after warmup)
            if epoch > noise_after and noise_std > 0:
                real_imgs = _maybe_add_noise(real_imgs, noise_std)

            # -----------------------------
            # Fake batch (generate)
            # -----------------------------
            z = np.random.normal(0.0, 1.0, size=(n, LATENT_DIM)).astype(np.float32)
            fake_class_int = np.random.randint(0, NUM_CLASSES, size=(n,))
            fake_lbls = tf.keras.utils.to_categorical(fake_class_int, NUM_CLASSES).astype(np.float32)

            gen_imgs = G.predict([z, fake_lbls], verbose=0)

            if epoch > noise_after and noise_std > 0:
                gen_imgs = _maybe_add_noise(gen_imgs, noise_std)

            # -----------------------------
            # Label smoothing
            # -----------------------------
            real_y = np.random.uniform(label_smooth[0], label_smooth[1], size=(n, 1)).astype(np.float32)
            fake_y = np.random.uniform(fake_label_range[0], fake_label_range[1], size=(n, 1)).astype(np.float32)

            # -----------------------------
            # Train Discriminator (real + fake)
            # -----------------------------
            D.trainable = True
            d_out_real = D.train_on_batch([real_imgs, real_lbls], real_y)
            d_out_fake = D.train_on_batch([gen_imgs, fake_lbls], fake_y)
            d_loss = 0.5 * (_to_float(d_out_real) + _to_float(d_out_fake))

            # -----------------------------
            # Train Generator via Combined graph (D frozen)
            # -----------------------------
            D.trainable = False
            z2 = np.random.normal(0.0, 1.0, size=(n, LATENT_DIM)).astype(np.float32)
            g_lbls_int = np.random.randint(0, NUM_CLASSES, size=(n,))
            g_lbls = tf.keras.utils.to_categorical(g_lbls_int, NUM_CLASSES).astype(np.float32)

            g_out = COMBINED.train_on_batch([z2, g_lbls], np.ones((n, 1), dtype=np.float32))
            g_loss = _to_float(g_out)

            d_losses.append(float(d_loss))
            g_losses.append(float(g_loss))

        # -----------------------------
        # End of epoch: aggregate losses
        # -----------------------------
        d_loss_ep = float(np.mean(d_losses)) if d_losses else float("nan")
        g_loss_ep = float(np.mean(g_losses)) if g_losses else float("nan")

        # -----------------------------
        # Preview grid (optional) -> run_dir/samples/
        # -----------------------------
        if grid is not None:
            rows, cols = grid
            m = rows * cols
            z = np.random.normal(0.0, 1.0, size=(m, LATENT_DIM)).astype(np.float32)
            cyc = np.arange(m) % NUM_CLASSES
            y_cyc = tf.keras.utils.to_categorical(cyc, NUM_CLASSES).astype(np.float32)

            g = G.predict([z, y_cyc], verbose=0)
            g01 = np.clip((g + 1.0) / 2.0, 0.0, 1.0)
            _save_grid(g01, IMG_SHAPE, rows, cols, samples_dir / f"grid_epoch_{epoch:04d}.png")

        # -----------------------------
        # FID (optional, lightweight)
        # -----------------------------
        if compute_fid_01 is not None and (epoch % max(1, eval_every) == 0):
            n_fid = min(200, x_val01.shape[0])  # frequent FID should be cheap
            real01 = x_val01[:n_fid]

            z = np.random.normal(0.0, 1.0, size=(n_fid, LATENT_DIM)).astype(np.float32)
            labels_int = np.random.randint(0, NUM_CLASSES, size=(n_fid,))
            y_oh = tf.keras.utils.to_categorical(labels_int, NUM_CLASSES).astype(np.float32)

            g = G.predict([z, y_oh], verbose=0)
            fake01 = np.clip((g + 1.0) / 2.0, 0.0, 1.0)

            try:
                fid_val = float(compute_fid_01(real01, fake01))  # type: ignore[misc]
            except Exception:
                fid_val = None

            # Save best-by-FID under run_dir/checkpoints/
            if fid_val is not None and fid_val < best_fid:
                best_fid = fid_val
                G.save_weights(str(ckpt_dir / "G_best.weights.h5"))
                D.save_weights(str(ckpt_dir / "D_best.weights.h5"))
                best_meta_path.write_text(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "best_fid": best_fid,
                            "timestamp": now_iso(),
                            "dataset_id": ctx.dataset_id,
                            "model_tag": ctx.model_tag,
                            "run_id": ctx.run_id,
                        },
                        indent=2,
                    )
                )
                logger.info(f"[BEST] epoch={epoch} new best FID={best_fid:.4f} → saved *_best.weights.h5")

        # -----------------------------
        # Periodic checkpoints -> run_dir/checkpoints/
        # -----------------------------
        if (epoch % max(1, save_every) == 0) or (epoch == EPOCHS):
            G.save_weights(str(ckpt_dir / "G_last.weights.h5"))
            D.save_weights(str(ckpt_dir / "D_last.weights.h5"))
            # Optional epoch snapshot
            G.save_weights(str(ckpt_dir / f"G_epoch_{epoch:04d}.weights.h5"))
            D.save_weights(str(ckpt_dir / f"D_epoch_{epoch:04d}.weights.h5"))

        # -----------------------------
        # Logging + TensorBoard scalars
        # -----------------------------
        if fid_val is not None:
            logger.info(
                f"epoch={epoch:04d} | D_loss={d_loss_ep:.4f} | G_loss={g_loss_ep:.4f} | FID={fid_val:.4f}"
            )
        else:
            logger.info(f"epoch={epoch:04d} | D_loss={d_loss_ep:.4f} | G_loss={g_loss_ep:.4f}")

        with writer.as_default():
            tf.summary.scalar("loss/D", d_loss_ep, step=epoch)
            tf.summary.scalar("loss/G", g_loss_ep, step=epoch)
            if fid_val is not None:
                tf.summary.scalar("fid/val", fid_val, step=epoch)
        writer.flush()

    logger.info("Training complete.")

    # -------------------------------------------------------------
    # Optional post-training sampling (legacy feature)
    # -------------------------------------------------------------
    # In the new architecture, sampling should normally happen via model.sample()
    # so it can produce a manifest.json under the run directory.
    if sample_after and samples_per_class > 0:
        out_dir = ensure_dir(run_dir / "samples" / "post_train_npy")
        logger.info(f"Sampling {samples_per_class} per class → {out_dir}")

        for k in range(NUM_CLASSES):
            z = np.random.normal(0.0, 1.0, size=(samples_per_class, LATENT_DIM)).astype(np.float32)
            y = tf.keras.utils.to_categorical(
                np.full((samples_per_class,), k), NUM_CLASSES
            ).astype(np.float32)

            g = G.predict([z, y], verbose=0)          # [-1,1]
            g01 = np.clip((g + 1.0) / 2.0, 0.0, 1.0)  # [0,1]

            np.save(out_dir / f"gen_class_{k}.npy", g01)
            np.save(out_dir / f"labels_class_{k}.npy", np.full((samples_per_class,), k, dtype=np.int32))

        logger.info("Post-train sampling done.")

    logger.info("=== DCGAN TRAIN END ===")
    return 0


# -----------------------------------------------------------------------------
# Orchestrator adapter
# -----------------------------------------------------------------------------
def train(cfg_or_argv):
    """
    Orchestrator entrypoint.

    Accepts:
      - argv-like list/tuple  -> CLI args
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


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"Train Conditional DCGAN ({MODEL_TAG})")

    default_cfg = Path("configs/config.yaml")
    parser.add_argument("--config", type=Path, default=default_cfg, help="Path to YAML config")

    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--eval-every", type=int, default=25, help="Evaluate FID every N epochs")
    parser.add_argument("--save-every", type=int, default=50, help="Save checkpoints every N epochs")

    parser.add_argument(
        "--label-smooth",
        type=float,
        nargs=2,
        default=(0.9, 1.0),
        help="Real label smoothing range [low high]",
    )
    parser.add_argument(
        "--fake-label-range",
        type=float,
        nargs=2,
        default=(0.0, 0.1),
        help="Fake label range [low high]",
    )

    parser.add_argument("--noise-after", type=int, default=200, help="Start noise injection after this epoch")
    parser.add_argument("--noise-std", type=float, default=0.01, help="Gaussian noise std after warmup")

    parser.add_argument("--grid", type=int, nargs=2, default=None, help="Save preview grid: ROWS COLS")

    parser.add_argument("--g-weights", type=Path, default=None, help="Resume generator weights")
    parser.add_argument("--d-weights", type=Path, default=None, help="Resume discriminator weights")

    parser.add_argument("--sample-after", action="store_true", help="Generate per-class .npy samples after training")
    parser.add_argument("--samples-per-class", type=int, default=0, help="Samples per class if --sample-after")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args(argv)

    return run_from_file(
        cfg_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        save_every=args.save_every,
        label_smooth=tuple(args.label_smooth),
        fake_label_range=tuple(args.fake_label_range),
        noise_after=args.noise_after,
        noise_std=args.noise_std,
        grid=tuple(args.grid) if args.grid else None,
        g_weights=args.g_weights,
        d_weights=args.d_weights,
        sample_after=args.sample_after,
        samples_per_class=int(args.samples_per_class),
        seed=args.seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
