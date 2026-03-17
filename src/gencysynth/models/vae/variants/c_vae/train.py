# src/gencysynth/models/vae/variants/c_vae/train.py
"""
Rule A — Conditional VAE (cVAE) trainer.

This module is a *variant_local* trainer that plugs into the GenCyberSynth
unified orchestration (app/main.py).

What this trainer does
----------------------
1) Normalizes config using Rule A conventions (dataset_scoped run bundle).
2) Loads dataset via the shared loader (common.data.load_dataset_npy).
3) Converts inputs for training (decoder uses tanh => expects [-1, 1]).
4) Trains via the variant pipeline (VAEPipeline).
5) Writes:
   - checkpoints under:  {paths.artifacts}/runs/<dataset>/<vae/c_vae>/<run_id>/checkpoints
   - summaries   under:  {paths.artifacts}/runs/<dataset>/<vae/c_vae>/<run_id>/summaries
   - tensorboard under:  {paths.artifacts}/runs/<dataset>/<vae/c_vae>/<run_id>/summaries/tb

Notes
-----
- We keep this file *thin*: artifact resolution and defaults are handled by
  the variant config normalizer (config.py) so the code scales across datasets.
- This module accepts both argv_style inputs and dict configs (Rule A pattern).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import tensorflow as tf
import yaml

from common.data import load_dataset_npy, to_minus1_1

# IMPORTANT: In the new structure, this variant lives under:
#   src/gencysynth/models/vae/variants/c_vae/
# so imports should be package_relative where possible.
#
# We keep a small sys.path guard so running the file directly still works in
# some environments, but the preferred entrypoint is via app/main.py.
THIS_DIR = Path(__file__).resolve().parent
REPO_SRC = THIS_DIR.parents[4]  # .../src
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

# Variant_local pipeline + sampler
from .pipeline import VAEPipeline
from .sample import save_grid_from_decoder as save_grid

# Rule A config normalizer for this variant (expected to exist in this folder)
from .config import normalize as normalize_cfg


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def _set_seed(seed: int) -> None:
    """Best_effort reproducibility across NumPy + TF."""
    np.random.seed(int(seed))
    tf.random.set_seed(int(seed))


def _read_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf_8") as f:
        return yaml.safe_load(f) or {}


def _ensure_dir(p: Union[str, Path]) -> Path:
    """Create directory (parents included) if needed; return Path."""
    pp = p if isinstance(p, Path) else Path(str(p))
    pp.mkdir(parents=True, exist_ok=True)
    return pp


def _cfg_get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


# -----------------------------------------------------------------------------
# TensorBoard_friendly logging callback
# -----------------------------------------------------------------------------
def _make_log_cb(tb_dir: Optional[Path]):
    """
    Build a small callback the pipeline can call once per epoch.

    Contract (expected by VAEPipeline):
      cb(epoch, train_loss, recon_loss, kl_loss, val_loss)
    """
    writer = None
    if tb_dir is not None:
        tb_dir = _ensure_dir(tb_dir)
        writer = tf.summary.create_file_writer(str(tb_dir))

    def cb(epoch: int, train_loss: float, recon_loss: float, kl_loss: float, val_loss: float):
        print(
            f"[epoch {epoch:05d}] "
            f"train={train_loss:.4f} | recon={recon_loss:.4f} | KL={kl_loss:.4f} | val={val_loss:.4f}"
        )
        if writer is not None:
            with writer.as_default():
                tf.summary.scalar("loss/train_total", train_loss, step=epoch)
                tf.summary.scalar("loss/train_recon", recon_loss, step=epoch)
                tf.summary.scalar("loss/train_kl",    kl_loss,    step=epoch)
                tf.summary.scalar("loss/val_total",   val_loss,   step=epoch)
                writer.flush()

    return cb


# -----------------------------------------------------------------------------
# Core training runner (Rule A)
# -----------------------------------------------------------------------------
def _run_train(cfg_in: Dict[str, Any]) -> Dict[str, str]:
    """
    Train cVAE and write run_scoped artifacts.

    Returns a small summary dict with key file locations (strings).
    """
    # Be nice to GPUs (harmless on CPU nodes)
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    # ---- Rule A: normalize config (defaults + overrides + artifact paths) ----
    cfg = normalize_cfg(cfg_in)

    # ---- Seed ----
    seed = int(_cfg_get(cfg, "run.seed", _cfg_get(cfg, "SEED", 42)))
    _set_seed(seed)

    # ---- Resolve key training knobs ----
    img_shape = tuple(_cfg_get(cfg, "model.img_shape", _cfg_get(cfg, "IMG_SHAPE", (40, 40, 1))))
    num_classes = int(_cfg_get(cfg, "model.num_classes", _cfg_get(cfg, "NUM_CLASSES", _cfg_get(cfg, "num_classes", 9))))
    val_frac = float(_cfg_get(cfg, "train.val_fraction", _cfg_get(cfg, "VAL_FRACTION", 0.5)))
    latent_dim = int(_cfg_get(cfg, "model.latent_dim", _cfg_get(cfg, "LATENT_DIM", 100)))

    # Dataset root (prefer data.root; allow legacy DATA_DIR)
    data_root = Path(_cfg_get(cfg, "data.root", _cfg_get(cfg, "DATA_DIR", "data")))

    # ---- Rule A: resolved artifact directories ----
    ckpt_dir = _ensure_dir(_cfg_get(cfg, "artifacts.checkpoints", _cfg_get(cfg, "ARTIFACTS.checkpoints")))
    sums_dir = _ensure_dir(_cfg_get(cfg, "artifacts.summaries", _cfg_get(cfg, "ARTIFACTS.summaries")))

    # Place TB logs under summaries/tb (variant_scoped, run_scoped)
    tb_dir = _ensure_dir(sums_dir / "tb")

    # ---- Human_friendly run banner ----
    dataset_id = str(_cfg_get(cfg, "data.dataset_id", "dataset"))
    run_id = str(_cfg_get(cfg, "run.run_id", "run"))
    print(f"[c_vae] dataset_id={dataset_id} run_id={run_id} seed={seed}")
    print(f"[c_vae] data_root={data_root} img_shape={img_shape} num_classes={num_classes} latent_dim={latent_dim}")
    print(f"[c_vae] artifacts: ckpt={ckpt_dir} summaries={sums_dir} tb={tb_dir}")

    # ---- Load dataset in [0,1]; loader will handle split test->(val,test) ----
    x_train01, y_train, x_val01, y_val, _x_test01, _y_test = load_dataset_npy(
        data_root, img_shape, num_classes, val_fraction=val_frac
    )

    # ---- Map x to [-1,1] for tanh decoder ----
    x_train = to_minus1_1(x_train01)
    x_val = to_minus1_1(x_val01)

    # ---- Provide logger callback to pipeline ----
    cfg["LOG_CB"] = _make_log_cb(tb_dir)

    # ---- Train via pipeline (pipeline should save best/last as designed) ----
    pipe = VAEPipeline(cfg)
    enc, dec = pipe.train(
        x_train=x_train, y_train=y_train,
        x_val=x_val,     y_val=y_val,
    )

    # ---- Ensure we always have a "last" snapshot (Rule A robustness) ----
    # Even if the pipeline already writes these, duplicating "last" is harmless.
    try:
        enc.save_weights(str(ckpt_dir / "VAE_encoder_last.weights.h5"), overwrite=True)
        dec.save_weights(str(ckpt_dir / "VAE_decoder_last.weights.h5"), overwrite=True)
    except Exception as e:
        print(f"[c_vae][warn] encoder/decoder last snapshot save failed: {e}")

    # ---- Small preview grid (1 per class, up to K) ----
    preview_path = sums_dir / "train_preview.png"
    try:
        save_grid(
            dec,
            num_classes=num_classes,
            latent_dim=latent_dim,
            n=min(num_classes, 9),
            path=preview_path,
        )
        print(f"[c_vae] preview grid → {preview_path}")
    except Exception as e:
        print(f"[c_vae][warn] preview grid failed: {e}")

    # "OK" marker for orchestration sanity checks
    (ckpt_dir / "VAE_LAST_OK").write_text("ok", encoding="utf_8")

    return {"checkpoints": str(ckpt_dir), "summaries": str(sums_dir), "preview": str(preview_path)}


# -----------------------------------------------------------------------------
# Unified entrypoints (Rule A)
# -----------------------------------------------------------------------------
def _coerce_cfg(cfg_or_argv: Union[Dict[str, Any], list, tuple, None]) -> Dict[str, Any]:
    """
    Accept either:
      - a dict config (already parsed/merged by the caller), or
      - argv list/tuple containing '--config <path>'.
    """
    if isinstance(cfg_or_argv, dict):
        return dict(cfg_or_argv)

    # argv parsing
    p = argparse.ArgumentParser(description="Train Conditional VAE (c_vae)")
    p.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    args = p.parse_args([] if cfg_or_argv is None else list(cfg_or_argv))

    cfg_path = Path(args.config)
    print(f"[config] Using {cfg_path.resolve()}")
    return _read_yaml(cfg_path)


def main(argv=None) -> int:
    """
    argv_style entrypoint so app/main.py can call:
      main(['--config','...'])
    Also accepts a dict config for flexibility.
    """
    cfg = _coerce_cfg(argv)
    _run_train(cfg)
    return 0


def train(cfg_or_argv) -> int:
    """
    Dict/argv adapter so the unified CLI can call train(config_dict) *or*
    train(['--config','...']). Returns 0 on success.
    """
    cfg = _coerce_cfg(cfg_or_argv)
    _run_train(cfg)
    return 0


__all__ = ["main", "train"]

if __name__ == "__main__":
    raise SystemExit(main())
