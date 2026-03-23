# src/gencysynth/models/vae/variants/c_vae/sample.py
"""
Rule A — Sampling utilities + unified_CLI synth() entrypoint for Conditional VAE (cVAE).

This module supports TWO usage modes:

1) Unified Orchestrator (preferred)
   --------------------------------
   app/main.py calls:
       synth(cfg, output_root, seed) -> manifest
   where `output_root` is already a run_scoped directory like:
       {paths.artifacts}/runs/<dataset>/<vae/c_vae>/<run_id>/png/

   In this mode:
   - We DO NOT invent artifact folders.
   - We ONLY write PNGs under output_root/{class}/{seed}/...
   - We return a JSON_serializable manifest dict.

2) Standalone CLI (optional/dev convenience)
   ----------------------------------------
   python -m gencysynth.models.vae.variants.c_vae.sample --config <yaml>
   This can write `x_synth.npy/y_synth.npy` and per_class dumps under the
   run_scoped synthetic folder from Rule A config normalization.

Key conventions
--------------
- Decoder outputs tanh in [-1, 1]; we rescale to [0, 1] for saving and PNGs.
- Artifacts are dataset_scoped and run_scoped (Rule A):
    {paths.artifacts}/runs/<dataset_id>/<model_id>/<run_id>/
      checkpoints/
      summaries/
      synthetic/   (npy dumps)
      png/         (optional; orchestrator passes output_root for PNGs)
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from PIL import Image

# Variant_local model factory
from .models import build_models

# Rule A config normalizer for this variant (expected to exist)
from .config import normalize as normalize_cfg


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    tf.random.set_seed(int(seed))
    # Keras helper sets seeds for TF ops where applicable
    try:
        tf.keras.utils.set_random_seed(int(seed))
    except Exception:
        pass


def _cfg_get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _ensure_dir(p: Union[str, Path]) -> Path:
    pp = p if isinstance(p, Path) else Path(str(p))
    pp.mkdir(parents=True, exist_ok=True)
    return pp


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


# -----------------------------------------------------------------------------
# Image I/O helpers
# -----------------------------------------------------------------------------
def _to_uint8(img01: np.ndarray) -> np.ndarray:
    return np.clip(np.round(img01 * 255.0), 0, 255).astype(np.uint8)


def _save_png(img01: np.ndarray, out_path: Path) -> None:
    """
    Save a single image in [0,1] to PNG (supports 1 or 3 channels).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(img01)

    if x.ndim == 3 and x.shape[-1] == 1:
        x2 = x[..., 0]
        mode = "L"
    elif x.ndim == 3 and x.shape[-1] == 3:
        x2 = x
        mode = "RGB"
    else:
        x2 = x.squeeze()
        mode = "L"

    Image.fromarray(_to_uint8(x2), mode=mode).save(out_path)


# -----------------------------------------------------------------------------
# Decoder build/load
# -----------------------------------------------------------------------------
def _build_decoder(
    *,
    latent_dim: int,
    num_classes: int,
    img_shape: Tuple[int, int, int],
    lr: float = 2e_4,
    beta_1: float = 0.5,
    beta_kl: float = 1.0,
) -> tf.keras.Model:
    """
    Build decoder via variant_local build_models(...).

    NOTE: We pass optimizer params for parity with training config, but the
    decoder graph itself is what matters for sampling (weights define behavior).
    """
    mdict = build_models(
        img_shape=img_shape,
        latent_dim=latent_dim,
        num_classes=num_classes,
        lr=lr,
        beta_1=beta_1,
        beta_kl=beta_kl,
    )
    return mdict["decoder"]


def _load_decoder_weights(decoder: tf.keras.Model, weights_path: Optional[Path]) -> None:
    """
    Load decoder weights if available. If missing, we proceed (untrained decoder),
    but we warn loudly because outputs will be meaningless.
    """
    if weights_path is None:
        print("[c_vae][warn] No decoder weights provided; using untrained decoder.")
        return
    if not weights_path.exists():
        print(f"[c_vae][warn] Decoder weights not found at {weights_path}; using untrained decoder.")
        return
    decoder.load_weights(str(weights_path))
    print(f"[c_vae] Loaded decoder weights: {weights_path}")


# -----------------------------------------------------------------------------
# Latent sampling + decode
# -----------------------------------------------------------------------------
def sample_latents(n: int, dim: int, truncation: Optional[float] = None) -> np.ndarray:
    """
    Sample z ~ N(0, I). If truncation is provided, clip to [-t, t].
    """
    z = np.random.normal(0.0, 1.0, size=(int(n), int(dim))).astype(np.float32)
    if truncation is not None and float(truncation) > 0:
        t = float(truncation)
        z = np.clip(z, -t, t)
    return z


def _decode_to_01(decoder: tf.keras.Model, z: np.ndarray, y_onehot: np.ndarray) -> np.ndarray:
    """
    Decoder outputs tanh in [-1,1]; rescale to [0,1].
    """
    g = decoder.predict([z, y_onehot], verbose=0)  # [-1, 1]
    return np.clip((g + 1.0) / 2.0, 0.0, 1.0).astype(np.float32, copy=False)


def _one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    labels = labels.astype(np.int32).ravel()
    y = np.zeros((len(labels), int(num_classes)), dtype=np.float32)
    y[np.arange(len(labels)), labels] = 1.0
    return y


# -----------------------------------------------------------------------------
# Rule A: Unified Orchestrator entrypoint
# -----------------------------------------------------------------------------
def synth(cfg: dict, output_root: str, seed: int = 42) -> Dict[str, Any]:
    """
    Rule A unified synth entrypoint.

    Parameters
    ----------
    cfg : dict
        Full config dict (may contain legacy keys, but normalize_cfg handles Rule A).
    output_root : str
        Directory where PNGs must be written. The orchestrator supplies a run_scoped
        folder (do NOT create your own artifacts/ layout here).
    seed : int
        Sampling seed for reproducibility.

    Output layout (must match GAN/other variants)
    --------------------------------------------
    output_root/
      <class_id>/<seed>/
        vae_00000.png
        vae_00001.png
        ...

    Returns
    -------
    manifest : dict (JSON_serializable)
        {
          "dataset": "...",
          "seed": 42,
          "per_class_counts": {"0": 25, ...},
          "paths": [{"path": "...", "label": 0}, ...],
          "created_at": "ISO timestamp"
        }
    """
    # Normalize config (Rule A) so we consistently locate checkpoints across datasets/runs
    cfgN = normalize_cfg(cfg)

    # Resolve shapes/counts (support both new + legacy spellings)
    img_shape = tuple(_cfg_get(cfgN, "model.img_shape", _cfg_get(cfgN, "IMG_SHAPE", _cfg_get(cfgN, "img.shape", (40, 40, 1)))))
    K = int(_cfg_get(cfgN, "model.num_classes", _cfg_get(cfgN, "NUM_CLASSES", _cfg_get(cfgN, "num_classes", 9))))
    latent_dim = int(_cfg_get(cfgN, "model.latent_dim", _cfg_get(cfgN, "LATENT_DIM", _cfg_get(cfgN, "vae.latent_dim", 100))))
    S = int(_cfg_get(cfgN, "synth.samples_per_class", _cfg_get(cfgN, "SAMPLES_PER_CLASS", _cfg_get(cfgN, "samples_per_class", 25))))

    # Optional knobs
    trunc = _cfg_get(cfgN, "synth.truncation", None)
    lr = float(_cfg_get(cfgN, "train.lr", _cfg_get(cfgN, "LR", _cfg_get(cfgN, "vae.lr", 2e_4))))
    beta_1 = float(_cfg_get(cfgN, "train.beta_1", _cfg_get(cfgN, "BETA_1", _cfg_get(cfgN, "vae.beta_1", 0.5))))
    beta_kl = float(_cfg_get(cfgN, "train.beta_kl", _cfg_get(cfgN, "BETA_KL", _cfg_get(cfgN, "vae.beta_kl", 1.0))))

    # Decoder weights path should come from Rule A artifacts first.
    # Preferred keys:
    #   artifacts.checkpoints (dir) + decoder best file within it
    ckpt_dir = Path(_cfg_get(cfgN, "artifacts.checkpoints", _cfg_get(cfgN, "ARTIFACTS.checkpoints", "artifacts")))
    # Common naming (adjust if your pipeline uses different filename)
    default_weights = ckpt_dir / "VAE_decoder_best.weights.h5"
    fallback_weights = ckpt_dir / "VAE_decoder_last.weights.h5"
    weights_path = Path(_cfg_get(cfgN, "artifacts.decoder_weights", str(default_weights)))

    if not weights_path.exists() and default_weights.exists():
        weights_path = default_weights
    elif not weights_path.exists() and fallback_weights.exists():
        weights_path = fallback_weights
    elif not weights_path.exists():
        # Final fallback: allow old location if someone still has it in legacy artifacts/vae/checkpoints
        legacy_root = Path(_cfg_get(cfgN, "paths.artifacts", "artifacts")) / "vae" / "checkpoints"
        legacy_best = legacy_root / "D_best.weights.h5"
        legacy_last = legacy_root / "D_last.weights.h5"
        if legacy_best.exists():
            weights_path = legacy_best
        elif legacy_last.exists():
            weights_path = legacy_last
        else:
            weights_path = None  # untrained decoder

    # Seed
    _set_seed(int(seed))

    # Build + load decoder
    H, W, C = img_shape
    decoder = _build_decoder(
        latent_dim=latent_dim,
        num_classes=K,
        img_shape=(H, W, C),
        lr=lr,
        beta_1=beta_1,
        beta_kl=beta_kl,
    )
    _load_decoder_weights(decoder, weights_path if isinstance(weights_path, Path) else None)

    out_root = Path(output_root)
    _ensure_dir(out_root)

    per_class_counts: Dict[str, int] = {str(k): 0 for k in range(K)}
    paths: List[Dict[str, Any]] = []

    # Generate per class and write PNGs
    for k in range(K):
        z = sample_latents(S, latent_dim, truncation=trunc)
        y = _one_hot(np.full((S,), k, dtype=np.int32), K)
        imgs01 = _decode_to_01(decoder, z, y)  # (S,H,W,C)

        cls_dir = out_root / str(k) / str(seed)
        cls_dir.mkdir(parents=True, exist_ok=True)

        for j in range(S):
            p = cls_dir / f"vae_{j:05d}.png"
            _save_png(imgs01[j], p)
            paths.append({"path": str(p), "label": int(k)})

        per_class_counts[str(k)] = int(S)

    manifest: Dict[str, Any] = {
        "dataset": _cfg_get(cfgN, "data.root", _cfg_get(cfgN, "DATA_DIR", "data")),
        "seed": int(seed),
        "per_class_counts": per_class_counts,
        "paths": paths,
        "created_at": _now_iso(),
        "model": "vae/c_vae",
    }
    return manifest


# -----------------------------------------------------------------------------
# Standalone CLI (optional / dev convenience)
# -----------------------------------------------------------------------------
def _save_npy_bundle(
    *,
    synth_dir: Path,
    x_synth: np.ndarray,
    y_synth: np.ndarray,
    num_classes: int,
) -> None:
    """
    Save evaluator_friendly .npy artifacts in a single directory:
      gen_class_{k}.npy, labels_class_{k}.npy, x_synth.npy, y_synth.npy
    """
    synth_dir.mkdir(parents=True, exist_ok=True)

    # Per_class
    y_int = np.argmax(y_synth, axis=1).astype(np.int32)
    for k in range(int(num_classes)):
        idx = (y_int == k)
        xk = x_synth[idx]
        np.save(synth_dir / f"gen_class_{k}.npy", xk.astype(np.float32))
        np.save(synth_dir / f"labels_class_{k}.npy", np.full((len(xk),), k, dtype=np.int32))

    # Combined
    np.save(synth_dir / "x_synth.npy", x_synth.astype(np.float32))
    np.save(synth_dir / "y_synth.npy", y_synth.astype(np.float32))


def _coerce_cfg(cfg_or_argv: Union[Dict[str, Any], list, tuple, None]) -> Dict[str, Any]:
    """
    Accept either:
      - dict config, or
      - argv list/tuple containing '--config <path>'.
    """
    if isinstance(cfg_or_argv, dict):
        return dict(cfg_or_argv)

    p = argparse.ArgumentParser(description="Sample Conditional VAE (c_vae)")
    p.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    args = p.parse_args([] if cfg_or_argv is None else list(cfg_or_argv))

    cfg_path = Path(args.config)
    with open(cfg_path, "r", encoding="utf_8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def main(argv=None) -> int:
    """
    Standalone entrypoint.

    Writes .npy artifacts into the run_scoped synthetic directory resolved by
    Rule A normalize_cfg(config).
    """
    import yaml  # local import to avoid dependency for unified runner paths

    ap = argparse.ArgumentParser(description="Sample synthetic data from a trained cVAE decoder (standalone).")
    ap.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--samples_per_class", type=int, default=None)
    ap.add_argument("--num_samples", type=int, default=None)
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--classes", nargs="+", default=None, help="Subset classes, e.g. --classes 0 1 3")
    ap.add_argument("--truncation", type=float, default=None)
    ap.add_argument("--write_npy", action="store_true", help="Write evaluator .npy bundle to artifacts.synth_dir")
    ap.add_argument("--metadata", action="store_true", help="Write metadata.json into artifacts.synth_dir")
    args = ap.parse_args(argv)

    cfg_raw = {}
    with open(Path(args.config), "r", encoding="utf_8") as f:
        cfg_raw = yaml.safe_load(f) or {}

    cfg = normalize_cfg(cfg_raw)

    # Resolve key parameters
    img_shape = tuple(_cfg_get(cfg, "model.img_shape", _cfg_get(cfg, "IMG_SHAPE", (40, 40, 1))))
    K = int(_cfg_get(cfg, "model.num_classes", _cfg_get(cfg, "NUM_CLASSES", 9)))
    latent_dim = int(_cfg_get(cfg, "model.latent_dim", _cfg_get(cfg, "LATENT_DIM", 100)))

    # Where to write .npy bundles (Rule A run_scoped synthetic dir)
    synth_dir = Path(_cfg_get(cfg, "artifacts.synthetic", _cfg_get(cfg, "ARTIFACTS.synthetic", "artifacts")))

    # Determine sampling plan
    selected = [int(x) for x in args.classes] if args.classes else list(range(K))

    if args.samples_per_class is not None:
        counts = {c: int(args.samples_per_class) for c in selected}
    elif args.num_samples is not None:
        if args.balanced:
            per = int(math.floor(int(args.num_samples) / max(1, len(selected))))
            counts = {c: per for c in selected}
        elif len(selected) == 1:
            counts = {selected[0]: int(args.num_samples)}
        else:
            raise ValueError("Use --balanced with --num_samples, or provide exactly one class via --classes.")
    else:
        counts = {c: 1000 for c in selected}

    # Build decoder + load weights using the SAME logic as synth()
    _set_seed(args.seed)
    ckpt_dir = Path(_cfg_get(cfg, "artifacts.checkpoints", _cfg_get(cfg, "ARTIFACTS.checkpoints", "artifacts")))
    default_weights = ckpt_dir / "VAE_decoder_best.weights.h5"
    fallback_weights = ckpt_dir / "VAE_decoder_last.weights.h5"
    weights = default_weights if default_weights.exists() else (fallback_weights if fallback_weights.exists() else None)

    lr = float(_cfg_get(cfg, "train.lr", _cfg_get(cfg, "LR", 2e_4)))
    beta_1 = float(_cfg_get(cfg, "train.beta_1", _cfg_get(cfg, "BETA_1", 0.5)))
    beta_kl = float(_cfg_get(cfg, "train.beta_kl", _cfg_get(cfg, "BETA_KL", 1.0)))

    H, W, C = img_shape
    decoder = _build_decoder(latent_dim=latent_dim, num_classes=K, img_shape=(H, W, C), lr=lr, beta_1=beta_1, beta_kl=beta_kl)
    _load_decoder_weights(decoder, weights)

    # Generate
    xs, ys = [], []
    for cls, n in counts.items():
        z = sample_latents(n, latent_dim, truncation=(args.truncation if args.truncation is not None else _cfg_get(cfg, "synth.truncation", None)))
        y = _one_hot(np.full((n,), int(cls), dtype=np.int32), K)
        imgs01 = _decode_to_01(decoder, z, y)
        xs.append(imgs01.reshape((-1, H, W, C)))
        ys.append(y)

    x_synth = np.concatenate(xs, axis=0).astype(np.float32, copy=False) if xs else np.empty((0, H, W, C), dtype=np.float32)
    y_synth = np.concatenate(ys, axis=0).astype(np.float32, copy=False) if ys else np.empty((0, K), dtype=np.float32)

    print(f"[c_vae] generated x={x_synth.shape} y={y_synth.shape} (standalone)")

    # Optional: write evaluator_friendly npy bundle
    if args.write_npy:
        _save_npy_bundle(synth_dir=synth_dir, x_synth=x_synth, y_synth=y_synth, num_classes=K)
        print(f"[c_vae] wrote .npy bundle -> {synth_dir}")

    # Optional: metadata
    if args.metadata:
        synth_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "created_at": _now_iso(),
            "seed": int(args.seed),
            "img_shape": list(img_shape),
            "num_classes": int(K),
            "latent_dim": int(latent_dim),
            "counts": {str(k): int(v) for k, v in counts.items()},
            "weights": str(weights) if weights is not None else None,
            "synthetic_dir": str(synth_dir),
        }
        (synth_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf_8")
        print(f"[c_vae] wrote metadata.json -> {synth_dir}")

    return 0


__all__ = [
    "sample_latents",
    "synth",
    "main",
]
