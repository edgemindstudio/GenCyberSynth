# src/gencysynth/models/diffusion/variants/c_ddpm/pipeline.py
"""
GenCyberSynth — Diffusion family — c_DDPM variant (Conditional) — Pipeline
=========================================================================
(Training + Synthesis convenience wrapper)

RULE A (Scalable artifact policy)
---------------------------------
Everything is keyed by:
    (dataset_id, model_tag, run_id)

Therefore ALL outputs MUST live under the resolved run directory:

  artifacts/
    runs/<dataset_id>/<model_tag>/<run_id>/
      checkpoints/            # weights + best tracking
      samples/                # preview grids / debug images (optional)
      tensorboard/            # event files (optional)
      manifest.json           # written by sampling (or orchestration)
      run_meta_snapshot.json  # config snapshot for audit/debug

And logs MUST go to:
  artifacts/logs/<dataset_id>/<model_tag>/<run_id>/
    run.log

Why this file exists
--------------------
This module is a small "pipeline" wrapper that can be useful for experiments or
legacy integrations. HOWEVER, in the GenCyberSynth architecture you already
have:
  - variants/c_ddpm/train.py   : training loop + checkpoints/logging
  - variants/c_ddpm/samply.py  : sampling + manifest writing

So this pipeline is intentionally a *thin adapter* that:
  1) resolves RunContext (Rule A)
  2) trains via the shared training function (train_diffusion)
  3) samples via the shared sampler (sample_batch, PNG writing, manifest)

What this module MUST NOT do
----------------------------
- invent non_scalable paths like artifacts/diffusion/... (dataset_less)
- write to global synthetic folders
- bypass RunContext (dataset_id/model_tag/run_id)

Compatibility notes
-------------------
- Images are expected in [0,1] for training and sampling.
- Diffusion model signature:
      model([x_t, y_onehot, t_vec]) -> eps_hat
- Checkpoints use Keras 3_friendly "*.weights.h5".

This pipeline keeps the old "class" interface but now it is Rule_A compliant.
"""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import tensorflow as tf

# -----------------------------------------------------------------------------
# Local model builder (kept import_safe)
# -----------------------------------------------------------------------------
from diffusion.models import build_diffusion_model

# -----------------------------------------------------------------------------
# Training + sampling primitives (reuse variant modules; single source of truth)
# -----------------------------------------------------------------------------
from .train import build_alpha_hat, train_diffusion  # Rule_A updates live in train.py
from .samply import sample_batch  # Rule_A updates live in samply.py

# -----------------------------------------------------------------------------
# GenCyberSynth shared plumbing (Rule A)
# -----------------------------------------------------------------------------
from gencysynth.models.base_types import RunContext, SampleResult
from gencysynth.orchestration.context import resolve_run_context
from gencysynth.orchestration.logger import get_run_logger
from gencysynth.utils.io import write_json
from gencysynth.utils.paths import ensure_dir
from gencysynth.utils.reproducibility import now_iso

# -----------------------------------------------------------------------------
# Legacy dataset loader (kept for compatibility; swap later to gencysynth.data)
# -----------------------------------------------------------------------------
from common.data import load_dataset_npy  # type: ignore


# -----------------------------------------------------------------------------
# Variant identity (must match folder structure and registry tag)
# -----------------------------------------------------------------------------
FAMILY: str = "diffusion"
VARIANT: str = "c_ddpm"
MODEL_TAG: str = "diffusion/c_ddpm"


# =============================================================================
# Small config helpers
# =============================================================================
def _cfg_get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    """Read nested config values using a dotted key, e.g. 'paths.artifacts'."""
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _enable_gpu_mem_growth() -> None:
    """Avoid TF reserving all VRAM on shared GPUs (HPC_friendly)."""
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass


def _snapshot_run_meta(cfg: Dict[str, Any], run_dir: Path) -> Path:
    """
    Write a stable config snapshot for forensic reproducibility.

    Stored in:
      artifacts/runs/<dataset_id>/<model_tag>/<run_id>/run_meta_snapshot.json
    """
    out = run_dir / "run_meta_snapshot.json"
    payload = {
        "timestamp": now_iso(),
        "model_tag": _cfg_get(cfg, "model.tag", MODEL_TAG),
        "dataset_id": _cfg_get(cfg, "dataset.id", _cfg_get(cfg, "dataset_id", None)),
        "run_meta": cfg.get("run_meta", {}),
        "paths": cfg.get("paths", {}),
        "cfg": cfg,
    }
    return write_json(out, payload, indent=2, sort_keys=True, atomic=True)


def _to_uint8(img01: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(img01 * 255.0), 0, 255).astype(np.uint8)


def _save_png(img01: np.ndarray, out_path: Path) -> None:
    """Save float image in [0,1] as PNG (grayscale or RGB)."""
    from PIL import Image

    ensure_dir(out_path.parent)
    x = img01
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
        mode = "L"
    elif x.ndim == 3 and x.shape[-1] == 3:
        mode = "RGB"
    else:
        x = x.squeeze()
        mode = "L"
    Image.fromarray(_to_uint8(x), mode=mode).save(out_path)


# =============================================================================
# DiffusionPipeline (Rule_A compliant)
# =============================================================================
class DiffusionPipeline:
    """
    Convenience wrapper around:
      - build_diffusion_model(...)
      - train_diffusion(...)
      - sample_batch(...)

    IMPORTANT:
    This class is now Rule_A compliant: it always resolves a RunContext and
    reads/writes ONLY under the run directory for that context.
    """

    DEFAULTS: Dict[str, Any] = {
        "IMG_SHAPE": (40, 40, 1),
        "NUM_CLASSES": 9,
        "EPOCHS": 200,
        "BATCH_SIZE": 128,
        "LR": 2e_4,
        "BETA_1": 0.9,
        "BASE_FILTERS": 64,
        "DEPTH": 2,
        "TIME_EMB_DIM": 128,
        "TIMESTEPS": 1000,          # training diffusion steps
        "SCHEDULE": "linear",       # "linear" or "cosine" (if supported by build_alpha_hat)
        "BETA_START": 1e_4,
        "BETA_END": 2e_2,
        "LOG_EVERY": 25,            # periodic epoch checkpoints
        "PATIENCE": 10,             # early stopping patience
        "SAMPLES_PER_CLASS": 1000,  # default synthesis budget (legacy)
        "DIFFUSION_STEPS": 200,     # reverse sampling steps (preview_friendly)
    }

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg: Dict[str, Any] = cfg or {}

        # Ensure model identity exists (helps registry + collisions)
        self.cfg.setdefault("model", {})
        if isinstance(self.cfg["model"], dict):
            self.cfg["model"].setdefault("tag", MODEL_TAG)
            self.cfg["model"].setdefault("family", FAMILY)
            self.cfg["model"].setdefault("variant", VARIANT)

        # Resolve RunContext (Rule A) and create directories
        resolved = resolve_run_context(self.cfg, create_dirs=True)
        self.cfg = resolved.cfg
        self.ctx: RunContext = resolved.ctx

        if self.ctx.run_dir is None or self.ctx.logs_dir is None:
            raise ValueError("resolve_run_context must provide run_dir and logs_dir.")

        # Canonical run_scoped directories
        self.run_dir = Path(self.ctx.run_dir)
        self.log_dir = Path(self.ctx.logs_dir)

        self.ckpt_dir = ensure_dir(self.run_dir / "checkpoints")
        self.samples_dir = ensure_dir(self.run_dir / "samples")
        self.tb_root = ensure_dir(self.run_dir / "tensorboard")

        # Logger (writes to artifacts/logs/.../run.log)
        self.logger = get_run_logger(name=f"{MODEL_TAG}:{self.ctx.run_id}:pipeline", log_dir=self.log_dir)

        # GPU behavior
        _enable_gpu_mem_growth()

        # Snapshot config for audit/debug
        _snapshot_run_meta(self.cfg, self.run_dir)

        # -----------------------
        # Hyperparameters / shapes
        # -----------------------
        d = self.DEFAULTS
        self.img_shape: Tuple[int, int, int] = tuple(self.cfg.get("IMG_SHAPE", d["IMG_SHAPE"]))
        self.num_classes: int = int(self.cfg.get("NUM_CLASSES", d["NUM_CLASSES"]))

        self.epochs: int = int(self.cfg.get("EPOCHS", d["EPOCHS"]))
        self.batch_size: int = int(self.cfg.get("BATCH_SIZE", d["BATCH_SIZE"]))
        self.lr: float = float(self.cfg.get("LR", d["LR"]))
        self.beta_1: float = float(self.cfg.get("BETA_1", d["BETA_1"]))

        self.base_filters: int = int(self.cfg.get("BASE_FILTERS", d["BASE_FILTERS"]))
        self.depth: int = int(self.cfg.get("DEPTH", d["DEPTH"]))
        self.time_emb_dim: int = int(self.cfg.get("TIME_EMB_DIM", d["TIME_EMB_DIM"]))

        self.T_train: int = int(self.cfg.get("TIMESTEPS", d["TIMESTEPS"]))
        self.schedule: str = str(self.cfg.get("SCHEDULE", d["SCHEDULE"]))
        self.beta_start: float = float(self.cfg.get("BETA_START", d["BETA_START"]))
        self.beta_end: float = float(self.cfg.get("BETA_END", d["BETA_END"]))

        self.log_every: int = int(self.cfg.get("LOG_EVERY", d["LOG_EVERY"]))
        self.patience: int = int(self.cfg.get("PATIENCE", d["PATIENCE"]))

        # Synthesis knobs (kept for legacy convenience)
        self.samples_per_class: int = int(self.cfg.get("SAMPLES_PER_CLASS", d["SAMPLES_PER_CLASS"]))
        self.T_sample: int = int(self.cfg.get("DIFFUSION_STEPS", d["DIFFUSION_STEPS"]))

        # Optional external logging callback (epoch, train_loss, val_loss)
        self.log_cb = self.cfg.get("LOG_CB", None)

        # Seed
        seed = int(getattr(self.ctx, "seed", _cfg_get(self.cfg, "run_meta.seed", 42)))
        tf.keras.utils.set_random_seed(seed)
        np.random.seed(seed)

        # TensorBoard writer under run directory
        tb_run_dir = self.tb_root / datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tb_writer = tf.summary.create_file_writer(str(tb_run_dir))

        # Build model (fresh init; weights will be saved/loaded from run_scoped ckpts)
        self.model: tf.keras.Model = build_diffusion_model(
            img_shape=self.img_shape,
            num_classes=self.num_classes,
            base_filters=self.base_filters,
            depth=self.depth,
            time_emb_dim=self.time_emb_dim,
            learning_rate=self.lr,
            beta_1=self.beta_1,
        )

        self.logger.info("=== c_DDPM PIPELINE INIT ===")
        self.logger.info(f"dataset_id={self.ctx.dataset_id} model_tag={self.ctx.model_tag} run_id={self.ctx.run_id} seed={seed}")
        self.logger.info(f"run_dir={self.run_dir}")
        self.logger.info(f"ckpt_dir={self.ckpt_dir}")
        self.logger.info(f"samples_dir={self.samples_dir}")
        self.logger.info(f"tensorboard={tb_run_dir}")
        self.logger.info(
            f"cfg: img_shape={self.img_shape} K={self.num_classes} "
            f"epochs={self.epochs} bs={self.batch_size} lr={self.lr} beta1={self.beta_1} "
            f"T_train={self.T_train} schedule={self.schedule}"
        )

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    def train(self) -> tf.keras.Model:
        """
        Train using the shared train_diffusion(...) utility, writing checkpoints
        ONLY under this run's checkpoints directory (Rule A).

        Data loading:
        ------------
        This wrapper uses the legacy loader `common.data.load_dataset_npy`.
        It expects the loader returns:
          x_train01, y_train_oh, x_val01, y_val_oh, x_test01, y_test_oh

        You can later swap the loader import without changing artifact logic.
        """
        # -----------------------------
        # Resolve dataset path
        # -----------------------------
        data_root = None
        if isinstance(self.cfg.get("paths"), dict):
            data_root = self.cfg["paths"].get("data_root", None)
        data_root = data_root or self.cfg.get("DATA_DIR") or _cfg_get(self.cfg, "data.root", None)

        if data_root is None:
            # conservative fallback: relative to config location handled upstream
            data_root = "data"

        data_dir = Path(str(data_root))

        # Validation fraction (legacy)
        val_fraction = float(self.cfg.get("VAL_FRACTION", 0.5))

        self.logger.info(f"[data] DATA_DIR={data_dir} val_fraction={val_fraction}")

        # Load data in [0,1]
        x_tr, y_tr_oh, x_va, y_va_oh, _x_te, _y_te = load_dataset_npy(
            data_dir, self.img_shape, self.num_classes, val_fraction=val_fraction
        )

        # -----------------------------
        # Noise schedule (alpha_hat) used by training objective
        # -----------------------------
        alpha_hat = build_alpha_hat(
            T=self.T_train,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            schedule=self.schedule,
        )

        # -----------------------------
        # TensorBoard hook
        # -----------------------------
        def _log_cb(epoch: int, train_loss: float, val_loss: Optional[float]) -> None:
            # External callback (optional)
            if self.log_cb is not None:
                self.log_cb(epoch, train_loss, val_loss)

            # Canonical TB scalars under run_dir/tensorboard/
            with self.tb_writer.as_default():
                tf.summary.scalar("loss/train", float(train_loss), step=epoch)
                if val_loss is not None:
                    tf.summary.scalar("loss/val", float(val_loss), step=epoch)
            self.tb_writer.flush()

            # Also write to run.log
            if val_loss is not None:
                self.logger.info(f"epoch={epoch:04d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
            else:
                self.logger.info(f"epoch={epoch:04d} train_loss={train_loss:.6f}")

        # -----------------------------
        # Train (writes checkpoints under ckpt_dir)
        # -----------------------------
        metrics = train_diffusion(
            model=self.model,
            x_train=x_tr, y_train=y_tr_oh,
            x_val=x_va,   y_val=y_va_oh,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.lr,
            beta_1=self.beta_1,
            alpha_hat=alpha_hat,
            T=self.T_train,
            patience=self.patience,
            ckpt_dir=self.ckpt_dir,
            log_cb=_log_cb,
        )

        self.logger.info(f"[train] done metrics={metrics}")
        return self.model

    # -------------------------------------------------------------------------
    # Synthesis (Rule A compliant)
    # -------------------------------------------------------------------------
    def synthesize(self, model: Optional[tf.keras.Model] = None) -> SampleResult:
        """
        Generate class_balanced PNGs under:
          run_dir/samples/generated/class_<k>/diff_00000.png

        And write:
          run_dir/manifest.json

        Notes
        -----
        - This uses the lightweight sampler from samply.py (reverse diffusion).
        - For very large synthesis (millions), prefer a dedicated sampling script
          + sharding; this function is meant for "run_sized" generation.
        """
        if model is None:
            model = self._build_and_load_for_sampling()

        out_root = ensure_dir(self.run_dir / "samples" / "generated")
        H, W, C = self.img_shape
        K = int(self.num_classes)
        S = int(self.samples_per_class)
        T = int(self.T_sample)

        self.logger.info(f"[sample] start K={K} samples_per_class={S} T_sample={T}")

        # Precompute schedule once for speed
        alpha_hat = build_alpha_hat(T=T, beta_start=self.beta_start, beta_end=self.beta_end, schedule="linear")

        paths: List[Dict[str, Any]] = []
        per_class_counts: Dict[str, int] = {str(k): 0 for k in range(K)}

        # Deterministic sampling per class (seed + 1000*k)
        base_seed = int(getattr(self.ctx, "seed", _cfg_get(self.cfg, "run_meta.seed", 42)))

        for k in range(K):
            class_seed = base_seed + 1000 * k
            class_ids = np.full((S,), k, dtype=np.int32)

            x01, _ = sample_batch(
                model=model,
                num_samples=S,
                num_classes=K,
                img_shape=(H, W, C),
                T=T,
                alpha_hat=alpha_hat,
                class_ids=class_ids,
                seed=class_seed,
            )

            class_dir = ensure_dir(out_root / f"class_{k}")
            for j in range(S):
                p = class_dir / f"diff_{j:05d}.png"
                _save_png(x01[j], p)
                paths.append({"path": str(p.relative_to(self.run_dir)), "label": int(k)})

            per_class_counts[str(k)] = int(S)

        manifest: Dict[str, Any] = {
            "dataset_id": str(self.ctx.dataset_id),
            "model_tag": str(self.ctx.model_tag),
            "run_id": str(self.ctx.run_id),
            "seed": int(base_seed),

            "family": FAMILY,
            "variant": VARIANT,

            "run_dir": str(self.run_dir),
            "samples_dir": str(self.run_dir / "samples" / "generated"),
            "paths": paths,
            "per_class_counts": per_class_counts,

            "img_shape": [int(H), int(W), int(C)],
            "num_classes": int(K),
            "samples_per_class": int(S),
            "num_fake": int(len(paths)),

            "sample_steps": int(T),

            "checkpoints_dir": str(self.ckpt_dir),
            "checkpoint_used": str(self._preferred_ckpt()) if self._preferred_ckpt() else None,
        }

        manifest_path = self.run_dir / "manifest.json"
        write_json(manifest_path, manifest, indent=2, sort_keys=True, atomic=True)
        self.logger.info(f"[sample] wrote manifest {manifest_path}")

        return SampleResult(
            ok=True,
            message="sampling complete",
            num_generated=int(len(paths)),
            manifest_path=str(manifest_path),
            extra={"per_class_counts": per_class_counts, "samples_dir": str(out_root)},
        )

    # -------------------------------------------------------------------------
    # Internals: checkpoint selection + model reconstruction
    # -------------------------------------------------------------------------
    def _preferred_ckpt(self) -> Optional[Path]:
        """
        Choose which checkpoint to load for sampling:
          prefer DIF_best.weights.h5, then DIF_last.weights.h5.
        """
        best = self.ckpt_dir / "DDPM_best.weights.h5"
        last = self.ckpt_dir / "DDPM_last.weights.h5"

        if best.exists():
            return best
        if last.exists():
            return last
        return None

    def _build_and_load_for_sampling(self) -> tf.keras.Model:
        """
        Build a fresh model with the same architecture and load run_scoped weights.
        """
        model = build_diffusion_model(
            img_shape=self.img_shape,
            num_classes=self.num_classes,
            base_filters=self.base_filters,
            depth=self.depth,
            time_emb_dim=self.time_emb_dim,
            learning_rate=self.lr,
            beta_1=self.beta_1,
        )

        # Build variables (Keras 3 friendly)
        H, W, C = self.img_shape
        dummy_x = tf.zeros((1, H, W, C), dtype=tf.float32)
        dummy_y = tf.one_hot([0], depth=self.num_classes, dtype=tf.float32)
        dummy_t = tf.zeros((1,), dtype=tf.int32)
        _ = model([dummy_x, dummy_y, dummy_t], training=False)

        ckpt = self._preferred_ckpt()
        if ckpt is None:
            self.logger.warning(f"[sample] no checkpoint found in {self.ckpt_dir}; using random weights.")
            return model

        try:
            model.load_weights(str(ckpt))
            self.logger.info(f"[sample] loaded checkpoint {ckpt}")
        except Exception as e:
            self.logger.warning(f"[sample] failed to load checkpoint {ckpt.name}: {e} -> using random weights.")
        return model


# =============================================================================
# Minimal CLI (optional) — keeps parity with GAN modules
# =============================================================================
def _coerce_cfg(cfg_or_argv) -> Dict[str, Any]:
    """
    Accept either:
      - a parsed dict
      - an argv list/tuple like ['--config', 'configs/config.yaml']

    Returns a dict.
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


def run(cfg_or_argv, *, do_train: bool = True, do_sample: bool = True) -> Dict[str, Any]:
    """
    Convenience entrypoint for orchestration/CLI:
      - builds pipeline (resolves RunContext)
      - optionally trains
      - optionally samples (writes manifest)
    """
    cfg = _coerce_cfg(cfg_or_argv)
    pipe = DiffusionPipeline(cfg)

    out: Dict[str, Any] = {"ok": True, "run_dir": str(pipe.run_dir), "model_tag": str(pipe.ctx.model_tag), "run_id": str(pipe.ctx.run_id)}
    if do_train:
        pipe.train()
        out["trained"] = True
    if do_sample:
        res = pipe.synthesize()
        out["sampled"] = True
        out["manifest_path"] = res.manifest_path
        out["num_generated"] = res.num_generated
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=f"c_DDPM Pipeline ({MODEL_TAG})")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"), help="Path to YAML config")
    parser.add_argument("--no_train", action="store_true", help="Skip training")
    parser.add_argument("--no_sample", action="store_true", help="Skip sampling")
    args = parser.parse_args(argv)

    # Run using the same pattern other variants use
    _ = run(["--config", str(args.config)], do_train=(not args.no_train), do_sample=(not args.no_sample))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DiffusionPipeline", "run"]
