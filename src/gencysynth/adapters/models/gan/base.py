# src/gencysynth/adapters/models/gan/base.py
"""
GANAdapterBase.

Rule A responsibilities
-----------------------
A GAN adapter is responsible for:
1) Training:
   - consume DatasetSplits (train/val/test)
   - write checkpoints and summaries under the run's artifact root (RunIO)
2) Synthesis:
   - write evaluator contract files under run synthetic dir:
       gen_class_{k}.npy, labels_class_{k}.npy, x_synth.npy, y_synth.npy
   - optionally write PNG galleries (run-scoped) if enabled

This base class does NOT implement training/sampling itself.
It only provides *consistent* paths, logging, and small helpers shared by GAN variants.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from gencysynth.adapters.models.base import BaseModelAdapter, ModelAdapterSpec, TrainResult, SynthesizeResult
from gencysynth.adapters.normalize import ensure_float32, ensure_nhwc, ensure_onehot, to_minus1_1, from_minus1_1
from gencysynth.adapters.run_io import RunIO
from gencysynth.adapters.errors import AdapterContractError
from gencysynth.adapters.datasets.splits import DatasetSplits


class GANAdapterBase(BaseModelAdapter):
    """
    Common GAN adapter behavior: standardized inputs/outputs + canonical run paths.

    Concrete variants should:
      - subclass this
      - implement _train_impl(...) and _synth_impl(...)
    """

    def __init__(self, variant: str, description: str = "") -> None:
        super().__init__(ModelAdapterSpec(family="gan", variant=variant, description=description))

    # -----------------------------
    # Canonical run paths
    # -----------------------------
    def _io(self, run_ctx: Any) -> RunIO:
        return self.run_io(run_ctx)

    def checkpoints_dir(self, run_ctx: Any) -> Path:
        return self._io(run_ctx).checkpoints_dir(family="gan", variant=self.spec.variant)

    def summaries_dir(self, run_ctx: Any) -> Path:
        return self._io(run_ctx).summaries_dir(family="gan", variant=self.spec.variant)

    def synthetic_dir(self, run_ctx: Any) -> Path:
        return self._io(run_ctx).synthetic_dir(family="gan", variant=self.spec.variant)

    def plots_dir(self, run_ctx: Any) -> Path:
        return self._io(run_ctx).plots_dir(family="gan", variant=self.spec.variant)

    # -----------------------------
    # Input standardization
    # -----------------------------
    def standardize_train_inputs(self, cfg: Dict[str, Any], data: DatasetSplits) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns x_m11, y_onehot for GAN training.
        GANs typically use tanh output, so we standardize to x in [-1,1].
        """
        x01, y1h = self._standardize_train_inputs(cfg, data)  # from BaseModelAdapter
        y1h = ensure_onehot(y1h, num_classes=int(y1h.shape[1]))
        x_m11 = to_minus1_1(x01)
        return ensure_float32(ensure_nhwc(x_m11)), ensure_float32(y1h)

    # -----------------------------
    # Contract writing helper
    # -----------------------------
    def write_synth_contract(self, out_dir: Path, x01: np.ndarray, y_onehot: np.ndarray) -> SynthesizeResult:
        """
        Write evaluator-friendly synthetic outputs to out_dir.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        x01 = ensure_float32(ensure_nhwc(x01))
        y1h = ensure_float32(y_onehot)
        if y1h.ndim != 2:
            raise AdapterContractError(f"Expected y_onehot (N,K), got {y1h.shape}")
        y_int = np.argmax(y1h, axis=1).astype(np.int32)

        # Per-class dumps
        K = int(y1h.shape[1])
        for k in range(K):
            idx = (y_int == k)
            np.save(out_dir / f"gen_class_{k}.npy", x01[idx])
            np.save(out_dir / f"labels_class_{k}.npy", np.full((int(idx.sum()),), k, dtype=np.int32))

        x_path = out_dir / "x_synth.npy"
        y_path = out_dir / "y_synth.npy"
        np.save(x_path, x01)
        np.save(y_path, y1h)

        return SynthesizeResult(synth_dir=str(out_dir), x_path=str(x_path), y_path=str(y_path))

    # -----------------------------
    # Public adapter API
    # -----------------------------
    def train(self, run_ctx: Any, cfg: Dict[str, Any], data: DatasetSplits) -> TrainResult:
        ckpt_dir = self.checkpoints_dir(run_ctx)
        sum_dir = self.summaries_dir(run_ctx)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        sum_dir.mkdir(parents=True, exist_ok=True)

        x_m11, y1h = self.standardize_train_inputs(cfg, data)
        self.log(run_ctx, "train", f"{self.spec.family}/{self.spec.variant}: x {x_m11.shape} in [-1,1], y {y1h.shape}")

        return self._train_impl(run_ctx, cfg, data, x_m11=x_m11, y_onehot=y1h, ckpt_dir=ckpt_dir, summaries_dir=sum_dir)

    def synthesize(self, run_ctx: Any, cfg: Dict[str, Any], data: DatasetSplits) -> SynthesizeResult:
        out_dir = self.synthetic_dir(run_ctx)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.log(run_ctx, "synth", f"{self.spec.family}/{self.spec.variant}: writing synth to {out_dir}")

        # Delegate to variant impl, but enforce that it returns x in [0,1] + y_onehot
        x01, y1h = self._synth_impl(run_ctx, cfg, data, out_dir=out_dir)
        return self.write_synth_contract(out_dir, x01=x01, y_onehot=y1h)

    def evaluate(self, run_ctx: Any, cfg: Dict[str, Any], data: DatasetSplits):
        # Usually centralized in gencysynth.eval; variants may override.
        return None

    # -----------------------------
    # Variant hooks
    # -----------------------------
    def _train_impl(
        self,
        run_ctx: Any,
        cfg: Dict[str, Any],
        data: DatasetSplits,
        *,
        x_m11: np.ndarray,
        y_onehot: np.ndarray,
        ckpt_dir: Path,
        summaries_dir: Path,
    ) -> TrainResult:
        raise NotImplementedError

    def _synth_impl(
        self,
        run_ctx: Any,
        cfg: Dict[str, Any],
        data: DatasetSplits,
        *,
        out_dir: Path,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Must return:
          x01: (N,H,W,C) float32 in [0,1]
          y1h: (N,K) float32 one-hot
        """
        raise NotImplementedError
