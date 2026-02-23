# src/gencysynth/adapters/models/maskedautoflow/base.py
"""
MaskedAutoFlowAdapterBase.

Flow conventions
----------------
- Flows commonly train on continuous vectors; we flatten x01.
- Optionally, variants may standardize data (mean/std) and store those params in checkpoints.
- Synthetic outputs are stored in [0,1] NHWC for evaluator.

This base provides:
- standardize_train_inputs -> (x_flat, y_onehot, img_shape)
- common synth contract writing
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from gencysynth.adapters.models.base import BaseModelAdapter, ModelAdapterSpec, TrainResult, SynthesizeResult
from gencysynth.adapters.normalize import ensure_float32, ensure_nhwc, ensure_onehot
from gencysynth.adapters.run_io import RunIO
from gencysynth.adapters.errors import AdapterContractError
from gencysynth.adapters.datasets.splits import DatasetSplits


class MaskedAutoFlowAdapterBase(BaseModelAdapter):
    def __init__(self, variant: str, description: str = "") -> None:
        super().__init__(ModelAdapterSpec(family="maskedautoflow", variant=variant, description=description))

    def _io(self, run_ctx: Any) -> RunIO:
        return self.run_io(run_ctx)

    def checkpoints_dir(self, run_ctx: Any) -> Path:
        return self._io(run_ctx).checkpoints_dir(family="maskedautoflow", variant=self.spec.variant)

    def summaries_dir(self, run_ctx: Any) -> Path:
        return self._io(run_ctx).summaries_dir(family="maskedautoflow", variant=self.spec.variant)

    def synthetic_dir(self, run_ctx: Any) -> Path:
        return self._io(run_ctx).synthetic_dir(family="maskedautoflow", variant=self.spec.variant)

    def standardize_train_inputs(self, cfg: Dict[str, Any], data: DatasetSplits) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
        x01, y1h = self._standardize_train_inputs(cfg, data)
        x01 = ensure_float32(ensure_nhwc(x01))
        y1h = ensure_onehot(ensure_float32(y1h), num_classes=int(y1h.shape[1]))
        img_shape = tuple(x01.shape[1:])
        x_flat = x01.reshape((x01.shape[0], -1))
        return x_flat, y1h, img_shape

    def write_synth_contract(self, out_dir: Path, x01: np.ndarray, y_onehot: np.ndarray) -> SynthesizeResult:
        out_dir.mkdir(parents=True, exist_ok=True)
        x01 = ensure_float32(ensure_nhwc(x01))
        y1h = ensure_float32(y_onehot)
        if y1h.ndim != 2:
            raise AdapterContractError(f"Expected y_onehot (N,K), got {y1h.shape}")
        y_int = np.argmax(y1h, axis=1).astype(np.int32)

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

    def train(self, run_ctx: Any, cfg: Dict[str, Any], data: DatasetSplits) -> TrainResult:
        ckpt_dir = self.checkpoints_dir(run_ctx)
        sum_dir = self.summaries_dir(run_ctx)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        sum_dir.mkdir(parents=True, exist_ok=True)

        x_flat, y1h, img_shape = self.standardize_train_inputs(cfg, data)
        self.log(run_ctx, "train", f"{self.spec.family}/{self.spec.variant}: x_flat {x_flat.shape}, y {y1h.shape}")
        return self._train_impl(run_ctx, cfg, data, x_flat=x_flat, y_onehot=y1h, img_shape=img_shape, ckpt_dir=ckpt_dir, summaries_dir=sum_dir)

    def synthesize(self, run_ctx: Any, cfg: Dict[str, Any], data: DatasetSplits) -> SynthesizeResult:
        out_dir = self.synthetic_dir(run_ctx)
        out_dir.mkdir(parents=True, exist_ok=True)

        x01, y1h = self._synth_impl(run_ctx, cfg, data, out_dir=out_dir)
        return self.write_synth_contract(out_dir, x01=x01, y_onehot=y1h)

    def evaluate(self, run_ctx: Any, cfg: Dict[str, Any], data: DatasetSplits):
        return None

    def _train_impl(
        self,
        run_ctx: Any,
        cfg: Dict[str, Any],
        data: DatasetSplits,
        *,
        x_flat: np.ndarray,
        y_onehot: np.ndarray,
        img_shape: Tuple[int, int, int],
        ckpt_dir: Path,
        summaries_dir: Path,
    ) -> TrainResult:
        raise NotImplementedError

    def _synth_impl(self, run_ctx: Any, cfg: Dict[str, Any], data: DatasetSplits, *, out_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
