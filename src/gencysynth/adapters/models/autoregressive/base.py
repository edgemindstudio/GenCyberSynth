# src/gencysynth/adapters/models/autoregressive/base.py
"""
AutoregressiveAdapterBase.

Autoregressive conventions
--------------------------
- Some AR models train on discrete pixels (0..255) or normalized [0,1].
- Rule A provides x01; variants may quantize or keep float.
- Synthetic outputs are stored in [0,1] for evaluator.

This base defaults to training on x01.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from gencysynth.adapters.models.base import (
    BaseModelAdapter,
    ModelAdapterSpec,
    TrainResult,
    SynthesizeResult,
)
from gencysynth.adapters.normalize import ensure_float32, ensure_onehot, to_01
from gencysynth.adapters.run_io import RunIO
from gencysynth.adapters.errors import AdapterContractError
from gencysynth.adapters.datasets.splits import DatasetSplits


def _ensure_nhwc_best_effort(x: np.ndarray, *, cfg: Dict[str, Any]) -> np.ndarray:
    """
    Best_effort ensure NHWC.

    Accepts:
      - (N,H,W,C) -> returned as_is
      - (N, H*W*C) -> reshapes using cfg['dataset']['image_hw'] + channels=1 by default

    Recommended config:
      cfg['dataset']['image_hw'] = [H,W]
      cfg['dataset']['channels'] = C
    """
    x = np.asarray(x)
    if x.ndim == 4:
        return x

    if x.ndim == 2:
        hw = cfg.get("dataset", {}).get("image_hw", [40, 40])
        ch = cfg.get("dataset", {}).get("channels", 1)
        try:
            H, W = int(hw[0]), int(hw[1])
            C = int(ch)
        except Exception:
            H, W, C = 40, 40, 1
        D = H * W * C
        if x.shape[1] != D:
            raise AdapterContractError(
                f"Flattened images have D={x.shape[1]} but expected H*W*C={D} "
                f"from dataset.image_hw={hw}, channels={ch}."
            )
        return x.reshape((-1, H, W, C))

    raise AdapterContractError(f"Expected images as (N,H,W,C) or (N,D). Got shape {x.shape}.")


class AutoregressiveAdapterBase(BaseModelAdapter):
    def __init__(self, variant: str, description: str = "") -> None:
        super().__init__(ModelAdapterSpec(family="autoregressive", variant=variant, description=description))

    # ---- RunIO helpers ----
    def _io(self, run_ctx: Any) -> RunIO:
        return self.run_io(run_ctx)

    def _run_root(self, run_ctx: Any) -> Path:
        return self._io(run_ctx).run_paths.root_dir

    def checkpoints_dir(self, run_ctx: Any) -> Path:
        return self._io(run_ctx).run_paths.checkpoints_dir

    def summaries_dir(self, run_ctx: Any) -> Path:
        return self._run_root(run_ctx) / "summaries"

    def synthetic_dir(self, run_ctx: Any) -> Path:
        return self._run_root(run_ctx) / "synthetic"

    # ---- inputs ----
    def standardize_train_inputs(self, cfg: Dict[str, Any], data: DatasetSplits) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          x01: float32 NHWC in [0,1]
          y1h: float32 (N,K) one_hot
        """
        self._assert_basic(cfg, data)

        x01 = ensure_float32(_ensure_nhwc_best_effort(data.train.x01, cfg=cfg))
        y1h = ensure_float32(data.train.y_onehot)

        if y1h.ndim != 2:
            raise AdapterContractError(f"Expected train.y_onehot as (N,K); got {y1h.shape}")

        y1h = ensure_onehot(y1h, num_classes=int(y1h.shape[1]))
        x01 = to_01(x01)
        return x01, y1h

    # ---- contract writing ----
    def write_synth_contract(self, out_dir: Path, x01: np.ndarray, y_onehot: np.ndarray) -> SynthesizeResult:
        """
        Writes evaluator contract files into out_dir:
          - gen_class_{k}.npy
          - labels_class_{k}.npy
          - x_synth.npy
          - y_synth.npy (one_hot)
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        x01 = ensure_float32(_ensure_nhwc_best_effort(x01, cfg=cfg_for_hw(x01)))
        x01 = to_01(x01)

        y1h = ensure_float32(y_onehot)
        if y1h.ndim != 2:
            raise AdapterContractError(f"Expected y_onehot (N,K), got {y1h.shape}")

        K = int(y1h.shape[1])
        y1h = ensure_onehot(y1h, num_classes=K)
        y_int = np.argmax(y1h, axis=1).astype(np.int32)

        for k in range(K):
            idx = (y_int == k)
            np.save(out_dir / f"gen_class_{k}.npy", x01[idx])
            np.save(out_dir / f"labels_class_{k}.npy", np.full((int(idx.sum()),), k, dtype=np.int32))

        x_path = out_dir / "x_synth.npy"
        y_path = out_dir / "y_synth.npy"
        np.save(x_path, x01)
        np.save(y_path, y1h)

        return SynthesizeResult(synth_dir=str(out_dir), x_path=str(x_path), y_path=str(y_path))

    # ---- public API ----
    def train(self, run_ctx: Any, cfg: Dict[str, Any], data: DatasetSplits) -> TrainResult:
        io = self._io(run_ctx)
        io.ensure_dirs(include_eval=False)

        ckpt_dir = self.checkpoints_dir(run_ctx)
        sum_dir = self.summaries_dir(run_ctx)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        sum_dir.mkdir(parents=True, exist_ok=True)

        x01, y1h = self.standardize_train_inputs(cfg, data)
        self.log(run_ctx, "train", f"{self.spec.family}/{self.spec.variant}: x {x01.shape} in [0,1], y {y1h.shape}")

        return self._train_impl(
            run_ctx,
            cfg,
            data,
            x01=x01,
            y_onehot=y1h,
            ckpt_dir=ckpt_dir,
            summaries_dir=sum_dir,
        )

    def synthesize(self, run_ctx: Any, cfg: Dict[str, Any], data: DatasetSplits) -> SynthesizeResult:
        io = self._io(run_ctx)
        io.ensure_dirs(include_eval=False)

        out_dir = self.synthetic_dir(run_ctx)
        out_dir.mkdir(parents=True, exist_ok=True)

        x01, y1h = self._synth_impl(run_ctx, cfg, data, out_dir=out_dir)
        return self.write_synth_contract(out_dir, x01=x01, y_onehot=y1h)

    def evaluate(self, run_ctx: Any, cfg: Dict[str, Any], data: DatasetSplits):
        return None

    # ---- variant hooks ----
    def _train_impl(
        self,
        run_ctx: Any,
        cfg: Dict[str, Any],
        data: DatasetSplits,
        *,
        x01: np.ndarray,
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
          x01: float32 NHWC in [0,1]
          y1h: float32 one_hot (N,K)
        """
        raise NotImplementedError


def cfg_for_hw(x01: np.ndarray) -> Dict[str, Any]:
    """
    Utility to generate a minimal cfg dict with dataset H/W/C derived from an NHWC tensor,
    so _ensure_nhwc_best_effort can validate/reshape consistently.
    """
    if x01 is None:
        return {"dataset": {"image_hw": [40, 40], "channels": 1}}

    x = np.asarray(x01)
    if x.ndim == 4:
        H, W, C = int(x.shape[1]), int(x.shape[2]), int(x.shape[3])
        return {"dataset": {"image_hw": [H, W], "channels": C}}
    return {"dataset": {"image_hw": [40, 40], "channels": 1}}