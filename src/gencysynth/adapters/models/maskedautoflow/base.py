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
    Best-effort ensure NHWC.

    Accepts:
      - (N,H,W,C) -> returned as-is
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


def _cfg_for_hw(x: np.ndarray) -> Dict[str, Any]:
    """Utility: derive dataset.image_hw + channels from an NHWC array (or fallback)."""
    if x is None:
        return {"dataset": {"image_hw": [40, 40], "channels": 1}}
    arr = np.asarray(x)
    if arr.ndim == 4:
        H, W, C = int(arr.shape[1]), int(arr.shape[2]), int(arr.shape[3])
        return {"dataset": {"image_hw": [H, W], "channels": C}}
    return {"dataset": {"image_hw": [40, 40], "channels": 1}}


class MaskedAutoFlowAdapterBase(BaseModelAdapter):
    def __init__(self, variant: str, description: str = "") -> None:
        super().__init__(ModelAdapterSpec(family="maskedautoflow", variant=variant, description=description))

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
    def standardize_train_inputs(self, cfg: Dict[str, Any], data: DatasetSplits) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
        """
        Returns:
          x_flat: float32 (N, H*W*C) in [0,1]
          y1h   : float32 (N,K) one-hot
          img_shape: (H,W,C)
        """
        self._assert_basic(cfg, data)

        x01 = ensure_float32(_ensure_nhwc_best_effort(data.train.x01, cfg=cfg))
        x01 = to_01(x01)

        y1h = ensure_float32(data.train.y_onehot)
        if y1h.ndim != 2:
            raise AdapterContractError(f"Expected train.y_onehot as (N,K); got {y1h.shape}")

        K = int(y1h.shape[1])
        y1h = ensure_onehot(y1h, num_classes=K)

        img_shape = (int(x01.shape[1]), int(x01.shape[2]), int(x01.shape[3]))
        x_flat = x01.reshape((x01.shape[0], -1))
        return x_flat, y1h, img_shape

    # ---- contract writing ----
    def write_synth_contract(self, out_dir: Path, x01: np.ndarray, y_onehot: np.ndarray) -> SynthesizeResult:
        """
        Writes evaluator contract files into out_dir:
          - gen_class_{k}.npy
          - labels_class_{k}.npy
          - x_synth.npy
          - y_synth.npy (one-hot)
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        x01 = ensure_float32(_ensure_nhwc_best_effort(x01, cfg=_cfg_for_hw(x01)))
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

        x_flat, y1h, img_shape = self.standardize_train_inputs(cfg, data)
        self.log(run_ctx, "train", f"{self.spec.family}/{self.spec.variant}: x_flat {x_flat.shape}, y {y1h.shape}")

        return self._train_impl(
            run_ctx,
            cfg,
            data,
            x_flat=x_flat,
            y_onehot=y1h,
            img_shape=img_shape,
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
        x_flat: np.ndarray,
        y_onehot: np.ndarray,
        img_shape: Tuple[int, int, int],
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
          y1h: float32 one-hot (N,K)
        """
        raise NotImplementedError
