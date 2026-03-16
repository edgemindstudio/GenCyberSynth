# src/gencysynth/adapters/models/maskedautoflow/stub.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from gencysynth.adapters.models.maskedautoflow.base import MaskedAutoFlowAdapterBase
from gencysynth.adapters.models.base import TrainResult
from gencysynth.adapters.datasets.splits import DatasetSplits


class MaskedAutoFlowStubAdapter(MaskedAutoFlowAdapterBase):
    """Placeholder used to prove registration + plumbing. Replace with real variant adapters later."""

    def __init__(self, variant: str):
        super().__init__(variant=variant, description="STUB (not implemented yet)")

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
        raise NotImplementedError(f"{self.spec.family}/{self.spec.variant}: train not implemented")

    def _synth_impl(
        self,
        run_ctx: Any,
        cfg: Dict[str, Any],
        data: DatasetSplits,
        *,
        out_dir: Path,
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(f"{self.spec.family}/{self.spec.variant}: synth not implemented")