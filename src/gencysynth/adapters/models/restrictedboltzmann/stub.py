# src/gencysynth/adapters/models/restrictedboltzmann/stub.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from gencysynth.adapters.models.base import BaseModelAdapter, ModelAdapterSpec, TrainResult, SynthesizeResult
from gencysynth.adapters.datasets.splits import DatasetSplits


class RBMStubAdapter(BaseModelAdapter):
    """
    Placeholder adapter for RBM family.
    NOTE: RBM base not provided above; keep this stub minimal but registry-visible.
    """

    def __init__(self, variant: str):
        super().__init__(ModelAdapterSpec(family="restrictedboltzmann", variant=variant, description="STUB"))

    def train(self, run_ctx: Any, cfg: Dict[str, Any], data: DatasetSplits) -> TrainResult:
        raise NotImplementedError(f"{self.spec.family}/{self.spec.variant}: train not implemented")

    def synthesize(self, run_ctx: Any, cfg: Dict[str, Any], data: DatasetSplits) -> SynthesizeResult:
        raise NotImplementedError(f"{self.spec.family}/{self.spec.variant}: synth not implemented")

    def evaluate(self, run_ctx: Any, cfg: Dict[str, Any], data: DatasetSplits):
        return None