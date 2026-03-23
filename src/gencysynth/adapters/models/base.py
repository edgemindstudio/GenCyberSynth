# src/gencysynth/adapters/models/base.py
"""
ModelAdapter protocol + common base class.

Rule A design goals
-------------------
1) Orchestration sees ONE stable interface across all model families.
2) Each adapter:
   - reads training data from DatasetSplits (standardized arrays)
   - writes artifacts ONLY under run_scoped artifact roots (run_id)
   - emits run events (optional) to a run_events log

3) Model adapters should not guess paths. They must use:
     - adapters.run_io helpers (canonical run paths)
     - utils.paths for low_level joins

This file defines:
- ModelAdapterSpec: stable identity
- Results dataclasses
- ModelAdapter Protocol
- BaseModelAdapter: a convenience base with logging + safe cfg getters
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable

from gencysynth.adapters.errors import AdapterContractError
from gencysynth.adapters.normalize import ensure_nhwc, ensure_float32
from gencysynth.adapters.run_io import RunIO

from gencysynth.adapters.datasets.splits import DatasetSplits


# -----------------------------
# Identity & result contracts
# -----------------------------
@dataclass(frozen=True)
class ModelAdapterSpec:
    """
    Stable identity for a model adapter.

    family:  high_level family name (gan, vae, diffusion, rbm, ...)
    variant: concrete variant folder name (c_vae, c_wgan_gp, c_rbm_bernoulli, ...)
    """
    family: str
    variant: str
    description: str = ""


@dataclass(frozen=True)
class TrainResult:
    """
    Minimal return payload from training.

    Paths should be *run_scoped* (i.e., inside the current run's artifacts root).
    """
    checkpoints_dir: str
    summaries_dir: Optional[str] = None
    best_checkpoint: Optional[str] = None
    last_checkpoint: Optional[str] = None


@dataclass(frozen=True)
class SynthesizeResult:
    """
    Minimal return payload from synthesis.

    - synth_dir: folder containing evaluator contract files:
        gen_class_{k}.npy, labels_class_{k}.npy, x_synth.npy, y_synth.npy
    """
    synth_dir: str
    x_path: str
    y_path: str


@dataclass(frozen=True)
class EvalResult:
    """
    Optional return payload from evaluation (if adapter handles it directly).

    Usually evaluation is centralized in gencysynth.eval, so adapters can skip this.
    """
    eval_dir: str
    summary_path: Optional[str] = None


# -----------------------------
# Protocol
# -----------------------------
@runtime_checkable
class ModelAdapter(Protocol):
    """
    Stable adapter interface used by orchestration.

    Each method is expected to:
      - operate deterministically when run_ctx.seed is fixed
      - write artifacts under run_scoped paths (RunIO)
      - avoid global "artifacts/<family>/" paths without run_id separation

    run_ctx is intentionally typed as Any to avoid tight coupling:
    orchestration provides it (includes run_id, seed, paths, event writer, etc.).
    """
    spec: ModelAdapterSpec

    def train(self, run_ctx: Any, cfg: Dict[str, Any], data: DatasetSplits) -> TrainResult:
        ...

    def synthesize(self, run_ctx: Any, cfg: Dict[str, Any], data: DatasetSplits) -> SynthesizeResult:
        ...

    def evaluate(self, run_ctx: Any, cfg: Dict[str, Any], data: DatasetSplits) -> Optional[EvalResult]:
        ...


# -----------------------------
# Convenience base class
# -----------------------------
class BaseModelAdapter:
    """
    Small helper base class for adapters.

    This is NOT required, but it reduces repeated boilerplate:
    - cfg getters
    - standardized logging
    - standardized RunIO access
    """

    spec: ModelAdapterSpec

    def __init__(self, spec: ModelAdapterSpec) -> None:
        self.spec = spec

    # ---- cfg utilities ----
    @staticmethod
    def cfg_get(cfg: Dict[str, Any], dotted: str, default=None):
        cur: Any = cfg
        for key in dotted.split("."):
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur

    # ---- logging ----
    def log(self, run_ctx: Any, stage: str, msg: str) -> None:
        """
        Prefer run_ctx event writer if present; fallback to print.

        run_ctx may provide:
          - run_ctx.log(stage, msg)
          - run_ctx.events.write(...)
        """
        if hasattr(run_ctx, "log"):
            try:
                run_ctx.log(stage, msg)
                return
            except Exception:
                pass
        print(f"[{stage}] {msg}")

    # ---- standardization ----
    def _assert_basic(self, cfg: Dict[str, Any], data: DatasetSplits) -> None:
        """
        Fail fast if dataset splits aren't usable for training/synthesis.
        """
        if data is None or data.train is None:
            raise AdapterContractError("DatasetSplits.train is required.")
        if data.train.x01 is None or data.train.y_int is None or data.train.y_onehot is None:
            raise AdapterContractError("DatasetSplits.train must include x01, y_int, y_onehot arrays.")

    def _standardize_train_inputs(self, cfg: Dict[str, Any], data: DatasetSplits) -> Tuple[Any, Any]:
        """
        Returns:
          x01: float32 NHWC [0,1]
          y_onehot: float32 (N,K)

        Adapters that need other ranges (e.g. [-1,1]) should convert from x01.
        """
        self._assert_basic(cfg, data)
        x01 = ensure_float32(ensure_nhwc(data.train.x01))
        y1h = ensure_float32(data.train.y_onehot)
        return x01, y1h

    # ---- RunIO helper ----
    def run_io(self, run_ctx: Any) -> RunIO:
        """
        Standard way to obtain canonical paths for this run.
        """
        if hasattr(run_ctx, "io") and isinstance(run_ctx.io, RunIO):
            return run_ctx.io
        # Fallback: build from run_ctx if it contains run_root / artifacts_root
        return RunIO.from_run_ctx(run_ctx)
