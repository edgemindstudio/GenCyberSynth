# src/gencysynth/adapters/__init__.py
"""
GenCyberSynth Adapters
======================

Adapters are the stable interface between:
  - orchestration (run context, run ids, dataset selection)
  - model implementations (GAN/VAE/RBM/etc. variants)

Rule A (dataset_scalable paths)
-------------------------------
Adapters MUST read/write artifacts using (dataset_id, model_tag, run_id) via
utils.paths resolvers (never hardcode artifacts/vae/... etc).

Public API
----------
- contracts:   Protocols + dataclasses defining the adapter contract
- registry:    registration + resolution of adapters (by model_tag)
- run_io:      canonical artifact path helpers (wrappers over utils.paths)
- normalize:   shared normalization helpers (labels, shapes, ranges)
"""

from .contracts import (
    Adapter,
    AdapterInfo,
    AdapterResult,
    AdapterTrainResult,
    AdapterSynthResult,
    AdapterEvalResult,
    TrainRequest,
    SynthRequest,
    EvalRequest,
)
from .errors import (
    AdapterError,
    AdapterNotFoundError,
    AdapterContractError,
)
from .registry import (
    register_adapter,
    resolve_adapter,
    list_adapters,
)
from .run_io import (
    resolve_run_paths,
    resolve_eval_paths,
    resolve_logs_paths,
    ensure_run_dirs,
)

__all__ = [
    # contracts
    "Adapter",
    "AdapterInfo",
    "AdapterResult",
    "AdapterTrainResult",
    "AdapterSynthResult",
    "AdapterEvalResult",
    "TrainRequest",
    "SynthRequest",
    "EvalRequest",
    # errors
    "AdapterError",
    "AdapterNotFoundError",
    "AdapterContractError",
    # registry
    "register_adapter",
    "resolve_adapter",
    "list_adapters",
    # run_io
    "resolve_run_paths",
    "resolve_eval_paths",
    "resolve_logs_paths",
    "ensure_run_dirs",
]
