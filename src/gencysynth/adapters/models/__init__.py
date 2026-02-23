# src/gencysynth/adapters/models/__init__.py
"""
Model adapters (front-door for orchestration).

Rule A intent
-------------
Orchestration should never import model-family internals directly.
Instead it resolves a (family, variant) -> ModelAdapter and calls a stable API:

  adapter.train(run_ctx, cfg, dataset_splits) -> TrainResult
  adapter.synthesize(run_ctx, cfg, dataset_splits) -> SynthesizeResult
  adapter.evaluate(run_ctx, cfg, dataset_splits) -> EvalResult (optional)

This package provides:
- ModelAdapter Protocol + a small BaseModelAdapter helper
- A registry mapping (family, variant) to an adapter factory

Adapters live elsewhere (typically under src/gencysynth/models/<family>/variants/<variant>/)
but register themselves here.
"""

from .base import (
    ModelAdapter,
    ModelAdapterSpec,
    TrainResult,
    SynthesizeResult,
    EvalResult,
    BaseModelAdapter,
)
from .registry import (
    register_model_adapter,
    resolve_model_adapter,
    list_model_adapters,
)

__all__ = [
    "ModelAdapter",
    "ModelAdapterSpec",
    "TrainResult",
    "SynthesizeResult",
    "EvalResult",
    "BaseModelAdapter",
    "register_model_adapter",
    "resolve_model_adapter",
    "list_model_adapters",
]
