# src/gencysynth/adapters/contracts.py
"""
Stable adapter contracts (Rule A)
================================

Adapters are the *only* layer that:
- takes a RunContext-like request (dataset_id, model_tag, run_id, artifacts_root),
- resolves canonical output paths via utils.paths,
- calls model training/sampling/eval code,
- returns structured results with stable meaning.

This keeps model implementations flexible while ensuring end-to-end
orchestration + artifact structure stays consistent across families.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from gencysynth.utils.paths import EvalPaths, LogsPaths, RunPaths


# =============================================================================
# Identity + requests
# =============================================================================
@dataclass(frozen=True)
class AdapterInfo:
    """Static identity for an adapter implementation."""
    model_tag: str                       # e.g. "gan/dcgan" or "vae/c-vae"
    family: str                          # e.g. "gan", "vae", "rbm"
    variant: str                         # e.g. "dcgan", "c-vae", "c-rbm-bernoulli"
    version: str = "0.1.0"               # adapter version (not model weights)


@dataclass(frozen=True)
class TrainRequest:
    """
    Training request passed from orchestration.

    Notes
    -----
    - `cfg` is already the merged config (base + dataset + family + variant + overrides).
    - adapters must not mutate `cfg` in-place (treat as read-only).
    """
    cfg: Dict[str, Any]
    dataset_id: str
    artifacts_root: str
    model_tag: str
    run_id: str


@dataclass(frozen=True)
class SynthRequest:
    """
    Synthesis request (generate samples).

    output_root is the *run-scoped* samples directory from RunPaths by default.
    """
    cfg: Dict[str, Any]
    dataset_id: str
    artifacts_root: str
    model_tag: str
    run_id: str
    seed: int = 42


@dataclass(frozen=True)
class EvalRequest:
    """
    Evaluation request.

    Typically used by eval/runner.py or orchestration 'onepass' mode.
    """
    cfg: Dict[str, Any]
    dataset_id: str
    artifacts_root: str
    model_tag: str
    run_id: str


# =============================================================================
# Results (thin, structured, stable)
# =============================================================================
@dataclass(frozen=True)
class AdapterResult:
    """Base result for train/synth/eval operations."""
    ok: bool
    message: str = ""
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AdapterTrainResult(AdapterResult):
    """
    Training result.

    Convention:
    - checkpoints live under RunPaths.checkpoints_dir (Rule A).
    - training logs/events live under LogsPaths (Rule A).
    """
    checkpoints_dir: Optional[str] = None


@dataclass(frozen=True)
class AdapterSynthResult(AdapterResult):
    """
    Synthesis result.

    Convention:
    - samples written under RunPaths.samples_dir (Rule A)
    - adapter should ensure a run manifest exists (RunPaths.manifest_path)
    """
    samples_dir: Optional[str] = None
    manifest_path: Optional[str] = None


@dataclass(frozen=True)
class AdapterEvalResult(AdapterResult):
    """
    Evaluation result.

    Convention:
    - evaluation outputs go under EvalPaths.root_dir (Rule A)
    """
    eval_dir: Optional[str] = None
    eval_jsonl: Optional[str] = None
    eval_latest: Optional[str] = None


# =============================================================================
# Adapter Protocol
# =============================================================================
@runtime_checkable
class Adapter(Protocol):
    """
    Stable adapter interface.

    A concrete adapter typically wraps a specific variant implementation and knows:
      - how to call train (optional for baselines),
      - how to call synth (required),
      - how to ensure artifacts are written under RunPaths/EvalPaths/LogsPaths.
    """

    info: AdapterInfo

    def train(self, req: TrainRequest, *, run_paths: RunPaths, logs_paths: LogsPaths) -> AdapterTrainResult:
        ...

    def synth(self, req: SynthRequest, *, run_paths: RunPaths, logs_paths: LogsPaths) -> AdapterSynthResult:
        ...

    def evaluate(self, req: EvalRequest, *, run_paths: RunPaths, eval_paths: EvalPaths, logs_paths: LogsPaths) -> AdapterEvalResult:
        ...
