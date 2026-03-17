# src/gencysynth/models/base_types.py
"""
GenCyberSynth — Model Base Types (family/variant scalable)

This module defines:
- canonical model identity types (model_tag, run_id)
- core interfaces for training and sampling
- small, framework_agnostic records used across orchestration/eval

Why this exists
---------------
Your repo will contain:
- many model families (gan/vae/diffusion/...)
- many variants (gan/dcgan, gan/wgangp, diffusion/ddpm, ...)
- many datasets

To stay scalable, we standardize the "identity" of a run and the minimal API
every model should implement. This makes orchestration/eval tooling uniform.

Rule of the road
----------------
Everything is keyed by:
    (dataset_id, model_tag, run_id)

So all outputs are collision_free:
    artifacts/runs/<dataset_id>/<model_tag>/<run_id>/...
    artifacts/eval/<dataset_id>/<model_tag>/<run_id>/...
    artifacts/logs/<dataset_id>/<model_tag>/<run_id>/...

Design goals
------------
- Lightweight: no TensorFlow/PyTorch import required here.
- Clear contracts: minimal methods so pipelines can treat models uniformly.
- Flexible: supports both "train+sample" models and "sample_only" models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol, Tuple, runtime_checkable


# -----------------------------------------------------------------------------
# Canonical identifiers
# -----------------------------------------------------------------------------
ModelTag = str
RunId = str
DatasetId = str


@dataclass(frozen=True)
class RunContext:
    """
    Canonical identity + output locations for a single run.

    This is the *single object* you can pass around to ensure every component
    is writing under the correct dataset_scoped paths.

    Fields
    ------
    dataset_id:
        Stable dataset identifier, e.g. "USTC_TFC2016_40x40_gray"
    model_tag:
        Variant_aware model tag, e.g. "gan/dcgan", "diffusion/ddpm"
    run_id:
        Stable run identifier for this (dataset_id, model_tag) run, e.g. "A_seed42"
    seed:
        RNG seed (propagated consistently across pipeline)
    artifacts_root:
        Root artifacts directory (default in configs: "artifacts")

    resolved directories:
        run_dir:  artifacts/runs/<dataset_id>/<model_tag>/<run_id>/
        logs_dir: artifacts/logs/<dataset_id>/<model_tag>/<run_id>/
        eval_dir: artifacts/eval/<dataset_id>/<model_tag>/<run_id>/
    """
    dataset_id: DatasetId
    model_tag: ModelTag
    run_id: RunId
    seed: int
    artifacts_root: str = "artifacts"

    run_dir: Optional[Path] = None
    logs_dir: Optional[Path] = None
    eval_dir: Optional[Path] = None


@dataclass(frozen=True)
class TrainResult:
    """
    Minimal training result record.

    NOTE: We keep this general; framework_specific artifacts (ckpts, weights)
    should live under RunContext.run_dir and be referenced via relative paths.
    """
    ok: bool
    message: str = ""
    best_ckpt_path: Optional[str] = None  # relative to run_dir preferred
    extra: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class SampleResult:
    """
    Sampling result record.

    This is designed to map cleanly to a run manifest written by orchestration:
      - num_generated
      - manifest_path
      - optional per_class counts
    """
    ok: bool
    message: str = ""
    num_generated: Optional[int] = None
    manifest_path: Optional[str] = None  # absolute or relative to run_dir
    extra: Optional[Dict[str, Any]] = None


# -----------------------------------------------------------------------------
# Model interface contracts
# -----------------------------------------------------------------------------
@runtime_checkable
class GenModel(Protocol):
    """
    Minimal interface for a generative model family/variant.

    A model implementation is expected to be a small wrapper around a training
    and sampling pipeline (TF/PyTorch/NumPy/etc.), but this Protocol is kept
    framework_agnostic.

    Required properties
    -------------------
    model_tag:
        A stable tag like "gan/dcgan" used across configs/artifacts/eval.

    Required methods
    ----------------
    train(cfg, ctx):
        Train the model and write artifacts under ctx.run_dir.

    sample(cfg, ctx):
        Generate synthetic samples and write a run manifest under ctx.run_dir
        (or another known location) and return SampleResult.
    """
    model_tag: str

    def train(self, cfg: Mapping[str, Any], ctx: RunContext) -> TrainResult:
        ...

    def sample(self, cfg: Mapping[str, Any], ctx: RunContext) -> SampleResult:
        ...


@runtime_checkable
class SampleOnlyModel(Protocol):
    """
    Interface for models that only support sampling (no training in this repo),
    e.g., pre_trained external generators or baseline samplers.
    """
    model_tag: str

    def sample(self, cfg: Mapping[str, Any], ctx: RunContext) -> SampleResult:
        ...

@runtime_checkable
class RunnableModel(Protocol):
    model_tag: str
    def run(self, cfg: Mapping[str, Any], ctx: RunContext) -> SampleResult:
        ...


# -----------------------------------------------------------------------------
# Registry entry (what gets registered)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    """
    Registry entry describing one concrete model implementation.

    Fields
    ------
    model_tag:
        Unique key used in config and artifacts: "gan/dcgan"
    family:
        High_level family name: "gan", "vae", "diffusion"
    variant:
        Variant name: "dcgan", "wgangp", "ddpm"
    impl:
        Dotted import path to a callable that builds the model:
            "gencysynth.models.gan.variants.dcgan:build"
        The callable signature should be:
            build(cfg: Mapping[str, Any]) -> GenModel
    """
    model_tag: str
    family: str
    variant: str
    impl: str
    description: str = ""


__all__ = [
    "DatasetId",
    "ModelTag",
    "RunId",
    "RunContext",
    "TrainResult",
    "SampleResult",
    "GenModel",
    "SampleOnlyModel",
    "ModelSpec",
]