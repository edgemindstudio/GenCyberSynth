# src/gencysynth/adapters/models/gaussianmixture/registry.py

from __future__ import annotations

from pathlib import Path
from typing import Callable

from gencysynth.adapters.models.registry import register_model_adapter
from gencysynth.adapters.models.base import ModelAdapter
from gencysynth.adapters.registry import SKIPPED_IMPORTS
from gencysynth.adapters.models.gaussianmixture.stub import GaussianMixtureStubAdapter


def register_gaussianmixture_variant(variant: str, *, factory: Callable[[], ModelAdapter]) -> None:
    register_model_adapter(family="gaussianmixture", variant=variant, factory=factory)


def _variants_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "models" / "gaussianmixture" / "variants"


def register_all_gaussianmixture_variants() -> None:
    vdir = _variants_dir()
    if not vdir.exists():
        SKIPPED_IMPORTS["gencysynth.adapters.models.gaussianmixture.registry"] = f"variants dir missing: {vdir}"
        return
    for p in sorted([x for x in vdir.iterdir() if x.is_dir() and not x.name.startswith("_")]):
        register_gaussianmixture_variant(p.name, factory=lambda v=p.name: GaussianMixtureStubAdapter(variant=v))


try:
    register_all_gaussianmixture_variants()
except Exception as e:
    SKIPPED_IMPORTS["gencysynth.adapters.models.gaussianmixture.registry"] = f"{type(e).__name__}: {e}"
