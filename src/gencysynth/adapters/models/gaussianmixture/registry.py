# src/gencysynth/adapters/models/gaussianmixture/registry.py
from __future__ import annotations

import importlib
from typing import Callable

from gencysynth.adapters.models.registry import register_model_adapter
from gencysynth.adapters.models.base import ModelAdapter


def register_gaussianmixture_variant(variant: str, *, factory: Callable[[], ModelAdapter]) -> None:
    register_model_adapter(family="gaussianmixture", variant=variant, factory=factory)


def import_gaussianmixture_variants(module_path: str = "gencysynth.models.gaussianmixture.variants") -> None:
    importlib.import_module(module_path)
