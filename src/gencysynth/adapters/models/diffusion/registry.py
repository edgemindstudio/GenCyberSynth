# src/gencysynth/adapters/models/diffusion/registry.py
from __future__ import annotations

import importlib
from typing import Callable

from gencysynth.adapters.models.registry import register_model_adapter
from gencysynth.adapters.models.base import ModelAdapter


def register_diffusion_variant(variant: str, *, factory: Callable[[], ModelAdapter]) -> None:
    register_model_adapter(family="diffusion", variant=variant, factory=factory)


def import_diffusion_variants(module_path: str = "gencysynth.models.diffusion.variants") -> None:
    importlib.import_module(module_path)
