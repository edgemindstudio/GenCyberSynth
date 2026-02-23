# src/gencysynth/adapters/models/vae/registry.py
from __future__ import annotations

import importlib
from typing import Callable

from gencysynth.adapters.models.registry import register_model_adapter
from gencysynth.adapters.models.base import ModelAdapter


def register_vae_variant(variant: str, *, factory: Callable[[], ModelAdapter]) -> None:
    register_model_adapter(family="vae", variant=variant, factory=factory)


def import_vae_variants(module_path: str = "gencysynth.models.vae.variants") -> None:
    importlib.import_module(module_path)

