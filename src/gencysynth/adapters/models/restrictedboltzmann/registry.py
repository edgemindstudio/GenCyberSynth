# src/gencysynth/adapters/models/restrictedboltzmann/registry.py
from __future__ import annotations

import importlib
from typing import Callable

from gencysynth.adapters.models.registry import register_model_adapter
from gencysynth.adapters.models.base import ModelAdapter


def register_rbm_variant(variant: str, *, factory: Callable[[], ModelAdapter]) -> None:
    register_model_adapter(family="restrictedboltzmann", variant=variant, factory=factory)


def import_rbm_variants(module_path: str = "gencysynth.models.restrictedboltzmann.variants") -> None:
    importlib.import_module(module_path)
