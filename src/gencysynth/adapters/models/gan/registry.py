# src/gencysynth/adapters/models/gan/registry.py
"""
GAN family registry.

This module provides thin helpers to register GAN variants into the global adapter registry.

Pattern
-------
In a variant package (e.g., models/gan/variants/c-wgan-gp/adapter.py), do:

from gencysynth.adapters.models.gan.registry import register_gan_variant
register_gan_variant("c-wgan-gp", factory=lambda: CWGANGPAdapter())

Then orchestration resolves via:
resolve_model_adapter(family="gan", variant="c-wgan-gp")
"""

from __future__ import annotations

import importlib
from typing import Callable, Optional

from gencysynth.adapters.models.registry import register_model_adapter
from gencysynth.adapters.models.base import ModelAdapter


def register_gan_variant(variant: str, *, factory: Callable[[], ModelAdapter]) -> None:
    register_model_adapter(family="gan", variant=variant, factory=factory)


def import_gan_variants(module_path: str = "gencysynth.models.gan.variants") -> None:
    """
    Optional convenience: import the variants package so registrations run.

    If you keep a per-variant `adapter.py` that registers on import,
    this makes discovery explicit and deterministic.
    """
    importlib.import_module(module_path)
