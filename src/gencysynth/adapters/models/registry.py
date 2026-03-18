# src/gencysynth/adapters/models/registry.py
"""
Model adapter registry.

Maps (family, variant) -> factory() that returns a ModelAdapter instance.

Rule A
------
- Orchestration is the only consumer.
- Model implementations register themselves, so orchestration never imports
  model_family code directly.

Example registration (inside a variant package)
----------------------------------------------
from gencysynth.adapters.models.registry import register_model_adapter

register_model_adapter(
    family="vae",
    variant="c_vae",
    factory=lambda: CVAEAdapter(),   # implements ModelAdapter
)
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from gencysynth.adapters.errors import AdapterNotFoundError
from .base import ModelAdapter


ModelAdapterFactory = Callable[[], ModelAdapter]
_REGISTRY: Dict[Tuple[str, str], ModelAdapterFactory] = {}


def register_model_adapter(*, family: str, variant: str, factory: ModelAdapterFactory) -> None:
    """
    Register an adapter factory for (family, variant).

    Overwrites are allowed to support experimentation and local development.
    """
    key = (str(family), str(variant))
    _REGISTRY[key] = factory


def resolve_model_adapter(*, family: str, variant: str) -> ModelAdapter:
    """
    Instantiate and return the model adapter for (family, variant).
    """
    key = (str(family), str(variant))
    if key not in _REGISTRY:
        known = ", ".join([f"{f}/{v}" for (f, v) in sorted(_REGISTRY.keys())]) if _REGISTRY else "(none)"
        raise AdapterNotFoundError(
            f"No model adapter registered for family='{family}', variant='{variant}'. "
            f"Known: {known}. "
            "Did you import the variant module so it registers itself?"
        )
    return _REGISTRY[key]()


def list_model_adapters() -> List[str]:
    """Return registered adapters as strings: ['family/variant', ...]."""
    return [f"{f}/{v}" for (f, v) in sorted(_REGISTRY.keys())]


def register_builtin_adapters() -> None:
    """
    Import all built_in adapter family registries so they register variants.

    This is deliberately best_effort: missing optional deps should not crash the CLI.
    """
    from gencysynth.adapters.registry import SKIPPED_IMPORTS

    families = [
        "autoregressive",
        "diffusion",
        "gan",
        "gaussianmixture",
        "maskedautoflow",
        "restrictedboltzmann",
        "vae",
    ]

    for fam in families:
        mod = f"gencysynth.adapters.models.{fam}.registry"
        try:
            __import__(mod, fromlist=["*"])
        except Exception as e:
            SKIPPED_IMPORTS[mod] = f"{type(e).__name__}: {e}"
<<<<<<< HEAD
            
def register_builtin_adapters() -> None:
    """
    Import all adapter_family registry modules so they can register variants.

    This is the single bootstrap entrypoint the CLI (or tests) should call.
    """
    import importlib

    # Importing these modules triggers filesystem discovery + register_model_adapter(...)
    for mod in [
        "gencysynth.adapters.models.gan.registry",
        "gencysynth.adapters.models.vae.registry",
        "gencysynth.adapters.models.diffusion.registry",
        "gencysynth.adapters.models.autoregressive.registry",
        "gencysynth.adapters.models.gaussianmixture.registry",
        "gencysynth.adapters.models.maskedautoflow.registry",
        "gencysynth.adapters.models.restrictedboltzmann.registry",
    ]:
        importlib.import_module(mod)
=======
>>>>>>> bd867bd (WIP: local fixes (dataset CLI + adapters); ignore uv.lock)
