# src/gencysynth/adapters/registry.py
"""
Adapter registry.

We resolve adapters by `model_tag` (Rule A key used everywhere):
  - "gan/dcgan"
  - "vae/c-vae"
  - "restrictedboltzmann/c-rbm-bernoulli"
etc.

This keeps orchestration totally generic: it doesn't need to import model code.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from .contracts import Adapter
from .errors import AdapterNotFoundError

# Public diagnostic: adapters skipped due to optional import failures.
SKIPPED_IMPORTS: dict[str, str] = {}
# A factory returns a new Adapter instance (avoid cross-run state leaks).
AdapterFactory = Callable[[], Adapter]

_REGISTRY: Dict[str, AdapterFactory] = {}


def register_adapter(model_tag: str, factory: AdapterFactory) -> None:
    """
    Register an adapter factory under a canonical model_tag.

    Notes
    -----
    - model_tag should match utils.paths safe_slug expectations (may contain '/').
    - registering twice overwrites; that is intentional for local experimentation.
    """
    _REGISTRY[str(model_tag)] = factory


def resolve_adapter(model_tag: str) -> Adapter:
    """
    Instantiate and return the adapter for a given model_tag.
    """
    key = str(model_tag)
    if key not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY.keys())) if _REGISTRY else "(none)"
        raise AdapterNotFoundError(
            f"No adapter registered for model_tag='{key}'. Known: {known}. "
            "Did you import the family registry module (e.g., gencysynth.adapters.models.gan.registry)?"
        )
    return _REGISTRY[key]()


def list_adapters() -> List[str]:
    """Return registered model_tags (sorted)."""
    return sorted(_REGISTRY.keys())

def make_adapter(model_tag: str) -> Adapter:
    """CLI-friendly alias for resolve_adapter()."""
    return resolve_adapter(model_tag)
