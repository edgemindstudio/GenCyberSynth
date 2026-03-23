# src/gencysynth/adapters/models/gan/__init__.py
"""
GAN family adapters.

This package contains:
- GANAdapterBase: shared logic for GAN_like models (train + synth conventions)
- registry helpers to register/resolve GAN variants

Variants register themselves under:
  gencysynth.adapters.models.registry (global registry)
via this family's registry module.

from .base import GANAdapterBase
from .registry import (
    register_gan_variant,
    import_gan_variants,
)

__all__ = [
    "GANAdapterBase",
    "register_gan_variant",
    "import_gan_variants",
]
"""

# package marker (imports intentionally minimal)
