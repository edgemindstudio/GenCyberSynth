# src/gencysynth/adapters/models/restrictedboltzmann/__init__.py
"""
Restricted Boltzmann Machine (RBM) family adapters.
"""

from .base import RBMAdapterBase
from .registry import register_rbm_variant, import_rbm_variants

__all__ = ["RBMAdapterBase", "register_rbm_variant", "import_rbm_variants"]
