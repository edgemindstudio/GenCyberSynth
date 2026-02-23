# src/gencysynth/adapters/models/vae/__init__.py
"""
VAE family adapters.

Contains:
- VAEAdapterBase: common conventions for VAE-like models
- registry helpers to register VAE variants
"""

from .base import VAEAdapterBase
from .registry import register_vae_variant, import_vae_variants

__all__ = ["VAEAdapterBase", "register_vae_variant", "import_vae_variants"]
