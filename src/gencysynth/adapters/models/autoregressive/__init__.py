# src/gencysynth/adapters/models/autoregressive/__init__.py
"""
Autoregressive family adapters (PixelCNN, PixelRNN, MADE-like, etc.).
"""

from .base import AutoregressiveAdapterBase
from .registry import register_autoregressive_variant, import_autoregressive_variants

__all__ = ["AutoregressiveAdapterBase", "register_autoregressive_variant", "import_autoregressive_variants"]
