# src/gencysynth/models/gan/variants/dcgan/__init__.py

"""
GenCyberSynth — GAN family — DCGAN variant (Conditional).

This package provides the DCGAN (conditional) implementation for:
- model builders (G/D/combined)
- training (CLI + router entrypoints)
- pipeline orchestration
- synthesis/sampling utilities

Folder:
    src/gencysynth/models/gan/variants/dcgan/
"""

from .pipeline import ConditionalDCGANPipeline

# Optional: keep these imports only if you want a clean external API.
# If you prefer lazy imports (faster startup), remove them.
from .train import train, main as train_main  # noqa: F401

__all__ = [
    "ConditionalDCGANPipeline",
    "train",
    "train_main",
]