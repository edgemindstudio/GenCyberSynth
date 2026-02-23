# src/gencysynth/utils/run_id.py
"""
GenCyberSynth — Run identifier helpers

A run_id should be:
- stable (same inputs -> same output)
- short (fits nicely in paths)
- expressive (encodes config + seed at minimum)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from gencysynth.utils.paths import safe_slug


@dataclass(frozen=True)
class RunIdParts:
    config: str          # e.g., "A" or "B" or "baseline"
    seed: int            # e.g., 42
    extra: Optional[str] = None  # e.g., "cap200" or "tlite"


def make_run_id(config: str, seed: int, extra: Optional[str] = None) -> str:
    """
    Create a filesystem-safe run id.

    Examples:
      make_run_id("A", 42) -> "A_seed42"
      make_run_id("B", 7, "cap200") -> "B_seed7_cap200"
    """
    base = f"{config}_seed{int(seed)}"
    if extra:
        base += f"_{extra}"
    return safe_slug(base)
