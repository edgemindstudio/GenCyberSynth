# src/gencysynth/utils/reproducibility.py
"""
GenCyberSynth — Reproducibility helpers

This module centralizes:
- UTC timestamps used for audit logs and manifests
- seeding for NumPy and TensorFlow
"""

from __future__ import annotations

import datetime as _dt
from typing import Optional


def now_iso() -> str:
    """
    Return a UTC timestamp in RFC3339 / ISO_8601 with 'Z' suffix (no microseconds).
    Example: '2026_02_08T11:23:45Z'
    """
    return _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def set_global_seed(seed: int = 42, *, tf_deterministic: bool = False) -> None:
    """
    Set RNG seeds for reproducibility across NumPy and TensorFlow.

    tf_deterministic:
      - If True, attempt to enable deterministic TF ops (best_effort).
      - On some systems/kernels this may be ignored.
    """
    import numpy as np

    np.random.seed(int(seed))

    try:
        import tensorflow as tf
        tf.random.set_seed(int(seed))
        if tf_deterministic:
            # Best_effort determinism flag
            try:
                tf.config.experimental.enable_op_determinism()
            except Exception:
                pass
    except Exception:
        # TensorFlow may not be installed in some environments
        pass
