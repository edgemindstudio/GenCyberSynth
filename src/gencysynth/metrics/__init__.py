# src/gencysynth/metrics/__init__.py
"""
GenCyberSynth metrics package.

What lives here
---------------
- api: evaluate(...) entrypoint for orchestrators
- registry: metric plugin registry
- preprocess/contracts: consistent input normalization and validation
- writer: Rule A artifacts output

Registering metrics
-------------------
Metric plugins are registered at import-time in gencysynth.metrics.registry
by importing gencysynth.metrics (or gencysynth.metrics.registry bootstrap).

In this package we register a small default set of metrics:
- sanity.basic_stats
- sanity.shape_checks
- distribution.pixel_hist_l1
"""

from .api import evaluate
from .registry import REGISTRY, MetricRegistry
# inside metrics/__init__.py
from . import calibration, distribution, diversity, sanity, utility, privacy  # noqa


# Side-effect imports that register default metrics
from . import preprocess as _preprocess  # noqa: F401
from . import contracts as _contracts    # noqa: F401
from . import features as _features      # noqa: F401

from .registry import REGISTRY as registry  # alias

__all__ = ["evaluate",
           "REGISTRY",
           "MetricRegistry",
           "registry"]