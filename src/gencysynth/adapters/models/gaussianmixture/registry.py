# src/gencysynth/adapters/models/gaussianmixture/registry.py

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Callable, Optional

from gencysynth.adapters.models.base import ModelAdapter
from gencysynth.adapters.models.registry import register_model_adapter
from gencysynth.adapters.registry import SKIPPED_IMPORTS
from gencysynth.adapters.models.gaussianmixture.stub import GaussianMixtureStubAdapter


def register_gaussianmixture_variant(variant: str, *, factory: Callable[[], ModelAdapter]) -> None:
    register_model_adapter(family="gaussianmixture", variant=variant, factory=factory)


def _variants_dir() -> Path:
    # This file is: src/gencysynth/adapters/models/gaussianmixture/registry.py
    # Variants live next to it:
    #   src/gencysynth/adapters/models/gaussianmixture/variants/<variant>/
    return Path(__file__).resolve().parent / "variants"


def _load_variant_module(variant_dir: Path) -> Optional[object]:
    """
    Load variants/<variant>/adapter.py by file path (import-safe).

    Returns the loaded module object, or None if adapter.py missing/unloadable.
    """
    ap = variant_dir / "adapter.py"
    if not ap.exists():
        return None

    mod_name = f"gencysynth.adapters.models.gaussianmixture.variants.{variant_dir.name}.adapter"
    spec = spec_from_file_location(mod_name, ap)
    if spec is None or spec.loader is None:
        return None

    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def register_all_gaussianmixture_variants() -> None:
    vdir = _variants_dir()
    if not vdir.exists():
        SKIPPED_IMPORTS["gencysynth.adapters.models.gaussianmixture.registry"] = f"variants dir missing: {vdir}"
        return

    for p in sorted([x for x in vdir.iterdir() if x.is_dir() and not x.name.startswith("_")]):
        try:
            mod = _load_variant_module(p)

            # Preferred convention (simple & explicit):
            # - if adapter.py defines ADAPTER_FACTORY() -> ModelAdapter, use it
            if mod is not None and hasattr(mod, "ADAPTER_FACTORY"):
                factory_fn = getattr(mod, "ADAPTER_FACTORY")
                if callable(factory_fn):
                    register_gaussianmixture_variant(p.name, factory=lambda fn=factory_fn: fn())
                    continue

            # Special-case (your current plan): gmm_diag has GMMDiagAdapter
            if mod is not None and p.name == "gmm_diag" and hasattr(mod, "GMMDiagAdapter"):
                cls = getattr(mod, "GMMDiagAdapter")
                register_gaussianmixture_variant(p.name, factory=lambda C=cls: C())
                continue

            # Fallback: stub adapter for any variant without a real adapter
            register_gaussianmixture_variant(
                p.name,
                factory=lambda v=p.name: GaussianMixtureStubAdapter(variant=v),
            )

        except Exception as e:
            SKIPPED_IMPORTS[f"gencysynth.adapters.models.gaussianmixture.variants.{p.name}"] = f"{type(e).__name__}: {e}"
            # Still register stub so listing + sweeps don't break
            register_gaussianmixture_variant(
                p.name,
                factory=lambda v=p.name: GaussianMixtureStubAdapter(variant=v),
            )


try:
    register_all_gaussianmixture_variants()
except Exception as e:
    SKIPPED_IMPORTS["gencysynth.adapters.models.gaussianmixture.registry"] = f"{type(e).__name__}: {e}"
