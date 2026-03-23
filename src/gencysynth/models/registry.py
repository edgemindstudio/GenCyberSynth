# src/gencysynth/models/registry.py
"""
GenCyberSynth — Model Registry (family/variant scalable)

This registry is the single source of truth that maps:
    model_tag -> model implementation builder

Example model_tag
-----------------
- "gan/dcgan"
- "gan/wgangp"
- "vae/cvae"
- "diffusion/ddpm"

Why the registry matters
------------------------
- CLI/orchestration can select a model purely by model_tag (from config).
- Adding new families/variants becomes a one_line registration.
- You avoid brittle if/else logic across training/sampling/eval.

How it integrates with your scalability rule
--------------------------------------------
Everything is keyed by (dataset_id, model_tag, run_id)

The model_tag here is part of that key. This registry ensures model_tag is:
- stable
- unique
- variant_aware

Typical usage
-------------
    from gencysynth.models.registry import make_model_from_config

    model = make_model_from_config(cfg)   # picks cfg["model"]["tag"]
    model.train(cfg, ctx)
    model.sample(cfg, ctx)

Config expectations
-------------------
model:
  tag: "gan/dcgan"          # REQUIRED (recommended)
  # optionally:
  family: "gan"
  variant: "dcgan"

Design goals
------------
- Dependency_light: only imports model code when needed (lazy import).
- Clear errors: unknown tags show what is registered.
"""

from __future__ import annotations

import importlib
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from gencysynth.models.base_types import GenModel, ModelSpec


# -----------------------------------------------------------------------------
# Internal registry state
# -----------------------------------------------------------------------------
_REGISTRY: Dict[str, ModelSpec] = {}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _cfg_get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _split_impl(impl: str) -> Tuple[str, str]:
    """
    Split "pkg.module:attr" into ("pkg.module", "attr").
    """
    if ":" not in impl:
        raise ValueError(
            f"Invalid impl path '{impl}'. Expected format 'pkg.module:callable_name'."
        )
    mod, attr = impl.split(":", 1)
    mod = mod.strip()
    attr = attr.strip()
    if not mod or not attr:
        raise ValueError(f"Invalid impl path '{impl}'.")
    return mod, attr


def _load_builder(impl: str) -> Callable[[Mapping[str, Any]], GenModel]:
    """
    Lazy import builder callable for a model spec.
    """
    mod_name, attr = _split_impl(impl)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr, None)
    if fn is None or not callable(fn):
        raise ImportError(f"Builder '{attr}' not found or not callable in module '{mod_name}'.")
    return fn  # type: ignore[return_value]


def _safe_slug(s: str, *, max_len: int = 120) -> str:
    """
    Model tags are used in paths like artifacts/.../<model_tag>/...
    We allow '/', '-', '_', '.' to preserve family/variant organization.
    """
    if not isinstance(s, str):
        return "unknown"
    s = s.strip().replace(" ", "_")
    out: List[str] = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", ".", "/"):
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out).strip("_")
    return slug[:max_len] if slug else "unknown"


# -----------------------------------------------------------------------------
# Public registry operations
# -----------------------------------------------------------------------------
def register_model(spec: ModelSpec, *, overwrite: bool = False) -> None:
    """
    Register a model implementation by model_tag.

    Parameters
    ----------
    spec:
        ModelSpec describing the model and how to build it.
    overwrite:
        If False (default), raises if tag already exists.
    """
    tag = _safe_slug(spec.model_tag)
    if tag in _REGISTRY and not overwrite:
        raise KeyError(f"Model tag already registered: '{tag}'")
    _REGISTRY[tag] = ModelSpec(
        model_tag=tag,
        family=spec.family,
        variant=spec.variant,
        impl=spec.impl,
        description=spec.description,
    )


def is_registered(model_tag: str) -> bool:
    return _safe_slug(model_tag) in _REGISTRY


def get_spec(model_tag: str) -> ModelSpec:
    tag = _safe_slug(model_tag)
    if tag not in _REGISTRY:
        raise KeyError(
            f"Unknown model_tag '{tag}'. "
            f"Known tags: {', '.join(sorted(_REGISTRY.keys())) or '(none)'}"
        )
    return _REGISTRY[tag]


def known_model_tags() -> Dict[str, str]:
    """
    Return mapping: model_tag -> impl for quick printing (CLI).
    """
    return {k: v.impl for k, v in sorted(_REGISTRY.items(), key=lambda kv: kv[0])}


def list_specs() -> List[Dict[str, Any]]:
    """
    Return registry specs as dicts (useful for debugging/printing).
    """
    return [asdict(v) for _, v in sorted(_REGISTRY.items(), key=lambda kv: kv[0])]


def build_model(model_tag: str, cfg: Mapping[str, Any]) -> GenModel:
    """
    Build a model instance by model_tag using its registered builder.
    """
    spec = get_spec(model_tag)
    builder = _load_builder(spec.impl)
    m = builder(cfg)
    # Optional sanity: enforce a stable model_tag on the instance.
    try:
        if getattr(m, "model_tag", None) in (None, "", "unknown"):
            setattr(m, "model_tag", spec.model_tag)
    except Exception:
        pass
    return m


def make_model_from_config(cfg: Mapping[str, Any]) -> GenModel:
    """
    Build a model from config.

    Priority:
      1) cfg["model"]["tag"]  (recommended)
      2) cfg["run_meta"]["model_tag"] (if orchestrator injected)
      3) fallback: family/variant -> f"{family}/{variant}"

    Raises with a clear message if the resolved model_tag is not registered.
    """
    tag = _cfg_get(cfg, "model.tag", None)
    if not isinstance(tag, str) or not tag:
        tag = _cfg_get(cfg, "run_meta.model_tag", None)

    if not isinstance(tag, str) or not tag:
        fam = _cfg_get(cfg, "model.family", "unknown")
        var = _cfg_get(cfg, "model.variant", "unknown")
        tag = f"{fam}/{var}"

    tag = _safe_slug(tag)

    if not is_registered(tag):
        raise KeyError(
            f"Model tag '{tag}' is not registered.\n"
            f"Registered tags: {', '.join(sorted(_REGISTRY.keys())) or '(none)'}\n"
            f"Fix by:\n"
            f"  - setting model.tag in config, and/or\n"
            f"  - registering the model in gencysynth.models.registry\n"
        )

    return build_model(tag, cfg)


# -----------------------------------------------------------------------------
# Optional: eager registration helper
# -----------------------------------------------------------------------------
def register_builtin_models() -> None:
    """
    Register builtin model variants shipped in this repo.

    NOTE:
    - Keep this list minimal and stable.
    - Each variant should provide a builder function with signature:
          build(cfg) -> GenModel
    - If a variant is missing in the repo, leave it out (no import_time failure).
    """
    # GAN variants
    try:
        register_model(ModelSpec(
            model_tag="gan/dcgan",
            family="gan",
            variant="dcgan",
            impl="gencysynth.models.gan.variants.dcgan:build",
            description="Deep Convolutional GAN (baseline).",
        ), overwrite=False)
    except Exception:
        pass

    try:
        register_model(ModelSpec(
            model_tag="gan/wgan",
            family="gan",
            variant="wgan",
            impl="gencysynth.models.gan.variants.wgan:build",
            description="Wasserstein GAN.",
        ), overwrite=False)
    except Exception:
        pass

    try:
        register_model(ModelSpec(
            model_tag="gan/wgangp",
            family="gan",
            variant="wgangp",
            impl="gencysynth.models.gan.variants.wgangp:build",
            description="WGAN with Gradient Penalty.",
        ), overwrite=False)
    except Exception:
        pass

    # VAE variants (example placeholders; register when builders exist)
    try:
        register_model(ModelSpec(
            model_tag="vae/vae",
            family="vae",
            variant="vae",
            impl="gencysynth.models.vae.variants.vae:build",
            description="Vanilla VAE.",
        ), overwrite=False)
    except Exception:
        pass

def build_model_by_tag(model_tag: str, cfg: Mapping[str, Any]) -> GenModel:
    """
    Convenience wrapper: build directly by tag (used by orchestration).
    """
    return build_model(model_tag, cfg)


__all__ = [
    "register_model",
    "register_builtin_models",
    "is_registered",
    "get_spec",
    "known_model_tags",
    "list_specs",
    "build_model",
    "build_model_by_tag",
    "make_model_from_config",
]
