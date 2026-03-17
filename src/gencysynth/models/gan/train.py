# src/gencysynth/models/gan/train.py
"""
GenCyberSynth — GAN Family — Training Router (non_variant)
=========================================================

This module provides a stable training entrypoint for the GAN family and routes
to the selected variant trainer implementation:

    gencysynth.models.gan.variants.<variant>.train

Supported variant trainer contracts
-----------------------------------
We support either of these in the variant module:
  - main(argv: list[str] | None) -> int
  - run_from_file(cfg_path: Path, ...) -> int
  - train(cfg_or_argv) -> int   (adapter_style, optional)

We intentionally keep this router lightweight and variant_agnostic.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .base import build_identity


def train(cfg_or_argv: Mapping[str, Any] | Sequence[str]) -> int:
    """
    Adapter_friendly entrypoint:
      - if list/tuple argv -> calls main(argv)
      - if dict cfg -> expects cfg has "config_path" or a caller uses variant main another way

    In practice, your CLI may call variant train main() directly. This function is here
    so family_level routing is available when you say "--model gan" and configure variant.
    """
    if isinstance(cfg_or_argv, (list, tuple)):
        return main(list(cfg_or_argv))

    if isinstance(cfg_or_argv, Mapping):
        # If caller passes dict config, we try to locate the config file path.
        # If not found, we cannot reliably call run_from_file; variants may support train(cfg).
        cfg = dict(cfg_or_argv)
        ident = build_identity(cfg)
        mod_path = f"gencysynth.models.gan.variants.{ident.variant}.train"
        mod = import_module(mod_path)

        if hasattr(mod, "train") and callable(getattr(mod, "train")):
            print(f"[{ident.tag}] calling variant train(config_dict)")
            ret = mod.train(cfg)  # type: ignore[attr_defined]
            return int(ret) if isinstance(ret, int) else 0

        raise TypeError(
            f"[{ident.tag}] train(cfg) not supported by variant and no argv provided."
        )

    raise TypeError(f"Unsupported payload type: {type(cfg_or_argv)}")


def main(argv: Optional[list[str]] = None) -> int:
    """
    CLI_style router for training.

    If you call this module directly, you should pass through the variant trainer's
    CLI, e.g.:
      python -m gencysynth.models.gan.train --variant dcgan --config configs/config.yaml

    However, most users will call your unified CLI:
      gencs train --model gan --config configs/config.yaml
    """
    import argparse

    p = argparse.ArgumentParser(description="GAN family training router")
    p.add_argument("--variant", default=None, help="GAN variant (e.g., dcgan, wgan, wgangp). Overrides config.")
    p.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML config")
    args, rest = p.parse_known_args(argv)

    # Load config (keep dependency optional at top_level)
    try:
        import yaml  # type: ignore
    except Exception:
        raise SystemExit("PyYAML is required to read config files for routing.")

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        raise SystemExit(f"Config file not found: {cfg_path}")

    cfg: Dict[str, Any] = yaml.safe_load(cfg_path.read_text()) or {}
    if not isinstance(cfg, dict):
        raise SystemExit(f"Config file must contain a top_level mapping/dict: {cfg_path}")

    # Variant override by flag (highest priority)
    if args.variant:
        cfg.setdefault("model", {})
        if isinstance(cfg["model"], dict):
            cfg["model"]["family"] = "gan"
            cfg["model"]["variant"] = args.variant

    ident = build_identity(cfg)
    mod_path = f"gencysynth.models.gan.variants.{ident.variant}.train"

    print(f"[{ident.tag}] train router")
    print(f"[{ident.tag}] config={cfg_path}")

    mod = import_module(mod_path)

    # Preferred: variant exposes main(argv)
    if hasattr(mod, "main") and callable(getattr(mod, "main")):
        # Forward config path + remaining args to the variant trainer
        v_argv = ["--config", str(cfg_path)] + list(rest)
        print(f"[{ident.tag}] dispatch → {mod_path}.main({v_argv})")
        ret = mod.main(v_argv)  # type: ignore[attr_defined]
        return int(ret) if isinstance(ret, int) else 0

    # Fallback: variant exposes run_from_file(cfg_path, ...)
    if hasattr(mod, "run_from_file") and callable(getattr(mod, "run_from_file")):
        print(f"[{ident.tag}] dispatch → {mod_path}.run_from_file({cfg_path})")
        ret = mod.run_from_file(cfg_path)  # type: ignore[attr_defined]
        return int(ret) if isinstance(ret, int) else 0

    raise SystemExit(
        f"[{ident.tag}] Variant trainer does not expose main(argv) or run_from_file(cfg_path): {mod_path}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
