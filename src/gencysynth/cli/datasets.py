# src/gencysynth/cli/datasets.py
"""
GenCyberSynth — Dataset CLI Commands

Why this exists
---------------
As the repo scales to:
- multiple datasets
- multiple dataset loaders/formats
- multiple papers/suites and HPC sweeps

...you need a fast way to sanity-check dataset configs and data loading without
running a full model pipeline.

Commands provided
-----------------
1) dataset-info
   - loads dataset implementation from registry via config["dataset"]["type"]
   - prints dataset identity + resolved input locations
   - optionally loads arrays for basic shape / count sanity

2) dataset-cache-warm
   - loads arrays once to warm any caching layer you implement
   - optionally writes dataset fingerprint to artifacts/datasets/<dataset_id>/fingerprint.json

Notes
-----
- This module intentionally avoids heavy dependencies.
- It uses the same dataset registry as training/eval pipelines.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from gencysynth.data.datasets.registry import make_dataset_from_config, known_dataset_types


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    obj = yaml.safe_load(p.read_text())
    return obj if isinstance(obj, dict) else {}


def _cfg_get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _pretty_kv(title: str, kv: Dict[str, Any]) -> str:
    lines = [title]
    for k, v in kv.items():
        lines.append(f"  - {k}: {v}")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# dataset-info
# -----------------------------------------------------------------------------
def _cmd_dataset_info(args: argparse.Namespace) -> int:
    cfg = _load_yaml(args.config)

    ds = make_dataset_from_config(cfg)

    dataset_id = _cfg_get(cfg, "dataset.id", "unknown_dataset")
    dataset_type = _cfg_get(cfg, "dataset.type", "unknown_type")
    raw_root = _cfg_get(cfg, "dataset.raw_root", None)
    artifacts_root = _cfg_get(cfg, "paths.artifacts", "artifacts")

    print(_pretty_kv("[dataset-info] identity", {
        "dataset_id": dataset_id,
        "dataset_type": dataset_type,
        "raw_root": raw_root,
        "artifacts_root": artifacts_root,
        "dataset_class": type(ds).__name__,
    }))

    # If dataset implements a light "describe" method, print it
    if hasattr(ds, "describe"):
        try:
            desc = ds.describe(cfg)  # type: ignore[misc]
            if isinstance(desc, dict):
                print(_pretty_kv("[dataset-info] describe()", desc))
        except Exception as e:
            print(f"[dataset-info] describe() failed: {type(e).__name__}: {e}")

    if args.load:
        print("[dataset-info] loading arrays (this may take time depending on dataset size)...")

        # Load arrays: support both signatures:
        #   - load_arrays(self)
        #   - load_arrays(self, cfg)
        try:
            splits = ds.load_arrays(cfg)  # newer style
        except TypeError:
            splits = ds.load_arrays()  # older style

        # Be flexible: support either dict {"train": (x,y), ...} or a typed object.
        try:
            if isinstance(splits, dict):
                for k, v in splits.items():
                    if isinstance(v, tuple) and len(v) == 2:
                        x, y = v
                        xshape = getattr(x, "shape", None)
                        yshape = getattr(y, "shape", None)
                        print(f"  - {k}: x={xshape}  y={yshape}")
                    else:
                        print(f"  - {k}: {type(v).__name__}")
            else:
                print(f"[dataset-info] splits type: {type(splits).__name__}")
        except Exception as e:
            print(f"[dataset-info] could not summarize splits: {type(e).__name__}: {e}")

    return 0


# -----------------------------------------------------------------------------
# dataset-cache-warm
# -----------------------------------------------------------------------------
def _cmd_dataset_cache_warm(args: argparse.Namespace) -> int:
    cfg = _load_yaml(args.config)
    ds = make_dataset_from_config(cfg)

    print("[dataset-cache-warm] loading arrays once to warm cache...")
    # Load arrays: support both signatures:
    #   - load_arrays(self)
    #   - load_arrays(self, cfg)
    try:
        _ = ds.load_arrays(cfg)
    except TypeError:
        _ = ds.load_arrays()
    print("[dataset-cache-warm] done loading.")

    if args.write_fingerprint:
        # Fingerprint is optional — don’t hard-crash if not implemented yet.
        try:
            from gencysynth.data.fingerprint import compute_dataset_fingerprint
            from gencysynth.data.fingerprint_writer import write_dataset_fingerprint
        except Exception as e:
            print(f"[dataset-cache-warm] fingerprint modules unavailable: {type(e).__name__}: {e}")
            return 0

        fp = compute_dataset_fingerprint(cfg, ds)  # type: ignore[arg-type]
        out_path = write_dataset_fingerprint(cfg, fp)
        print(f"[dataset-cache-warm] wrote fingerprint -> {out_path}")

    return 0


# -----------------------------------------------------------------------------
# Registration into main CLI
# -----------------------------------------------------------------------------
def register_dataset_subcommands(subparsers: argparse._SubParsersAction) -> None:
    """
    Hook these dataset commands into the top-level CLI.

    Your main CLI should call:
        register_dataset_subcommands(subparsers)
    """
    # dataset-info
    p_info = subparsers.add_parser(
        "dataset-info",
        help="Print dataset identity + resolved paths; optionally load arrays for sanity checks.",
    )
    p_info.add_argument("--config", required=True, help="Path to YAML config.")
    p_info.add_argument(
        "--load",
        action="store_true",
        help="Actually load arrays and print split shapes (slower but more reliable).",
    )
    p_info.set_defaults(func=_cmd_dataset_info)

    # dataset-cache-warm
    p_warm = subparsers.add_parser(
        "dataset-cache-warm",
        help="Load dataset once to warm cache; optionally write fingerprint.",
    )
    p_warm.add_argument("--config", required=True, help="Path to YAML config.")
    p_warm.add_argument(
        "--write-fingerprint",
        action="store_true",
        help="Write artifacts/datasets/<dataset_id>/fingerprint.json (requires fingerprint modules).",
    )
    p_warm.set_defaults(func=_cmd_dataset_cache_warm)

    # Optional: list known dataset types
    p_list = subparsers.add_parser(
        "dataset-types",
        help="List dataset.type tokens recognized by the registry.",
    )
    p_list.set_defaults(func=lambda _args: (print(_pretty_kv("[dataset-types]", known_dataset_types())) or 0))


__all__ = ["register_dataset_subcommands"]
