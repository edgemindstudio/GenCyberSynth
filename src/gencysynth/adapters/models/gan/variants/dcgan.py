# src/gencysynth/adapters/models/gan/variants/dcgan.py
"""
Adapter: GAN / DCGAN (Conditional DCGAN)
=======================================

This adapter is the *bridge* between:
  (A) the repository_wide orchestration (CLI / evaluator runner)
and
  (B) the DCGAN implementation + sampler entrypoint.

Variant identity (must be explicit everywhere)
----------------------------------------------
- family  : gan
- variant : dcgan
- sampler : gan.sample.synth(cfg, output_root, seed) -> manifest dict

Scalable output layout (dataset_aware)
--------------------------------------
To support *multiple datasets* and *multiple variants* safely, this adapter writes
to a canonical, collision_proof directory:

  <ARTIFACTS_ROOT>/synth/<FAMILY>/<VARIANT>/<DATASET_ID>/seed_<SEED>/

Example:
  artifacts/synth/gan/dcgan/ustc_tfc2016_nhwc/seed_42/class_0/gan_00001.png
  artifacts/synth/gan/dcgan/ustc_tfc2016_nhwc/seed_42/manifest.json

Why this matters
----------------
- You can run multiple datasets without overwriting outputs.
- You can run multiple GAN variants (dcgan, wgan_gp, ...) side_by_side.
- The evaluator can always find the manifest in a consistent place.

Manifest contract (evaluator_facing)
------------------------------------
The evaluator expects a stable schema:

{
  "dataset": str,                # provenance label or path
  "dataset_id": str,             # stable dataset identifier used in paths
  "family": "gan",
  "variant": "dcgan",
  "seed": int,
  "created_at": str,
  "run_dir": str,                # directory containing outputs
  "paths": [{"path": str, "label": int}, ...],
  "per_class_counts": {"0": int, ..., "K_1": int},
  "num_fake": int,
  "budget_per_class": int | None
}

Important note about sampler output_root
----------------------------------------
The local sampler `gan.sample.synth(...)` already writes into:

  output_root/<class>/<seed>/...

BUT our canonical structure is:

  run_dir/class_<k>/

So this adapter uses one of two strategies:

Strategy A (preferred): pass `run_dir` to the sampler and then NORMALIZE
  - We pass `run_dir` as output_root.
  - If the sampler writes `run_dir/<k>/<seed>/...`, we rewrite/relocate paths
    into `run_dir/class_<k>/...` and ensure manifest paths reflect that.

Strategy B (minimal_change): accept sampler layout, but still isolate runs
  - We pass `run_dir` as output_root.
  - We allow files to live under `run_dir/<k>/<seed>/...`
  - We still write the manifest to `run_dir/manifest.json`

This implementation chooses Strategy B for safety (no moving files),
but it *still* keeps variant + dataset separation via `run_dir`.

If later you want `class_<k>` layout strictly, we can add an optional
"relocate outputs" step (safe but more I/O).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .base import Adapter


# ---------------------------------------------------------------------
# Local config + filesystem helpers
# ---------------------------------------------------------------------
def _cfg_get(cfg: Dict[str, Any], dotted: str, default=None):
    """
    Fetch nested config values using dotted paths (e.g., 'paths.artifacts').

    This keeps adapters resilient to partial configs and prevents KeyError chains.
    """
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _ensure_dir(p: Path) -> Path:
    """Create directory (and parents) if needed and return the same Path."""
    p.mkdir(parents=True, exist_ok=True)
    return p


def _resolve_seed(cfg: Dict[str, Any]) -> int:
    """
    Resolve seed deterministically (project convention):
      1) cfg["SEED"] (preferred single_run seed)
      2) cfg["random_seeds"][0] (legacy multi_seed fallback)
      3) 42
    """
    if "SEED" in cfg:
        return int(cfg["SEED"])
    rs = _cfg_get(cfg, "random_seeds", [42])
    if isinstance(rs, list) and rs:
        return int(rs[0])
    return 42


def _resolve_dataset_id(cfg: Dict[str, Any]) -> str:
    """
    Resolve a stable dataset identifier for folder naming.

    Priority:
      1) dataset.id
      2) data.id
      3) dataset.name / data.name
      4) fallback: "default_dataset"

    Keep this:
      - stable across runs
      - filesystem_safe
      - short but descriptive
    """
    did = (
        _cfg_get(cfg, "dataset.id")
        or _cfg_get(cfg, "data.id")
        or _cfg_get(cfg, "dataset.name")
        or _cfg_get(cfg, "data.name")
        or "default_dataset"
    )
    did = str(did).strip()
    return did.replace(" ", "_").replace("/", "_")


def _resolve_dataset_provenance(cfg: Dict[str, Any]) -> str:
    """
    Resolve the human/provenance dataset string stored in manifest["dataset"].

    Priority:
      1) data.root (preferred)
      2) DATA_DIR
      3) fallback label
    """
    return str(_cfg_get(cfg, "data.root", cfg.get("DATA_DIR", "unknown_dataset")))


# ---------------------------------------------------------------------
# Manifest normalization (evaluator_facing contract)
# ---------------------------------------------------------------------
def _normalize_manifest(manifest: Dict[str, Any], *, num_classes: int) -> Dict[str, Any]:
    """
    Normalize sampler output into a stable evaluator contract.

    Handles common sampler variations:
      - "samples" vs "paths"
      - Path objects vs strings
      - missing per_class_counts
      - missing derived fields

    Guarantees:
      - paths: list[{"path": str, "label": int}]
      - per_class_counts: {"0":..., "K_1":...}
      - num_fake
      - budget_per_class
    """
    if not isinstance(manifest, dict):
        manifest = {}

    # Legacy key normalization
    if "paths" not in manifest and isinstance(manifest.get("samples"), list):
        manifest["paths"] = manifest["samples"]

    raw_paths = manifest.get("paths")
    if not isinstance(raw_paths, list):
        raw_paths = []

    norm_paths: List[Dict[str, Any]] = []
    for it in raw_paths:
        if not isinstance(it, dict):
            continue

        p = it.get("path")
        y = it.get("label")

        if isinstance(p, Path):
            p = str(p)
        if not isinstance(p, str) or not p:
            continue

        try:
            y_int = int(y)
        except Exception:
            continue

        norm_paths.append({"path": p, "label": y_int})

    manifest["paths"] = norm_paths

    # per_class_counts
    pcc_in = manifest.get("per_class_counts")
    pcc: Dict[str, int] = {}

    if isinstance(pcc_in, dict) and pcc_in:
        for k, v in pcc_in.items():
            try:
                kk = str(int(k))
                vv = int(v)
            except Exception:
                continue
            if 0 <= int(kk) < num_classes and vv >= 0:
                pcc[kk] = vv
    else:
        for it in manifest["paths"]:
            kk = str(int(it["label"]))
            pcc[kk] = pcc.get(kk, 0) + 1

    manifest["per_class_counts"] = {str(k): int(pcc.get(str(k), 0)) for k in range(num_classes)}

    # Derived totals
    if manifest["paths"]:
        manifest["num_fake"] = int(len(manifest["paths"]))
    else:
        manifest["num_fake"] = int(sum(manifest["per_class_counts"].values()))

    vals = [int(v) for v in manifest["per_class_counts"].values()]
    manifest["budget_per_class"] = (min(vals) if vals and min(vals) > 0 else None)

    return manifest


# ---------------------------------------------------------------------
# Adapter implementation (GAN/DCGAN)
# ---------------------------------------------------------------------
class DCGANAdapter(Adapter):
    """
    Adapter for GAN family, DCGAN variant.

    IMPORTANT:
    - Do NOT name this "GANAdapter" here, because this file is variant_specific.
      Keeping the class name variant_specific prevents confusion once you add WGAN_GP.
    """

    # If your registry routes adapters by class attribute, keep these explicit.
    # (If your base AdapterInfo exists, you can migrate to: info = AdapterInfo(...))
    name = "gan"
    family = "gan"
    variant = "dcgan"

    def synth(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run DCGAN synthesis and write a dataset+variant_scoped manifest.

        Output directory (canonical):
          <artifacts>/synth/gan/dcgan/<dataset_id>/seed_<seed>/

        Manifest path:
          <run_dir>/manifest.json
        """
        # ----------------------------
        # Resolve canonical run paths
        # ----------------------------
        artifacts_root = Path(_cfg_get(config, "paths.artifacts", "artifacts")).expanduser()

        dataset_id = _resolve_dataset_id(config)
        dataset = _resolve_dataset_provenance(config)
        seed = _resolve_seed(config)

        # Canonical run directory (variant + dataset isolated)
        run_dir = _ensure_dir(
            artifacts_root / "synth" / self.family / self.variant / dataset_id / f"seed_{seed}"
        )

        # Manifest is always written here (single source of truth)
        man_path = run_dir / "manifest.json"

        # Minimal knobs (used even for stub manifests)
        H, W, C = tuple(_cfg_get(config, "IMG_SHAPE", (40, 40, 1)))
        K = int(_cfg_get(config, "NUM_CLASSES", 9))

        # ----------------------------
        # Stub manifest (fallback_safe)
        # ----------------------------
        manifest: Dict[str, Any] = {
            "dataset": dataset,
            "dataset_id": dataset_id,
            "family": self.family,
            "variant": self.variant,
            "seed": int(seed),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "run_dir": str(run_dir),
            "per_class_counts": {str(k): 0 for k in range(K)},
            "paths": [],
        }

        # ----------------------------
        # Call the DCGAN sampler
        # ----------------------------
        try:
            # Keep the import local to avoid breaking CLI import if GAN deps are missing.
            from gan.sample import synth as gan_synth  # type: ignore

            # Reproducibility (sampler may also set seeds internally)
            np.random.seed(seed)
            try:
                import tensorflow as tf
                tf.random.set_seed(seed)
            except Exception:
                pass

            print(f"[gan/dcgan] dataset_id={dataset_id}  seed={seed}")
            print(f"[gan/dcgan] IMG_SHAPE={H,W,C}  NUM_CLASSES={K}")
            print(f"[gan/dcgan] output_root={run_dir}")

            # IMPORTANT:
            # We pass run_dir to isolate this run (dataset_id + seed + variant).
            # The sampler may create its own subfolders (e.g., <k>/<seed>/...).
            # That is OK, because outputs remain isolated inside run_dir.
            man = gan_synth(config, str(run_dir), seed=seed)

            if isinstance(man, dict):
                manifest = dict(man)

        except Exception as e:
            print(f"[gan/dcgan][ERROR] Sampling failed: {type(e).__name__}: {e}")
            print("[gan/dcgan] Writing a stub manifest so the pipeline can proceed.")

        # ----------------------------
        # Normalize + enrich manifest
        # ----------------------------
        manifest = _normalize_manifest(manifest, num_classes=K)

        # Ensure identity + provenance is always present (even if sampler omitted it)
        manifest.setdefault("dataset", dataset)
        manifest.setdefault("dataset_id", dataset_id)
        manifest.setdefault("family", self.family)
        manifest.setdefault("variant", self.variant)
        manifest.setdefault("seed", int(seed))
        manifest.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
        manifest.setdefault("run_dir", str(run_dir))

        # ----------------------------
        # Persist manifest (canonical)
        # ----------------------------
        with open(man_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"[gan/dcgan] Wrote manifest → {man_path}")
        return manifest
