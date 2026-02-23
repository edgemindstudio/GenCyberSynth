# src/gencysynth/models/gan/base.py
"""
GenCyberSynth — GAN Family — Shared (non-variant) Helpers
=======================================================

This module is the *family-level* "glue" for GAN models.

Why this exists
---------------
Your repository supports:
  - multiple datasets
  - multiple GAN variants (dcgan, wgan, wgangp, stylegan2, ...)
  - a unified CLI/orchestrator that should not care about variant internals

So the GAN family provides shared helpers:
  - robust config access (nested dotted paths)
  - consistent path conventions (artifacts, synth output roots)
  - stable identity fields (family, variant, dataset_id)

What this file MUST NOT do
--------------------------
- It must not implement the DCGAN/WGAN training loops.
- It must not load datasets.
- It must not compute metrics.

Variants own model logic under:
    gencysynth.models.gan.variants.<variant>.(train|sample|model).py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

FAMILY: str = "gan"

# ---------------------------------------
# Config helpers
# ---------------------------------------
def cfg_get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    """
    Read nested configuration values using a dotted path, e.g.:
        cfg_get(cfg, "paths.artifacts", "artifacts")
        cfg_get(cfg, "data.dataset_id", "unknown_dataset")

    Returns default if the path is missing or intermediate keys are not mappings.
    """
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def ensure_dir(p: Path) -> Path:
    """Create a directory (and parents) if needed and return the same path."""
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------
# Dataset identity (scalable across datasets)
# ---------------------------------------
def resolve_dataset_id(cfg: Mapping[str, Any]) -> str:
    """
    Resolve a stable dataset identifier for manifests and output paths.

    Preferred:
      cfg["data"]["dataset_id"]

    Fallbacks (legacy / transitional):
      cfg["dataset_id"]
      cfg["data"]["name"]
      "unknown_dataset"

    Why dataset_id matters
    ----------------------
    We want synthesis outputs to be grouped by dataset in artifacts so that
    the same model/variant can be run on multiple datasets without collisions.
    """
    dsid = cfg_get(cfg, "data.dataset_id", None)
    if isinstance(dsid, str) and dsid.strip():
        return dsid.strip()

    dsid = cfg_get(cfg, "dataset_id", None)
    if isinstance(dsid, str) and dsid.strip():
        return dsid.strip()

    dsname = cfg_get(cfg, "data.name", None)
    if isinstance(dsname, str) and dsname.strip():
        return dsname.strip()

    return "unknown_dataset"


def resolve_dataset_root(cfg: Mapping[str, Any]) -> Optional[str]:
    """
    Optional: resolve a dataset root path for provenance only.

    Preferred:
      cfg["data"]["root"]

    Legacy fallback:
      cfg["DATA_DIR"]

    NOTE:
    - This value is recorded into manifests as informational provenance.
    - It should not be relied upon as a stable identifier across machines.
    """
    root = cfg_get(cfg, "data.root", None)
    if isinstance(root, str) and root.strip():
        return root.strip()

    data_dir = cfg.get("DATA_DIR")
    if isinstance(data_dir, str) and data_dir.strip():
        return data_dir.strip()

    return None


# ---------------------------------------
# Variant resolution
# ---------------------------------------
def resolve_variant(cfg: Mapping[str, Any], *, default: str = "dcgan") -> str:
    """
    Resolve which GAN variant to run.

    Preferred:
      cfg["model"]["variant"]   (where cfg["model"]["family"] == "gan")

    Also supported:
      cfg["gan"]["variant"]

    If missing, we default to `dcgan` so the family can run out-of-the-box.

    IMPORTANT:
      - Orchestrator can also specify explicit adapter ids like "gan/dcgan"
        (that routing is handled at adapter registry level).
      - This function is for *family-level* model routing when invoked as "gan".
    """
    v = cfg_get(cfg, "model.variant", None)
    if isinstance(v, str) and v.strip():
        return v.strip()

    v = cfg_get(cfg, "gan.variant", None)
    if isinstance(v, str) and v.strip():
        return v.strip()

    return default


# ---------------------------------------
# Artifact path conventions (single source of truth)
# ---------------------------------------
def resolve_artifacts_root(cfg: Mapping[str, Any]) -> Path:
    """
    Resolve the repo artifacts root.

    Priority:
      1) cfg["paths"]["artifacts"]
      2) "artifacts" (default)

    The caller may override cfg["paths"]["artifacts"] in CLI for experiments.

    NOTE:
      We return a Path but do not create it automatically here.
    """
    return Path(cfg_get(cfg, "paths.artifacts", "artifacts"))


def resolve_gan_checkpoints_dir(cfg: Mapping[str, Any]) -> Path:
    """
    Resolve where GAN checkpoints are stored.

    Preferred override:
      cfg["paths"]["gan_checkpoints"]

    Default:
      {paths.artifacts}/gan/checkpoints

    This is the canonical location your DCGAN sampler already expects.
    """
    artifacts_root = resolve_artifacts_root(cfg)
    return Path(cfg_get(cfg, "paths.gan_checkpoints", artifacts_root / FAMILY / "checkpoints"))


def resolve_gan_family_root(cfg: Mapping[str, Any]) -> Path:
    """
    Resolve family root directory under artifacts:
      {paths.artifacts}/gan/
    """
    return resolve_artifacts_root(cfg) / FAMILY


def resolve_synth_root(cfg: Mapping[str, Any]) -> Path:
    """
    Resolve the dataset-aware synthesis root directory:

      {artifacts}/synth/{dataset_id}

    This is intentionally family-agnostic so all families share a consistent
    "synth/<dataset_id>/" layout.

    Variants then write under:
      {synth_root}/gan/<variant>/...

    This prevents collisions across datasets and makes audits easy.
    """
    artifacts_root = resolve_artifacts_root(cfg)
    dataset_id = resolve_dataset_id(cfg)
    return artifacts_root / "synth" / dataset_id


@dataclass(frozen=True)
class GanIdentity:
    """Stable identity block used in manifests and logs."""
    family: str
    variant: str
    dataset_id: str

    @property
    def tag(self) -> str:
        """Convenient label like 'gan/dcgan'."""
        return f"{self.family}/{self.variant}"


def build_identity(cfg: Mapping[str, Any], *, default_variant: str = "dcgan") -> GanIdentity:
    """Build (family, variant, dataset_id) from config."""
    return GanIdentity(
        family=FAMILY,
        variant=resolve_variant(cfg, default=default_variant),
        dataset_id=resolve_dataset_id(cfg),
    )
