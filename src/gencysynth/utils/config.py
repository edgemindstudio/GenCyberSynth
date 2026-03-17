# src/gencysynth/utils/config.py
"""
GenCyberSynth — Config utilities (Rule A oriented)

Purpose
-------
Provide a small, predictable config layer used by all model variants so that:
- configs can be loaded from YAML
- defaults can be applied cleanly
- dotted access works consistently
- 'paths.*' normalization is standard
- multi_dataset scaling is supported (data.root + optional data.dataset_id)

This is intentionally lightweight: no Hydra/OmegaConf dependency.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from gencysynth.utils.paths import find_repo_root, ensure_dir


# -----------------------------
# YAML I/O
# -----------------------------
def load_yaml(path: Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf_8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(obj: Dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf_8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


# -----------------------------
# Dotted get/set
# -----------------------------
def cfg_get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def cfg_set(cfg: Dict[str, Any], dotted: str, value: Any) -> Dict[str, Any]:
    cur: Any = cfg
    keys = dotted.split(".")
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value
    return cfg


# -----------------------------
# Merge defaults
# -----------------------------
def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep_merge dictionaries:
    - dict values are merged recursively
    - non_dict values are overridden
    """
    out = deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)  # type: ignore[arg_type]
        else:
            out[k] = v
    return out


def load_with_defaults(
    *,
    config_path: Path,
    defaults_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Load a config YAML and optionally merge a defaults YAML underneath it.
    Defaults are applied first; config overrides them.
    """
    cfg = load_yaml(config_path)
    if defaults_path is None:
        return cfg

    defaults = load_yaml(defaults_path)
    return deep_merge(defaults, cfg)


# -----------------------------
# Rule A path normalization
# -----------------------------
def normalize_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize key paths used everywhere.

    Expected keys:
      paths:
        artifacts: artifacts
        runs: artifacts/runs     (optional)
      data:
        root: <dataset_root>     (e.g. USTC_TFC2016_malware)
        dataset_id: <optional>   (stable slug/id for multi_dataset runs)

    Behavior:
    - ensure paths.artifacts exists (default: <repo_root>/artifacts)
    - keep data.root as_is (relative roots are treated as repo_relative)
    """
    repo = find_repo_root()
    cfg = dict(cfg)

    # paths.artifacts
    arts = cfg_get(cfg, "paths.artifacts", None)
    if arts is None:
        arts = str(repo / "artifacts")
        cfg_set(cfg, "paths.artifacts", arts)

    # Optional paths.runs (nice convention; not mandatory)
    runs = cfg_get(cfg, "paths.runs", None)
    if runs is None:
        cfg_set(cfg, "paths.runs", str(Path(arts) / "runs"))

    return cfg


def resolve_repo_relative_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert known path_like fields into absolute paths for runtime reliability.
    (Still keeps original values in cfg; callers can choose what to write out.)

    Note: we intentionally only resolve a minimal set of fields.
    """
    repo = find_repo_root()
    cfg = dict(cfg)

    arts = Path(cfg_get(cfg, "paths.artifacts", "artifacts"))
    if not arts.is_absolute():
        arts = repo / arts
    cfg_set(cfg, "paths.artifacts_abs", str(arts))

    data_root = Path(cfg_get(cfg, "data.root", cfg_get(cfg, "DATA_DIR", "data")))
    if not data_root.is_absolute():
        data_root = repo / data_root
    cfg_set(cfg, "data.root_abs", str(data_root))

    return cfg


def ensure_artifact_dirs(cfg: Dict[str, Any], *relative_dirs: str) -> None:
    """
    Ensure artifact subdirectories exist under paths.artifacts.

    Example:
      ensure_artifact_dirs(cfg, "vae/checkpoints", "vae/summaries")
    """
    repo = find_repo_root()
    arts = Path(cfg_get(cfg, "paths.artifacts", "artifacts"))
    if not arts.is_absolute():
        arts = repo / arts
    ensure_dir(arts)

    for rel in relative_dirs:
        ensure_dir(arts / rel)
