# src/gencysynth/orchestration/context.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from gencysynth.models.base_types import RunContext
from gencysynth.utils.paths import (
    ensure_dir,
    resolve_eval_paths,
    resolve_logs_paths,
    resolve_run_paths,
)


def _cfg_get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def resolve_run_identity(cfg: Mapping[str, Any]) -> Tuple[str, str, str, int, str]:
    """
    Resolve the canonical run identity keys used everywhere in the repo:

        (dataset_id, model_tag, run_id, seed, artifacts_root)

    This is the *only* place where we fall back to placeholders if config
    is missing expected fields.
    """
    artifacts_root = str(_cfg_get(cfg, "paths.artifacts", "artifacts"))

    dataset_id = _cfg_get(cfg, "dataset.id", None) or _cfg_get(cfg, "run_meta.dataset_id", None)
    if not isinstance(dataset_id, str) or not dataset_id:
        dataset_id = "unknown_dataset"

    model_tag = _cfg_get(cfg, "model.tag", None) or _cfg_get(cfg, "run_meta.model_tag", None)
    if not isinstance(model_tag, str) or not model_tag:
        model_tag = "unknown_model"

    seed = int(cfg.get("SEED", cfg.get("seed", 0)))

    run_id = _cfg_get(cfg, "run_meta.run_id", None)
    if not isinstance(run_id, str) or not run_id:
        # Stable fallback; orchestrators should set run_meta.run_id explicitly
        run_id = f"run_seed{seed}"

    return str(dataset_id), str(model_tag), str(run_id), int(seed), artifacts_root


@dataclass(frozen=True)
class ResolvedContext:
    """
    Container returned by resolve_run_context():
      - ctx: the canonical RunContext
      - cfg: config with run_meta/path hints injected (non-destructive)
    """
    ctx: RunContext
    cfg: Dict[str, Any]


def resolve_run_context(cfg: Dict[str, Any], *, create_dirs: bool = True) -> ResolvedContext:
    """
    Build a RunContext and ensure directories exist (optional).

    This is used by orchestration/harvester.py and any CLI entrypoints.
    """
    dataset_id, model_tag, run_id, seed, artifacts_root = resolve_run_identity(cfg)

    run_paths = resolve_run_paths(
        artifacts_root=artifacts_root,
        dataset_id=dataset_id,
        model_tag=model_tag,
        run_id=run_id,
    )
    log_paths = resolve_logs_paths(
        artifacts_root=artifacts_root,
        dataset_id=dataset_id,
        model_tag=model_tag,
        run_id=run_id,
    )
    eval_paths = resolve_eval_paths(
        artifacts_root=artifacts_root,
        dataset_id=dataset_id,
        model_tag=model_tag,
        run_id=run_id,
    )

    if create_dirs:
        ensure_dir(run_paths.root_dir)
        ensure_dir(run_paths.samples_dir)
        ensure_dir(run_paths.checkpoints_dir)
        ensure_dir(log_paths.root_dir)
        ensure_dir(eval_paths.root_dir)

    # Canonical context object passed to models/eval/orchestration
    ctx = RunContext(
        dataset_id=dataset_id,
        model_tag=model_tag,
        run_id=run_id,
        seed=seed,
        artifacts_root=artifacts_root,
        run_dir=run_paths.root_dir,
        logs_dir=log_paths.root_dir,
        eval_dir=eval_paths.root_dir,
    )

    # Inject run_meta back into cfg (non-destructive)
    rm = cfg.get("run_meta") if isinstance(cfg.get("run_meta"), dict) else {}
    rm.setdefault("dataset_id", dataset_id)
    rm.setdefault("model_tag", model_tag)
    rm.setdefault("run_id", run_id)
    rm.setdefault("seed", seed)
    cfg["run_meta"] = rm

    cfg.setdefault("paths", {})
    if isinstance(cfg["paths"], dict):
        cfg["paths"].setdefault("artifacts", artifacts_root)

    return ResolvedContext(ctx=ctx, cfg=cfg)


__all__ = ["ResolvedContext", "resolve_run_identity", "resolve_run_context"]
