# src/gencysynth/orchestration/harvester.py
"""
GenCyberSynth — Harvester (one_run coordinator)
==============================================

Coordinates *one* run keyed by the scalability identity:

    (dataset_id, model_tag, run_id)

So all outputs are collision_free:
    artifacts/runs/<dataset_id>/<model_tag>/<run_id>/...
    artifacts/logs/<dataset_id>/<model_tag>/<run_id>/...
    artifacts/eval/<dataset_id>/<model_tag>/<run_id>/...

This module:
  1) Resolves a RunContext (dataset_id/model_tag/run_id/seed + output dirs)
  2) Builds a model from the registry
  3) Runs train (optional) + sample (required)
  4) Captures provenance + run_meta snapshot
  5) Optionally triggers evaluation (best_effort)

Model contract (from gencysynth.models.base_types)
-------------------------------------------------
GenModel:
  - train(cfg, ctx)  -> TrainResult
  - sample(cfg, ctx) -> SampleResult

Important policy
----------------
- The model is responsible for writing its manifest (recommended location):
      ctx.run_dir / "manifest.json"
- The harvester will inject:
      cfg["run_meta"]["manifest_path"] = <absolute path>
  so eval never has to guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from gencysynth.models.base_types import SampleResult, TrainResult
from gencysynth.models.registry import make_model_from_config
from gencysynth.orchestration.context import resolve_run_context
from gencysynth.orchestration.logger import get_run_logger, log_event_jsonl
from gencysynth.orchestration.provenance import build_provenance, write_provenance
from gencysynth.utils.io import write_json
from gencysynth.utils.paths import resolve_run_paths
from gencysynth.utils.reproducibility import now_iso


# -----------------------------------------------------------------------------
# Result record for callers (CLI, Slurm submitters, sweep tools)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class HarvestResult:
    dataset_id: str
    model_tag: str
    run_id: str
    seed: int

    run_dir: Path
    logs_dir: Path
    manifest_path: Path

    train_ok: Optional[bool] = None
    sample_ok: Optional[bool] = None
    eval_latest_path: Optional[Path] = None


# -----------------------------------------------------------------------------
# Small config helper
# -----------------------------------------------------------------------------
def _cfg_get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _resolve_manifest_path_from_sample(
    sample_res: SampleResult,
    *,
    default_manifest_path: Path,
) -> Path:
    """
    Prefer SampleResult.manifest_path if the model returned it.
    Otherwise fall back to the canonical run location.

    If the model returns a *relative* manifest path, interpret it relative to the
    run directory (i.e., sibling of default_manifest_path).
    """
    mp = sample_res.manifest_path
    if isinstance(mp, str) and mp.strip():
        p = Path(mp)
        if not p.is_absolute():
            # interpret relative path under run_dir
            return (default_manifest_path.parent / p).resolve()
        return p
    return default_manifest_path


# -----------------------------------------------------------------------------
# Main API
# -----------------------------------------------------------------------------
def run_one(cfg: Dict[str, Any], *, do_eval: bool = True) -> HarvestResult:
    """
    Execute one run end_to_end.

    Expected cfg fields (recommended)
    --------------------------------
    dataset:
      id: "<dataset_id>"

    model:
      tag: "<model_tag>"

    run_meta (optional; orchestrator may inject):
      run_id: "<run_id>"
      config_path: ".../config.yaml"
      overrides_path: ".../overrides.yaml"
    """
    # 1) Resolve RunContext + ensure output dirs
    resolved = resolve_run_context(cfg, create_dirs=True)
    ctx, cfg2 = resolved.ctx, resolved.cfg

    # Canonical run paths (run_dir/samples_dir/manifest_path etc.)
    run_paths = resolve_run_paths(
        artifacts_root=ctx.artifacts_root,
        dataset_id=ctx.dataset_id,
        model_tag=ctx.model_tag,
        run_id=ctx.run_id,
    )

    # 2) Configure per_run logger
    logger = get_run_logger(name="gencysynth.run", log_dir=Path(ctx.logs_dir))
    logger.info(
        "Run start: dataset_id=%s model_tag=%s run_id=%s seed=%s",
        ctx.dataset_id, ctx.model_tag, ctx.run_id, ctx.seed
    )

    log_event_jsonl(
        log_dir=Path(ctx.logs_dir),
        event={
            "ts": now_iso(),
            "event": "run_start",
            "dataset_id": ctx.dataset_id,
            "model_tag": ctx.model_tag,
            "run_id": ctx.run_id,
            "seed": ctx.seed,
        },
    )

    # 3) Provenance snapshot (always)
    prov = build_provenance(
        dataset_id=ctx.dataset_id,
        model_tag=ctx.model_tag,
        run_id=ctx.run_id,
        seed=ctx.seed,
        config_path=_cfg_get(cfg2, "run_meta.config_path", None),
        overrides_path=_cfg_get(cfg2, "run_meta.overrides_path", None),
        repo_root=Path("."),
        notes=_cfg_get(cfg2, "run_meta.notes", None),
    )
    write_provenance(run_paths.root_dir, prov)

    # 4) Build model from registry
    model = make_model_from_config(cfg2)

    # 5) Train (optional)
    skip_train = bool(_cfg_get(cfg2, "run.skip_train", False))
    train_ok: Optional[bool] = None

    if not skip_train:
        log_event_jsonl(
            log_dir=Path(ctx.logs_dir),
            event={"ts": now_iso(), "event": "train_start", "run_id": ctx.run_id},
        )

        tr: TrainResult = model.train(cfg2, ctx)
        train_ok = bool(tr.ok)

        log_event_jsonl(
            log_dir=Path(ctx.logs_dir),
            event={
                "ts": now_iso(),
                "event": "train_done",
                "run_id": ctx.run_id,
                "ok": train_ok,
                "message": tr.message,
                "best_ckpt_path": tr.best_ckpt_path,
            },
        )

        if not tr.ok:
            logger.warning("Training reported ok=False: %s", tr.message)
    else:
        logger.info("Skipping training (cfg.run.skip_train=True).")

    # 6) Sample (required)
    log_event_jsonl(
        log_dir=Path(ctx.logs_dir),
        event={"ts": now_iso(), "event": "sample_start", "run_id": ctx.run_id},
    )

    sr: SampleResult = model.sample(cfg2, ctx)
    sample_ok = bool(sr.ok)

    # Prefer model_returned manifest path; else canonical
    manifest_path = _resolve_manifest_path_from_sample(
        sr,
        default_manifest_path=run_paths.manifest_path,
    )

    log_event_jsonl(
        log_dir=Path(ctx.logs_dir),
        event={
            "ts": now_iso(),
            "event": "sample_done",
            "run_id": ctx.run_id,
            "ok": sample_ok,
            "message": sr.message,
            "num_generated": sr.num_generated,
            "manifest_path": str(manifest_path),
        },
    )

    if not sample_ok:
        logger.warning("Sampling reported ok=False: %s", sr.message)

    # 7) Snapshot run_meta + results (auditability)
    snap = {
        "timestamp": now_iso(),
        "dataset_id": ctx.dataset_id,
        "model_tag": ctx.model_tag,
        "run_id": ctx.run_id,
        "seed": ctx.seed,
        "train_ok": train_ok,
        "sample_ok": sample_ok,
        "sample_result": {
            "ok": sr.ok,
            "message": sr.message,
            "num_generated": sr.num_generated,
            "manifest_path": str(manifest_path),
            "extra": sr.extra,
        },
        "run_meta": cfg2.get("run_meta", {}),
    }
    # write_json(run_paths.run_meta_snapshot_path, snap, indent=2, sort_keys=True, atomic=True)
    write_json(run_paths.run_meta_path, snap, indent=2, sort_keys=True, atomic=True)

    # Critical: inject manifest_path so eval runner can always locate it
    rm = cfg2.get("run_meta") if isinstance(cfg2.get("run_meta"), dict) else {}
    rm["manifest_path"] = str(manifest_path)
    rm.setdefault("dataset_id", ctx.dataset_id)
    rm.setdefault("model_tag", ctx.model_tag)
    rm.setdefault("run_id", ctx.run_id)
    rm.setdefault("seed", ctx.seed)
    cfg2["run_meta"] = rm

    # 8) Optional evaluation (best_effort)
    eval_latest: Optional[Path] = None
    if do_eval:
        try:
            from gencysynth.eval.runner import evaluate_model_suite

            evaluate_model_suite(cfg2, model_name=ctx.model_tag, no_synth=False)

            # The runner writes latest.json under the dataset_scalable eval dir.
            eval_latest = Path(ctx.eval_dir) / "latest.json"

            log_event_jsonl(
                log_dir=Path(ctx.logs_dir),
                event={"ts": now_iso(), "event": "eval_done", "run_id": ctx.run_id, "latest": str(eval_latest)},
            )
        except Exception as e:
            logger.warning("Evaluation failed (continuing): %s: %s", type(e).__name__, e)
            log_event_jsonl(
                log_dir=Path(ctx.logs_dir),
                event={"ts": now_iso(), "event": "eval_failed", "run_id": ctx.run_id, "error": f"{type(e).__name__}: {e}"},
            )

    log_event_jsonl(
        log_dir=Path(ctx.logs_dir),
        event={"ts": now_iso(), "event": "run_done", "run_id": ctx.run_id, "manifest_path": str(manifest_path)},
    )
    logger.info("Run done: manifest=%s", manifest_path)

    return HarvestResult(
        dataset_id=ctx.dataset_id,
        model_tag=ctx.model_tag,
        run_id=ctx.run_id,
        seed=ctx.seed,
        run_dir=run_paths.root_dir,
        logs_dir=Path(ctx.logs_dir),
        manifest_path=manifest_path,
        train_ok=train_ok,
        sample_ok=sample_ok,
        eval_latest_path=eval_latest,
    )


__all__ = ["HarvestResult", "run_one"]
