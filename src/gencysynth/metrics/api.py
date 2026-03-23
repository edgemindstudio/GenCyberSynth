# src/gencysynth/metrics/api.py
"""
Public API for running a metrics suite.

This is what orchestrators should call.

Rule A behavior
---------------
- Normalizes dataset_id + run_id
- Writes one directory per (dataset_id, run_id)
- Writes per_metric JSON + summary.json + optional events.jsonl

This module does not assume a specific generator family. It only needs:
- real arrays
- synthetic arrays
- optional labels
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config import artifacts_root, enabled_metrics, metric_options, normalize_dataset_id, normalize_run_id
from .contracts import ShapeSpec
from .preprocess import PreprocessConfig, preprocess_for_metrics
from .registry import REGISTRY
from .types import DatasetMeta, RunMeta, MetricResult
from .writer import append_event, resolve_metrics_paths, write_metric_result, write_summary


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def evaluate(
    *,
    cfg: Dict[str, Any],
    x_real: np.ndarray,
    y_real: Optional[np.ndarray],
    x_synth: np.ndarray,
    y_synth: Optional[np.ndarray],
    defaults_enabled: List[str],
) -> Dict[str, Any]:
    """
    Run an enabled set of metrics and write artifacts.

    Returns
    -------
    summary dict (also written to summary.json).
    """
    # ---- Normalize global identities ----
    dataset_id = normalize_dataset_id(cfg)
    run_id = normalize_run_id(cfg)
    seed = int(cfg.get("SEED", 42))

    # ---- Resolve shape expectations from config ----
    img_shape = tuple(cfg.get("IMG_SHAPE", (40, 40, 1)))
    num_classes = int(cfg.get("NUM_CLASSES", cfg.get("num_classes", 9)))
    spec = ShapeSpec(img_shape=img_shape, num_classes=num_classes)

    # ---- Preprocess (validate + normalize to float01 + int labels) ----
    pp = PreprocessConfig(
        binarize=bool(cfg.get("metrics", {}).get("binarize", False)),
        bin_threshold=float(cfg.get("metrics", {}).get("bin_threshold", 0.5)),
    )

    x_real01, y_real_i = preprocess_for_metrics(x=x_real, y=y_real, spec=spec, pp=pp, name="real")
    x_syn01, y_syn_i = preprocess_for_metrics(x=x_synth, y=y_synth, spec=spec, pp=pp, name="synth")

    # ---- Prepare metadata ----
    ds = DatasetMeta(dataset_id=dataset_id, img_shape=img_shape, num_classes=num_classes)
    run = RunMeta(run_id=run_id, seed=seed, tags=dict(cfg.get("run", {}).get("tags", {})))

    # ---- Output paths ----
    arts_root = artifacts_root(cfg)
    paths = resolve_metrics_paths(artifacts_root=arts_root, dataset_id=dataset_id, run_id=run_id)

    # ---- Determine which metrics to run ----
    metric_names = enabled_metrics(cfg, defaults=defaults_enabled)

    # ---- Run suite ----
    results: List[MetricResult] = []
    append_event(paths, {"ts": _now_iso(), "event": "metrics_start", "dataset_id": dataset_id, "run_id": run_id})

    for name in metric_names:
        metric = REGISTRY.get(name)
        try:
            append_event(paths, {"ts": _now_iso(), "event": "metric_start", "name": name})
            res = metric(
                x_real01=x_real01,
                y_real=y_real_i,
                x_synth01=x_syn01,
                y_synth=y_syn_i,
                dataset=ds,
                run=run,
                cfg=cfg,
            )
            res.name = name  # enforce registry key
            out_path = write_metric_result(paths, res)
            res.artifacts.setdefault("result_json", str(out_path))
            results.append(res)
            append_event(paths, {"ts": _now_iso(), "event": "metric_done", "name": name, "status": res.status})
        except Exception as e:
            err = MetricResult(name=name, status="error", error=str(e))
            out_path = write_metric_result(paths, err)
            err.artifacts["result_json"] = str(out_path)
            results.append(err)
            append_event(paths, {"ts": _now_iso(), "event": "metric_done", "name": name, "status": "error", "error": str(e)})

    append_event(paths, {"ts": _now_iso(), "event": "metrics_done", "dataset_id": dataset_id, "run_id": run_id})

    # ---- Summary ----
    summary = {
        "dataset_id": dataset_id,
        "run_id": run_id,
        "created_at": _now_iso(),
        "img_shape": list(img_shape),
        "num_classes": int(num_classes),
        "seed": int(seed),
        "metrics_enabled": metric_names,
        "results": [r.to_dict() for r in results],
        "artifacts": {
            "root": str(paths.root),
            "summary_json": str(paths.summary_json),
            "events_jsonl": str(paths.events_jsonl),
            "by_metric": str(paths.by_metric_dir),
        },
    }
    write_summary(paths, summary)
    return summary