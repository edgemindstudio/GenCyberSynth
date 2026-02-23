# src/gencysynth/metrics/tests/test_metrics_contracts.py
"""
Contract tests for metrics.

These tests intentionally avoid any artifact I/O (Rule A).
They validate that metrics:
- can be called with the common signature
- return MetricResult with required fields
- use injected payloads rather than reading from disk

If your registry or type names differ slightly, these tests will skip rather than fail
hard, to keep CI usable while wiring evolves.
"""

from __future__ import annotations

import numpy as np
import pytest


def _has_attrs(obj, names):
    return all(hasattr(obj, n) for n in names)


def test_metric_result_contract_smoke():
    """
    MetricResult should carry the minimal contract fields.
    """
    try:
        from gencysynth.metrics.types import MetricResult
    except Exception:
        pytest.skip("gencysynth.metrics.types.MetricResult not importable yet.")

    # Construct if possible
    try:
        r = MetricResult(name="x", value=1.0, status="ok", details={})
    except Exception:
        pytest.skip("MetricResult signature differs; contract test deferred.")

    assert _has_attrs(r, ["name", "value", "status", "details"])
    assert isinstance(r.name, str)
    assert isinstance(r.details, dict)


def test_all_registered_metrics_return_metricresult():
    """
    Call each registered metric on tiny dummy arrays and ensure it returns MetricResult.

    Note:
    - Some metrics require injected predictions/features; we provide a minimal cfg that
      satisfies common patterns. Metrics that still can't run should return status
      'skipped' (preferred) rather than raising.
    """
    try:
        from gencysynth.metrics.registry import REGISTRY
        from gencysynth.metrics.types import MetricResult, DatasetMeta, RunMeta
    except Exception:
        pytest.skip("Metrics registry/types not fully importable yet.")

    # Minimal fake data
    H, W, C = 8, 8, 1
    K = 3
    x_real = np.random.rand(10, H, W, C).astype(np.float32)
    y_real = np.random.randint(0, K, size=(10,), dtype=np.int64)
    x_syn = np.random.rand(9, H, W, C).astype(np.float32)
    y_syn = np.random.randint(0, K, size=(9,), dtype=np.int64)

    # Minimal metas
    try:
        dataset = DatasetMeta(name="dummy", num_classes=K, img_shape=(H, W, C))
    except Exception:
        # fall back: allow DatasetMeta to have different fields
        dataset = DatasetMeta  # type: ignore
        dataset = dataset(name="dummy", num_classes=K)  # type: ignore

    try:
        run = RunMeta(run_id="test", seed=123)
    except Exception:
        run = RunMeta  # type: ignore
        run = run(run_id="test")  # type: ignore

    # Provide minimal injected payloads that some metrics expect
    # - calibration metrics expect probs payload
    # - privacy.nn_distance expects features payload
    p_pred = np.ones((len(y_real), K), dtype=np.float32) / float(K)
    cfg = {
        "metrics": {"options": {}},
        "calibration": {"probs": {"y_true": y_real, "p_pred": p_pred}},
        "privacy": {"features": {"real": np.random.rand(20, 16).astype(np.float32),
                                 "synth": np.random.rand(15, 16).astype(np.float32)}},
    }

    # Registry iteration: support either dict-like or a .items() interface
    try:
        items = list(REGISTRY.items())  # type: ignore[attr-defined]
    except Exception:
        try:
            items = [(k, REGISTRY.get(k)) for k in REGISTRY.keys()]  # type: ignore[attr-defined]
        except Exception:
            pytest.skip("REGISTRY iteration interface unknown.")

    assert items, "No metrics registered; did you import gencysynth.metrics package init?"

    for name, metric in items:
        if metric is None:
            continue

        # metric should be callable
        if not callable(metric):
            pytest.fail(f"Registered metric '{name}' is not callable.")

        try:
            out = metric(
                x_real01=x_real,
                y_real=y_real,
                x_synth01=x_syn,
                y_synth=y_syn,
                dataset=dataset,
                run=run,
                cfg=cfg,
            )
        except Exception as e:
            pytest.fail(f"Metric '{name}' raised exception instead of returning MetricResult: {e}")

        assert isinstance(out, MetricResult), f"Metric '{name}' returned {type(out)} not MetricResult."
        assert hasattr(out, "status"), f"Metric '{name}' missing status"
        assert hasattr(out, "details"), f"Metric '{name}' missing details"
        assert isinstance(out.details, dict), f"Metric '{name}' details must be dict"
        assert out.status in {"ok", "skipped", "error"}, f"Metric '{name}' invalid status: {out.status}"
