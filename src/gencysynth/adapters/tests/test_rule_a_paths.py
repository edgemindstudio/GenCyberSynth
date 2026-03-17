# src/gencysynth/adapters/tests/test_rule_a_paths.py
"""
Rule A path tests (adapter_level).

These tests enforce ONE thing:
    If orchestration asks an adapter for "where do I read/write artifacts for this run?",
    the returned paths MUST be:
      1) under the artifacts root,
      2) encode dataset identity,
      3) encode model identity (family + variant),
      4) encode run identity (run_id),
      5) stable & deterministic for the same inputs.

Why this matters
----------------
We are scaling to:
  - many datasets (USTC now, more later),
  - many model families + variants,
  - many runs (smoke, sweeps, paper modules),
so "paths must never be ad_hoc".

This test intentionally does NOT enforce exact folder names (you may evolve them),
but it DOES enforce invariants that must always be true under Rule A.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pytest


def _pick_callable(mod, names: list[str]) -> Callable:
    """
    Find the first callable attribute in `mod` matching any name in `names`.
    This makes tests resilient if you rename helpers (without breaking semantics).
    """
    for n in names:
        fn = getattr(mod, n, None)
        if callable(fn):
            return fn
    raise AssertionError(
        "Could not find any expected callable in module "
        f"{getattr(mod, '__name__', mod)}. Tried: {names}"
    )


def _as_posix(p: Path) -> str:
    return Path(p).as_posix()


def _assert_under(root: Path, p: Path) -> None:
    root = root.resolve()
    p = p.resolve()
    assert str(p).startswith(str(root)), f"Path must be under artifacts root.\nroot={root}\npath={p}"


def _assert_contains_segment(p: Path, seg: str) -> None:
    parts = set(p.parts)
    assert seg in parts, f"Expected segment '{seg}' in path.\npath={p}\nparts={p.parts}"


@pytest.mark.parametrize(
    "dataset_id,family,variant,run_id",
    [
        ("ustc_tfc2016", "gan", "dcgan", "RUN_0001"),
        ("ustc_tfc2016", "vae", "c_vae", "RUN_0002"),
        ("toy_dataset", "restrictedboltzmann", "c_rbm_bernoulli", "RUN_smoke"),
    ],
)
def test_rule_a_run_root_and_subpaths(tmp_path: Path, dataset_id: str, family: str, variant: str, run_id: str):
    """
    End_to_end invariants for Rule A run paths.

    We do not hardcode the exact naming convention (you may choose):
      artifacts/<dataset>/runs/<run_id>/<family>/<variant>/...
    OR
      artifacts/runs/<dataset>/<run_id>/<family>/<variant>/...

    BUT we DO enforce:
      - path is inside artifacts root
      - includes dataset_id, run_id, family, variant somewhere in the hierarchy
      - derived subpaths (checkpoints/synthetic/events/etc.) remain under same run root
    """
    # Import run_io (canonical adapter path helpers)
    import gencysynth.adapters.run_io as run_io  # noqa: F401

    # Candidate APIs we accept (pick the first that exists)
    run_root_fn = _pick_callable(
        run_io,
        [
            "run_root",
            "get_run_root",
            "resolve_run_root",
            "rule_a_run_root",
        ],
    )

    # Subpath helpers are optional; if present we test them too
    ckpt_fn: Optional[Callable] = getattr(run_io, "checkpoints_dir", None)
    synth_fn: Optional[Callable] = getattr(run_io, "synthetic_dir", None)
    events_fn: Optional[Callable] = getattr(run_io, "events_path", None)
    manifest_fn: Optional[Callable] = getattr(run_io, "manifest_path", None)

    artifacts_root = tmp_path / "artifacts"

    # Build the canonical run root
    rr = Path(run_root_fn(
        artifacts_root=artifacts_root,
        dataset_id=dataset_id,
        family=family,
        variant=variant,
        run_id=run_id,
    ))

    # ---- Rule A invariants for run root ----
    _assert_under(artifacts_root, rr)
    _assert_contains_segment(rr, dataset_id)
    _assert_contains_segment(rr, run_id)
    _assert_contains_segment(rr, family)
    _assert_contains_segment(rr, variant)

    # Must be deterministic: same inputs => same output
    rr2 = Path(run_root_fn(
        artifacts_root=artifacts_root,
        dataset_id=dataset_id,
        family=family,
        variant=variant,
        run_id=run_id,
    ))
    assert rr == rr2, f"Rule A run root must be deterministic.\nrr ={rr}\nrr2={rr2}"

    # ---- Optional subpaths: if helper exists, ensure they stay within rr ----
    if callable(ckpt_fn):
        ck = Path(ckpt_fn(run_root=rr))
        _assert_under(rr, ck)

    if callable(synth_fn):
        sd = Path(synth_fn(run_root=rr))
        _assert_under(rr, sd)

    if callable(events_fn):
        ep = Path(events_fn(run_root=rr))
        _assert_under(rr, ep)

    if callable(manifest_fn):
        mp = Path(manifest_fn(run_root=rr))
        _assert_under(rr, mp)

    # Bonus: run root should NOT silently escape via ".."
    assert ".." not in _as_posix(rr), f"Run root contains parent traversal: {rr}"


def test_rule_a_paths_do_not_depend_on_cwd(tmp_path: Path, monkeypatch):
    """
    Rule A paths must never depend on current working directory.
    The caller must provide artifacts_root (or it must come from config),
    but in either case the produced paths should be fully resolved from inputs.
    """
    import os
    import gencysynth.adapters.run_io as run_io

    run_root_fn = _pick_callable(run_io, ["run_root", "get_run_root", "resolve_run_root", "rule_a_run_root"])

    artifacts_root = tmp_path / "artifacts"

    # Compute in one cwd
    monkeypatch.chdir(tmp_path)
    rr1 = Path(run_root_fn(
        artifacts_root=artifacts_root,
        dataset_id="ustc_tfc2016",
        family="gan",
        variant="dcgan",
        run_id="RUN_CWD_1",
    ))

    # Compute in a different cwd
    other = tmp_path / "other_place"
    other.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(other)
    rr2 = Path(run_root_fn(
        artifacts_root=artifacts_root,
        dataset_id="ustc_tfc2016",
        family="gan",
        variant="dcgan",
        run_id="RUN_CWD_1",
    ))

    assert rr1 == rr2, f"Paths must not depend on CWD.\nrr1={rr1}\nrr2={rr2}"
    _assert_under(artifacts_root, rr1)
