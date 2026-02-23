#!/usr/bin/env python3
"""
tools/validate/validate_artifacts.py

Validate Rule A artifacts:
- Required tree presence
- JSON schema validation for:
  - run_manifest.json
  - run_events.jsonl
  - eval/eval_summary.json
  - (optional) dataset_fingerprint.json / dataset_registry.json if present
- Optional deep checks:
  - synthetic/index.json points to existing files
  - x_synth.npy / y_synth.npy shapes (light sanity)

This tool is intentionally "thin": it does not assume how runs are created.
It only verifies the *canonical outputs* and keeps errors human-actionable.

Usage
-----
# Validate all runs under artifacts root:
python tools/validate/validate_artifacts.py --artifacts-root artifacts

# Validate one dataset:
python tools/validate/validate_artifacts.py --artifacts-root artifacts --dataset-id ustc-tfc2016-npy

# Validate one run id:
python tools/validate/validate_artifacts.py --artifacts-root artifacts --dataset-id ustc-tfc2016-npy --run-id <run_id>

# Validate a run_root directly:
python tools/validate/validate_artifacts.py --run-root artifacts/datasets/<dataset_id>/runs/<run_id>

# Enable deep synthetic checks:
python tools/validate/validate_artifacts.py --artifacts-root artifacts --check-synth-index --check-npy
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Optional dependency; we degrade gracefully if missing.
try:
    import jsonschema  # type: ignore
except Exception:  # pragma: no cover
    jsonschema = None  # type: ignore


# ---------------------------
# Rule A required files
# ---------------------------
REQUIRED_RELATIVE_PATHS: Tuple[str, ...] = (
    "run_manifest.json",
    "run_events.jsonl",
    "resolved_config.yaml",
    "synthetic/index.json",
    "eval/eval_summary.json",
)

# Default schema file names
SCHEMA_FILES = {
    "run_manifest": "run_manifest.schema.json",
    "run_events": "run_events.schema.json",
    "eval_summary": "eval_summary.schema.json",
    # present in repo; not always emitted as per-run artifacts
    "dataset_fingerprint": "dataset_fingerprint.schema.json",
    "dataset_registry": "dataset_registry.schema.json",
}


# ---------------------------
# Small utilities
# ---------------------------
def _print(msg: str) -> None:
    print(msg, flush=True)


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSONL at {path}:{i}: {e}") from e


def exists_all(run_root: Path, relpaths: Iterable[str]) -> List[str]:
    missing: List[str] = []
    for rel in relpaths:
        if not (run_root / rel).exists():
            missing.append(rel)
    return missing


def default_schemas_dir() -> Path:
    """
    Resolve schemas dir in a repo-friendly way:
      - prefer src/gencysynth/schemas (repo tree)
      - allow override via env SCHEMAS_DIR
    """
    env = os.environ.get("SCHEMAS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    # Assume script lives at tools/validate/validate_artifacts.py
    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / "src" / "gencysynth" / "schemas").resolve()


def validate_with_schema(instance: Any, schema: Dict[str, Any], *, where: str) -> Optional[str]:
    """
    Validate `instance` against `schema`. Returns error string or None.
    """
    if jsonschema is None:
        return f"[skip] jsonschema not installed (cannot validate {where})."
    try:
        jsonschema.validate(instance=instance, schema=schema)
        return None
    except Exception as e:
        return f"{where}: schema validation failed: {e}"


@dataclass(frozen=True)
class RunTarget:
    dataset_id: str
    run_id: str
    run_root: Path


# ---------------------------
# Run discovery
# ---------------------------
def discover_run_roots(
    *,
    artifacts_root: Path,
    dataset_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> List[RunTarget]:
    """
    Discover Rule A run roots under:
      <artifacts_root>/datasets/<dataset_id>/runs/<run_id>/
    """
    datasets_dir = artifacts_root / "datasets"
    if not datasets_dir.exists():
        return []

    targets: List[RunTarget] = []

    dataset_dirs = []
    if dataset_id is not None:
        p = datasets_dir / dataset_id
        if p.exists():
            dataset_dirs = [p]
    else:
        dataset_dirs = [p for p in datasets_dir.iterdir() if p.is_dir()]

    for ds_dir in dataset_dirs:
        runs_dir = ds_dir / "runs"
        if not runs_dir.exists():
            continue

        run_dirs = []
        if run_id is not None:
            p = runs_dir / run_id
            if p.exists():
                run_dirs = [p]
        else:
            run_dirs = [p for p in runs_dir.iterdir() if p.is_dir()]

        for rdir in run_dirs:
            targets.append(
                RunTarget(
                    dataset_id=ds_dir.name,
                    run_id=rdir.name,
                    run_root=rdir,
                )
            )

    # deterministic ordering (nice for CI logs)
    targets.sort(key=lambda t: (t.dataset_id, t.run_id))
    return targets


# ---------------------------
# Deep checks (optional)
# ---------------------------
def deep_check_synth_index(run_root: Path) -> List[str]:
    """
    Validate that synthetic/index.json exists and all referenced paths exist.
    Expects index.json like:
      {
        "paths": [{"path": "...", "label": 0}, ...],
        ...
      }
    or
      {"items": [...]} — we support both.
    """
    errs: List[str] = []
    idx_path = run_root / "synthetic" / "index.json"
    if not idx_path.exists():
        errs.append("synthetic/index.json missing (required by Rule A).")
        return errs

    try:
        idx = read_json(idx_path)
    except Exception as e:
        return [f"synthetic/index.json unreadable: {e}"]

    items = idx.get("paths") or idx.get("items") or []
    if not isinstance(items, list):
        return ["synthetic/index.json must contain a list under 'paths' or 'items'."]

    missing_files = 0
    checked = 0

    for it in items:
        if not isinstance(it, dict):
            continue
        p = it.get("path")
        if not p:
            continue
        checked += 1

        # Allow both absolute and relative paths.
        pth = Path(p)
        if not pth.is_absolute():
            pth = (run_root / pth).resolve()

        if not pth.exists():
            missing_files += 1
            if missing_files <= 10:
                errs.append(f"synthetic/index.json references missing file: {pth}")

    if checked == 0:
        errs.append("synthetic/index.json has no usable 'path' entries (paths/items list empty).")
    elif missing_files > 10:
        errs.append(f"synthetic/index.json has {missing_files} missing files (showing first 10).")

    return errs


def deep_check_npy(run_root: Path) -> List[str]:
    """
    Lightweight checks for synthetic/npy outputs (no heavy dependencies).
    - x_synth.npy exists
    - y_synth.npy exists
    - shape sanity: x is 4D, y is 1D or 2D, lengths match
    """
    errs: List[str] = []
    npy_dir = run_root / "synthetic" / "npy"
    x_path = npy_dir / "x_synth.npy"
    y_path = npy_dir / "y_synth.npy"

    if not x_path.exists():
        errs.append(f"missing {x_path}")
        return errs
    if not y_path.exists():
        errs.append(f"missing {y_path}")
        return errs

    try:
        import numpy as np  # local import
        x = np.load(x_path, mmap_mode="r")
        y = np.load(y_path, mmap_mode="r")
    except Exception as e:
        return [f"failed to load npy files: {e}"]

    if getattr(x, "ndim", None) != 4:
        errs.append(f"x_synth.npy expected 4D (N,H,W,C), got shape={getattr(x,'shape',None)}")
    if getattr(y, "ndim", None) not in (1, 2):
        errs.append(f"y_synth.npy expected 1D (N,) or 2D (N,K), got shape={getattr(y,'shape',None)}")

    # Match N
    try:
        nx = int(x.shape[0])
        ny = int(y.shape[0])
        if nx != ny:
            errs.append(f"x/y length mismatch: x has N={nx}, y has N={ny}")
    except Exception:
        errs.append("could not compare x/y lengths (unexpected shapes).")

    return errs


# ---------------------------
# Main validation per run
# ---------------------------
def validate_run(
    *,
    target: RunTarget,
    schemas_dir: Path,
    check_synth_index: bool = False,
    check_npy: bool = False,
) -> Tuple[bool, List[str], List[str]]:
    """
    Returns: (ok, errors, warnings)
    """
    errors: List[str] = []
    warnings: List[str] = []

    run_root = target.run_root

    # 1) required tree
    missing = exists_all(run_root, REQUIRED_RELATIVE_PATHS)
    if missing:
        errors.append(f"Missing required Rule A outputs: {missing}")

    # 2) schema validation
    if schemas_dir.exists():
        # load schemas once per run (small files, fine)
        def load_schema(name: str) -> Optional[Dict[str, Any]]:
            p = schemas_dir / name
            if not p.exists():
                warnings.append(f"Schema not found: {p} (skipping)")
                return None
            try:
                return read_json(p)
            except Exception as e:
                warnings.append(f"Schema unreadable: {p}: {e}")
                return None

        # run_manifest
        manifest_path = run_root / "run_manifest.json"
        if manifest_path.exists():
            schema = load_schema(SCHEMA_FILES["run_manifest"])
            if schema is not None:
                err = validate_with_schema(read_json(manifest_path), schema, where=str(manifest_path))
                if err:
                    (warnings if err.startswith("[skip]") else errors).append(err)

        # eval_summary
        summary_path = run_root / "eval" / "eval_summary.json"
        if summary_path.exists():
            schema = load_schema(SCHEMA_FILES["eval_summary"])
            if schema is not None:
                err = validate_with_schema(read_json(summary_path), schema, where=str(summary_path))
                if err:
                    (warnings if err.startswith("[skip]") else errors).append(err)

        # run_events.jsonl
        events_path = run_root / "run_events.jsonl"
        if events_path.exists():
            schema = load_schema(SCHEMA_FILES["run_events"])
            if schema is not None:
                # validate each event line-by-line
                if jsonschema is None:
                    warnings.append("[skip] jsonschema not installed (cannot validate run_events.jsonl).")
                else:
                    try:
                        for ev in iter_jsonl(events_path):
                            jsonschema.validate(instance=ev, schema=schema)
                    except Exception as e:
                        errors.append(f"{events_path}: schema validation failed: {e}")

    else:
        warnings.append(f"Schemas dir not found: {schemas_dir} (skipping schema validation)")

    # 3) optional deep checks
    if check_synth_index:
        errors.extend(deep_check_synth_index(run_root))
    if check_npy:
        errors.extend(deep_check_npy(run_root))

    ok = len(errors) == 0
    return ok, errors, warnings


# ---------------------------
# CLI
# ---------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate Rule A artifacts (tree + schema validation).")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--artifacts-root", type=str, help="Root artifacts directory (Rule A): artifacts/")
    g.add_argument("--run-root", type=str, help="Direct run root: artifacts/datasets/<dataset_id>/runs/<run_id>/")

    p.add_argument("--dataset-id", type=str, default=None, help="Filter by dataset_id (when using --artifacts-root)")
    p.add_argument("--run-id", type=str, default=None, help="Filter by run_id (when using --artifacts-root)")

    p.add_argument("--schemas-dir", type=str, default=None, help="Schemas dir (default: src/gencysynth/schemas)")

    p.add_argument("--check-synth-index", action="store_true", help="Verify synthetic/index.json referenced files exist")
    p.add_argument("--check-npy", action="store_true", help="Check synthetic/npy x_synth/y_synth shapes")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    schemas_dir = Path(args.schemas_dir).expanduser().resolve() if args.schemas_dir else default_schemas_dir()

    targets: List[RunTarget] = []

    if args.run_root:
        run_root = Path(args.run_root).expanduser().resolve()
        # try to parse dataset_id/run_id from canonical structure
        # .../datasets/<dataset_id>/runs/<run_id>
        try:
            run_id = run_root.name
            dataset_id = run_root.parent.parent.name  # runs/<run_id> -> <dataset_id>
        except Exception:
            dataset_id, run_id = "unknown", run_root.name
        targets = [RunTarget(dataset_id=dataset_id, run_id=run_id, run_root=run_root)]
    else:
        artifacts_root = Path(args.artifacts_root).expanduser().resolve()
        targets = discover_run_roots(
            artifacts_root=artifacts_root,
            dataset_id=args.dataset_id,
            run_id=args.run_id,
        )

    if not targets:
        _print("[error] No runs found to validate.")
        return 2

    failures = 0

    for t in targets:
        _print(f"\n=== VALIDATE: dataset={t.dataset_id} run={t.run_id} ===")
        ok, errors, warnings = validate_run(
            target=t,
            schemas_dir=schemas_dir,
            check_synth_index=bool(args.check_synth_index),
            check_npy=bool(args.check_npy),
        )

        for w in warnings:
            _print(f"[warn] {w}")
        for e in errors:
            _print(f"[fail] {e}")

        if ok:
            _print("[ok] Rule A validation passed.")
        else:
            failures += 1
            _print("[bad] Rule A validation failed.")

    _print("\n========================")
    _print(f"VALIDATION SUMMARY: {len(targets) - failures} passed, {failures} failed")
    _print("========================")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())