# tools/smoke/smoke_onepass.py

#!/usr/bin/env python3
"""
tools/smoke/smoke_onepass.py

Smoke test runner: run a tiny end-to-end suite across multiple model families
(<5 epochs) and validate Rule A artifact layout.

What this script does
---------------------
For each (family, variant) in a smoke list:
  1) launches a run via the GenCyberSynth CLI (train + synth + eval + plots)
  2) discovers the created run root under:
        <ARTIFACTS_ROOT>/datasets/<dataset_id>/runs/<run_id>/
  3) validates required Rule A files:
        - run_manifest.json
        - run_events.jsonl
        - resolved_config.yaml
        - eval/eval_summary.json
        - synthetic/index.json
  4) optionally validates schemas (if jsonschema installed)

This is meant to catch:
- path regressions / hardcoded artifacts paths
- missing required outputs
- adapter registry resolution issues
- broken end-to-end wiring

Usage examples
--------------
# simplest (uses configs/smoke.yaml or your smoke suite config)
python tools/smoke/smoke_onepass.py \
  --dataset-id ustc-tfc2016-npy \
  --artifacts-root artifacts \
  --suite configs/smoke/smoke.yaml \
  --max-epochs 3

# customize the CLI invocation template if your command differs:
python tools/smoke/smoke_onepass.py \
  --cmd-template "python -m gencysynth.cli.main run --config {suite} --dataset {dataset_id} --family {family} --variant {variant} --override {override_yaml}"
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml


# -----------------------------
# Defaults: 7 families
# -----------------------------
DEFAULT_MODELS: List[Tuple[str, str]] = [
    ("gan", "dcgan"),
    ("vae", "c-vae"),
    ("diffusion", "c-ddpm"),
    ("autoregressive", "pixelcnnpp"),
    ("gaussianmixture", "c-gmm-diag"),
    ("maskedautoflow", "c-maf-affine"),
    ("restrictedboltzmann", "c-rbm-bernoulli"),
]


# -----------------------------
# Rule A required paths
# -----------------------------
REQUIRED_RELATIVE_PATHS = [
    "run_manifest.json",
    "run_events.jsonl",
    "resolved_config.yaml",
    "synthetic/index.json",
    "eval/eval_summary.json",
]


# -----------------------------
# Helpers
# -----------------------------
def _print(msg: str) -> None:
    print(msg, flush=True)


def read_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: Path, txt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(txt, encoding="utf-8")


def write_yaml(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def rule_a_run_root(artifacts_root: Path, dataset_id: str, run_id: str) -> Path:
    """Canonical Rule A run root."""
    return artifacts_root / "datasets" / dataset_id / "runs" / run_id


def find_latest_run_root(artifacts_root: Path, dataset_id: str) -> Optional[Path]:
    """
    Find the most recently modified run directory under:
      artifacts/datasets/<dataset_id>/runs/<run_id>
    """
    runs_dir = artifacts_root / "datasets" / dataset_id / "runs"
    if not runs_dir.exists():
        return None
    candidates = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def validate_rule_a_outputs(run_root: Path) -> List[str]:
    """
    Returns list of missing required files (relative to run_root).
    """
    missing: List[str] = []
    for rel in REQUIRED_RELATIVE_PATHS:
        if not (run_root / rel).exists():
            missing.append(rel)
    return missing


def try_validate_schemas(run_root: Path, schemas_dir: Path) -> List[str]:
    """
    Optional schema validation (requires `jsonschema`).
    Returns list of validation errors (strings).
    """
    try:
        import jsonschema  # type: ignore
    except Exception:
        return ["jsonschema not installed; skipping schema validation."]

    errors: List[str] = []

    def _load_schema(name: str) -> Dict:
        p = schemas_dir / name
        return read_json(p)

    # Validate run_manifest.json
    try:
        schema = _load_schema("run_manifest.schema.json")
        instance = read_json(run_root / "run_manifest.json")
        jsonschema.validate(instance=instance, schema=schema)
    except Exception as e:
        errors.append(f"run_manifest.json schema validation failed: {e}")

    # Validate eval_summary.json
    try:
        schema = _load_schema("eval_summary.schema.json")
        instance = read_json(run_root / "eval" / "eval_summary.json")
        jsonschema.validate(instance=instance, schema=schema)
    except Exception as e:
        errors.append(f"eval_summary.json schema validation failed: {e}")

    # Validate run_events.jsonl line-by-line
    try:
        schema = _load_schema("run_events.schema.json")
        events_path = run_root / "run_events.jsonl"
        with open(events_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                ev = json.loads(line)
                jsonschema.validate(instance=ev, schema=schema)
    except Exception as e:
        errors.append(f"run_events.jsonl schema validation failed: {e}")

    return errors


# -----------------------------
# Override config generation
# -----------------------------
def make_smoke_override_yaml(
    *,
    max_epochs: int,
    samples_per_class: int,
    seed: int,
) -> Dict:
    """
    A minimal override layer that most families can honor.

    Families/variants may use slightly different key names; your adapters/config
    normalization should map these to the correct internal keys.

    Keep it small and safe:
    - tiny epochs
    - small synthetic budget
    - deterministic seed
    """
    return {
        "SEED": int(seed),
        "EPOCHS": int(max_epochs),              # used by most deep models
        "epochs": int(max_epochs),              # extra alias (some variants)
        "SAMPLES_PER_CLASS": int(samples_per_class),
        "samples_per_class": int(samples_per_class),
        "TRAIN": {"max_epochs": int(max_epochs)},  # if your orchestrator uses nested keys
        "SYNTH": {"samples_per_class": int(samples_per_class)},
        # Optional: disable expensive metrics during smoke
        "EVAL": {
            "enable_distribution": True,
            "enable_diversity": True,
            "enable_downstream": False,         # keep smoke fast; turn on in deeper CI
            "enable_calibration": False,
            "enable_privacy": False,
        },
        # Optional: disable heavy plots
        "REPORTING": {"enabled": False},
    }


# -----------------------------
# CLI runner
# -----------------------------
def run_one(
    *,
    cmd_template: str,
    suite_path: Path,
    dataset_id: str,
    family: str,
    variant: str,
    override_yaml_path: Path,
    env: Dict[str, str],
) -> int:
    """
    Invoke the repo CLI using a command template.

    The template is formatted with:
      {suite} {dataset_id} {family} {variant} {override_yaml}

    You can adapt the template to your actual CLI without changing script logic.
    """
    cmd = cmd_template.format(
        suite=str(suite_path),
        dataset_id=dataset_id,
        family=family,
        variant=variant,
        override_yaml=str(override_yaml_path),
    )

    _print(f"\n=== RUN: {family}/{variant} ===")
    _print(f"[cmd] {cmd}")

    proc = subprocess.run(
        cmd,
        shell=True,
        env=env,
    )
    return int(proc.returncode)


# -----------------------------
# Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GenCyberSynth smoke suite runner (<5 epochs) + Rule A layout validation."
    )
    p.add_argument("--dataset-id", required=True, help="dataset_id from dataset registry (Rule A namespace)")
    p.add_argument("--suite", required=True, help="suite YAML config (smoke suite)")

    p.add_argument("--artifacts-root", default="artifacts", help="Rule A artifacts root")
    p.add_argument("--schemas-dir", default="src/gencysynth/schemas", help="schemas dir for optional validation")

    p.add_argument("--max-epochs", type=int, default=3, help="max epochs for quick training (<5 recommended)")
    p.add_argument("--samples-per-class", type=int, default=16, help="synthetic budget per class for smoke")
    p.add_argument("--seed", type=int, default=42, help="seed for determinism")

    p.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="optional list of family:variant entries; defaults to 7-family list",
    )

    # This is the only repo-specific part you may need to tweak once.
    p.add_argument(
        "--cmd-template",
        default="python -m gencysynth.cli.main onepass --suite {suite} --dataset {dataset_id} "
                "--family {family} --variant {variant} --override {override_yaml}",
        help=textwrap.dedent(
            """\
            Command template used to launch runs.
            Variables: {suite} {dataset_id} {family} {variant} {override_yaml}

            If your CLI differs, adjust this value once and keep the rest unchanged.
            """
        ),
    )

    return p.parse_args()


def main() -> int:
    args = parse_args()

    artifacts_root = Path(args.artifacts_root).resolve()
    schemas_dir = Path(args.schemas_dir).resolve()
    suite_path = Path(args.suite).resolve()

    if not suite_path.exists():
        _print(f"[error] suite config not found: {suite_path}")
        return 2

    # Model list
    if args.models:
        models: List[Tuple[str, str]] = []
        for item in args.models:
            if ":" not in item:
                _print(f"[error] invalid model entry '{item}', expected family:variant")
                return 2
            fam, var = item.split(":", 1)
            models.append((fam.strip(), var.strip()))
    else:
        models = list(DEFAULT_MODELS)

    # Prepare override yaml in a temp-ish location under artifacts_root/_smoke
    smoke_dir = artifacts_root / "_smoke"
    ensure_dir(smoke_dir)
    override = make_smoke_override_yaml(
        max_epochs=args.max_epochs,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )
    override_yaml_path = smoke_dir / "override_smoke.yaml"
    write_yaml(override_yaml_path, override)

    # Environment: ensure the CLI sees ARTIFACTS_ROOT for Rule A
    env = dict(os.environ)
    env["ARTIFACTS_ROOT"] = str(artifacts_root)

    # Run all models
    failures: List[str] = []
    validated: List[str] = []

    # Capture "before" latest run root (for nicer detection)
    before_latest = find_latest_run_root(artifacts_root, args.dataset_id)

    for family, variant in models:
        rc = run_one(
            cmd_template=args.cmd_template,
            suite_path=suite_path,
            dataset_id=args.dataset_id,
            family=family,
            variant=variant,
            override_yaml_path=override_yaml_path,
            env=env,
        )
        if rc != 0:
            failures.append(f"{family}/{variant}: CLI returned {rc}")
            continue

        # Determine the run root created by this run.
        # Simplest heuristic: newest run directory after the command.
        after_latest = find_latest_run_root(artifacts_root, args.dataset_id)
        if after_latest is None or after_latest == before_latest:
            failures.append(f"{family}/{variant}: could not detect new run root under Rule A")
            continue

        run_root = after_latest
        missing = validate_rule_a_outputs(run_root)
        if missing:
            failures.append(f"{family}/{variant}: missing required outputs: {missing} (run_root={run_root})")
            continue

        # Optional schema validation
        schema_errors = try_validate_schemas(run_root, schemas_dir)
        # If jsonschema missing, we treat it as informational.
        hard_schema_errors = [e for e in schema_errors if "not installed" not in e]
        if hard_schema_errors:
            failures.append(f"{family}/{variant}: schema validation errors: {hard_schema_errors} (run_root={run_root})")
            continue

        validated.append(f"{family}/{variant} OK -> {run_root}")
        before_latest = after_latest  # advance baseline

    _print("\n========================")
    _print("SMOKE SUMMARY")
    _print("========================")
    for v in validated:
        _print(f"[ok] {v}")
    for f in failures:
        _print(f"[fail] {f}")

    if failures:
        _print("\nSome runs failed. Fix these before scaling large sweeps.")
        return 1

    _print("\nAll smoke runs passed Rule A validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())