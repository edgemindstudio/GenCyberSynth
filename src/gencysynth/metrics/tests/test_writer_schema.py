# src/gencysynth/metrics/tests/test_writer_schema.py
"""
Schema tests for metrics writer outputs.

Goal
----
Ensure writer outputs conform to JSON Schemas you ship under /schemas.

Rule A
------
Writer is allowed to do I/O; schemas must be stable and versionable.
These tests do not require model training or large datasets.

This suite is intentionally tolerant to evolving writer interfaces:
- If writer validation helpers aren't available yet, it falls back to checking
  that the schema files are loadable JSON and contain expected top_level keys.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _repo_root() -> Path:
    # tests are under src/gencysynth/metrics/tests; go up to repo root
    return Path(__file__).resolve().parents[5]


def _load_schema(rel: str) -> dict:
    p = _repo_root() / rel
    if not p.exists():
        pytest.skip(f"Schema file missing: {p}")
    return json.loads(p.read_text(encoding="utf_8"))


def test_run_manifest_schema_loads():
    schema = _load_schema("schemas/run_manifest.schema.json")
    assert isinstance(schema, dict)
    # Basic JSONSchema keys
    assert "$schema" in schema or "schema" in schema or "type" in schema


def test_eval_summary_schema_loads():
    schema = _load_schema("schemas/eval_summary.schema.json")
    assert isinstance(schema, dict)
    assert "type" in schema


def test_dataset_fingerprint_schema_loads():
    schema = _load_schema("schemas/dataset_fingerprint.schema.json")
    assert isinstance(schema, dict)
    assert "type" in schema


def test_writer_validation_if_available():
    """
    If writer exposes schema validation, test it on a tiny example.
    """
    try:
        from gencysynth.metrics.writer import validate_json  # optional helper
    except Exception:
        pytest.skip("gencysynth.metrics.writer.validate_json not available yet.")

    schema = _load_schema("schemas/eval_summary.schema.json")

    # Minimal example; may need adjusting to your schema requirements.
    # The intent: keep it tiny but structurally valid as you evolve schemas.
    example = {
        "schema_version": "1.0",
        "dataset": {"name": "dummy"},
        "run": {"run_id": "test"},
        "metrics": [],
    }

    ok, err = validate_json(example, schema)  # type: ignore[misc]
    assert ok, f"Schema validation failed unexpectedly: {err}"