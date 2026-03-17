# src/gencysynth/utils/schema.py
"""
GenCyberSynth — Schema utilities (Rule A / manifest_driven)

Purpose
-------
Provide a single place to:
- locate JSON Schemas stored under <repo_root>/schemas/
- load schemas as dicts
- validate manifests / registry files / event logs
- write validated JSON for artifacts

This module assumes the repo contains:
  schemas/<name>.schema.json

Examples:
  - dataset_fingerprint.schema.json
  - run_manifest.schema.json
  - eval_summary.schema.json
  - dataset_registry.schema.json
  - run_events.schema.json

Validation is optional at runtime; if jsonschema is not installed,
we fail with a clear error message.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from gencysynth.utils.paths import find_repo_root
from gencysynth.utils.io import load_json, save_json


class SchemaNotFoundError(FileNotFoundError):
    pass


def schemas_dir() -> Path:
    """Return <repo_root>/schemas."""
    return find_repo_root() / "schemas"


def schema_path(name: str) -> Path:
    """
    Resolve schema path by name.

    Accepts:
      - "run_manifest"         -> schemas/run_manifest.schema.json
      - "run_manifest.schema"  -> schemas/run_manifest.schema.json (tolerant)
      - "run_manifest.schema.json" (tolerant)
    """
    n = name.strip()
    n = n.replace(".schema.json", "").replace(".schema", "")
    p = schemas_dir() / f"{n}.schema.json"
    if not p.exists():
        raise SchemaNotFoundError(f"Schema not found: {p}")
    return p


def load_schema(name: str) -> Dict[str, Any]:
    """Load a schema dict from schemas/<name>.schema.json."""
    p = schema_path(name)
    return load_json(p)


def validate(instance: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """
    Validate a JSON instance against a schema.

    Requires: `jsonschema` package.
    """
    try:
        import jsonschema  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "jsonschema is required for schema validation but is not installed. "
            "Install with: pip install jsonschema"
        ) from e

    jsonschema.validate(instance=instance, schema=schema)


def validate_named(instance: Dict[str, Any], schema_name: str) -> None:
    """Validate a JSON instance against a named schema in /schemas."""
    schema = load_schema(schema_name)
    validate(instance, schema)


def save_validated_json(
    instance: Dict[str, Any],
    out_path: Path,
    *,
    schema_name: Optional[str] = None,
    schema: Optional[Dict[str, Any]] = None,
    indent: int = 2,
) -> None:
    """
    Validate then write JSON.

    Use exactly one of:
      - schema_name="run_manifest"
      - schema={...}

    This is intended for writing artifacts like:
      artifacts/.../run_manifest.json
      artifacts/.../dataset_fingerprint.json
      artifacts/.../eval_summary.json
    """
    if (schema_name is None) == (schema is None):
        raise ValueError("Provide exactly one of schema_name or schema.")

    sch = schema if schema is not None else load_schema(schema_name or "")
    validate(instance, sch)

    # Use existing io.save_json to keep formatting consistent repo_wide.
    save_json(instance, out_path, indent=indent)
