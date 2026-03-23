# src/gencysynth/adapters/errors.py
"""
Adapter_specific exceptions.

These errors are meant to be:
- actionable (tell the user what to fix),
- contract_oriented (point to Rule A or missing required outputs),
- non_mysterious (include model_tag, dataset_id, run_id when possible).
"""

from __future__ import annotations


class AdapterError(RuntimeError):
    """Base class for all adapter errors."""


class AdapterNotFoundError(AdapterError):
    """Raised when no adapter is registered for a model_tag."""


class AdapterContractError(AdapterError):
    """
    Raised when an adapter violates the stable contract:
      - writes to wrong locations
      - returns malformed results
      - missing required artifacts (manifest, schema, etc.)
    """
