# src/gencysynth/adapters/tests/test_registry_resolution.py
"""
Registry resolution tests (adapter-level).

Purpose
-------
Adapters are our "front-door" for orchestration. Under Rule A:
  - datasets resolve by dataset_id
  - models resolve by (family, variant)
and resolution must fail with helpful, typed errors when something is missing.

These tests validate:
  1) Registration succeeds and resolution returns the exact registered adapter.
  2) Missing keys raise adapter-specific exceptions (or at least clear errors).
  3) Registry can list what is available (useful for CLI / debug).

The tests are written defensively to tolerate minor API naming differences,
but they enforce the core behavior strictly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, runtime_checkable

import pytest


def _pick_callable(mod, names: list[str]) -> Callable:
    for n in names:
        fn = getattr(mod, n, None)
        if callable(fn):
            return fn
    raise AssertionError(
        f"Could not find expected callable in {getattr(mod, '__name__', mod)}. Tried: {names}"
    )


def _maybe_callable(mod, names: list[str]) -> Optional[Callable]:
    for n in names:
        fn = getattr(mod, n, None)
        if callable(fn):
            return fn
    return None


@runtime_checkable
class _DummyAdapter(Protocol):
    """
    Minimal shape of an adapter for test purposes.
    Your real ModelAdapter/DatasetAdapter Protocols can be richer; we only need an object.
    """
    def name(self) -> str: ...


@dataclass(frozen=True)
class DummyModelAdapter:
    family: str
    variant: str

    def name(self) -> str:
        return f"{self.family}/{self.variant}"


@dataclass(frozen=True)
class DummyDatasetAdapter:
    dataset_id: str

    def name(self) -> str:
        return self.dataset_id


def test_model_registry_register_and_resolve_roundtrip():
    """
    Register (family, variant) -> adapter and ensure resolve returns the same object/class.
    """
    import gencysynth.adapters.models.registry as mreg

    register_fn = _pick_callable(mreg, ["register", "register_model", "register_adapter", "register_model_adapter"])
    resolve_fn = _pick_callable(mreg, ["resolve", "resolve_model", "get", "get_model", "resolve_model_adapter"])
    list_fn = _maybe_callable(mreg, ["list", "list_models", "available", "available_models"])

    # If the registry offers a clear/reset helper, use it so tests are isolated
    clear_fn = _maybe_callable(mreg, ["clear", "reset", "clear_registry", "reset_registry"])
    if clear_fn:
        clear_fn()

    adapter = DummyModelAdapter(family="gan", variant="dcgan")

    # Register it
    register_fn("gan", "dcgan", adapter)

    # Resolve it
    got = resolve_fn("gan", "dcgan")
    assert got is adapter or got == adapter, "Resolved adapter must match what was registered."

    # Optional listing
    if list_fn:
        items = list_fn()
        # tolerate dict/list/iterables
        s = str(items)
        assert "gan" in s and "dcgan" in s, f"Registry listing should include registered entries. Got: {items}"


def test_model_registry_missing_key_raises_helpful_error():
    """
    Missing (family, variant) must raise a clear error (preferably AdapterNotFoundError).
    """
    import gencysynth.adapters.models.registry as mreg
    import gencysynth.adapters.errors as aerr

    resolve_fn = _pick_callable(mreg, ["resolve", "resolve_model", "get", "get_model", "resolve_model_adapter"])

    with pytest.raises(Exception) as exc:
        resolve_fn("gan", "definitely-not-a-real-variant")

    # Prefer typed error if available
    not_found = getattr(aerr, "AdapterNotFoundError", None)
    if not_found and isinstance(exc.value, not_found):
        assert True
    else:
        # Otherwise ensure message is helpful
        msg = str(exc.value).lower()
        assert "gan" in msg
        assert "variant" in msg or "not found" in msg or "unknown" in msg


def test_dataset_registry_register_and_resolve_roundtrip():
    """
    Register dataset_id -> dataset adapter and ensure resolve returns the same.
    """
    import gencysynth.adapters.datasets.registry as dreg

    register_fn = _pick_callable(dreg, ["register", "register_dataset", "register_adapter", "register_dataset_adapter"])
    resolve_fn = _pick_callable(dreg, ["resolve", "resolve_dataset", "get", "get_dataset", "resolve_dataset_adapter"])
    list_fn = _maybe_callable(dreg, ["list", "list_datasets", "available", "available_datasets"])

    clear_fn = _maybe_callable(dreg, ["clear", "reset", "clear_registry", "reset_registry"])
    if clear_fn:
        clear_fn()

    adapter = DummyDatasetAdapter(dataset_id="ustc-tfc2016")

    register_fn("ustc-tfc2016", adapter)

    got = resolve_fn("ustc-tfc2016")
    assert got is adapter or got == adapter

    if list_fn:
        items = list_fn()
        assert "ustc" in str(items).lower()


def test_dataset_registry_missing_key_raises_helpful_error():
    """
    Missing dataset id must raise a clear error.
    """
    import gencysynth.adapters.datasets.registry as dreg
    import gencysynth.adapters.errors as aerr

    resolve_fn = _pick_callable(dreg, ["resolve", "resolve_dataset", "get", "get_dataset", "resolve_dataset_adapter"])

    with pytest.raises(Exception) as exc:
        resolve_fn("dataset-does-not-exist")

    not_found = getattr(aerr, "DatasetNotFoundError", None) or getattr(aerr, "AdapterNotFoundError", None)
    if not_found and isinstance(exc.value, not_found):
        assert True
    else:
        msg = str(exc.value).lower()
        assert "dataset" in msg or "not found" in msg or "unknown" in msg
