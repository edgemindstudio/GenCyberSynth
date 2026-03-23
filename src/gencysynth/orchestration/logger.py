# src/gencysynth/orchestration/logger.py
"""
GenCyberSynth — Run logger (per_run logs to artifacts/logs)
==========================================================

We want logs to be:
- per_run (no collisions)
- easy to tail on HPC
- written to both console and a file

Layout (Rule A)
---------------
  artifacts/logs/<dataset_id>/<model_tag>/<run_id>/
    run.log         (plain text)
    events.jsonl    (structured JSONL events, optional)

This module provides:
- get_run_logger(...) -> python logging.Logger configured for a run
- log_event_jsonl(...) -> append structured events (multi_process friendly)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from gencysynth.utils.io import append_jsonl
from gencysynth.utils.paths import ensure_dir


def get_run_logger(
    *,
    name: str,
    log_dir: Path,
    filename: str = "run.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create or reuse a logger that writes to both console + a run_scoped file.
    Safe to call multiple times (won't duplicate handlers).
    """
    ensure_dir(log_dir)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times
    if any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        return logger

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(str(Path(log_dir) / filename))
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Be nice as a library: don't propagate to root unless caller wants that
    logger.propagate = False
    return logger


def log_event_jsonl(
    *,
    log_dir: Path,
    event: Dict[str, Any],
    filename: str = "events.jsonl",
) -> Path:
    """
    Append a single structured JSON event to a per_run JSONL file.
    Non_atomic by design (multi_process logging).
    """
    ensure_dir(log_dir)
    return append_jsonl(Path(log_dir) / filename, event)


__all__ = ["get_run_logger", "log_event_jsonl"]
