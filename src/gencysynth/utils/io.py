# src/gencysynth/utils/io.py
"""
GenCyberSynth — Safe file IO helpers

Design
------
- Atomic overwrite for text/JSON (write temp + fsync + replace).
- Non-atomic append for JSONL (log behavior; supports multi-process appends).
- JSON encoding supports Path and optional NumPy scalars/arrays without hard dependency.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Generator, Optional, Union

from gencysynth.utils.paths import ensure_dir

_JSON_COMPACT_SEPARATORS = (",", ":")


def _json_default(obj: Any) -> Any:
    """
    Default converter for json.dumps to handle common non-JSON types.

    - pathlib.Path → str
    - NumPy scalar → native Python scalar
    - NumPy array  → list
    """
    if isinstance(obj, Path):
        return str(obj)

    # Optional NumPy support without hard dependency
    try:
        import numpy as _np  # type: ignore
        if isinstance(obj, _np.generic):
            return obj.item()
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _atomic_write_bytes(dst: Path, data: bytes) -> None:
    """
    Atomically write bytes to dst by writing a temp file in the same directory
    then os.replace() it into place.
    """
    dst = Path(dst)
    ensure_dir(dst.parent)

    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(dst.parent)) as tmp:
        tmp_path = Path(tmp.name)
        try:
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
            finally:
                raise

    os.replace(str(tmp_path), str(dst))


def write_text(
    path: Union[str, Path],
    text: str,
    *,
    append: bool = False,
    ensure_trailing_newline: bool = False,
    atomic: bool = True,
    encoding: str = "utf-8",
) -> Path:
    """
    Write or append UTF-8 text.

    - append=True is non-atomic by design (log behavior).
    - atomic=True applies only to overwrite mode (append=False).
    """
    p = Path(path)

    if ensure_trailing_newline and not text.endswith("\n"):
        text += "\n"

    if append:
        ensure_dir(p.parent)
        with p.open("a", encoding=encoding) as f:
            f.write(text)
        return p

    if atomic:
        _atomic_write_bytes(p, text.encode(encoding))
        return p

    ensure_dir(p.parent)
    with p.open("w", encoding=encoding) as f:
        f.write(text)
    return p


def write_json(
    path: Union[str, Path],
    obj: Any,
    *,
    indent: Optional[int] = 2,
    sort_keys: bool = True,
    ensure_ascii: bool = False,
    atomic: bool = True,
) -> Path:
    """
    Serialize obj to JSON and write to path (atomic by default).
    """
    separators = None if indent is not None else _JSON_COMPACT_SEPARATORS
    data = json.dumps(
        obj,
        indent=indent,
        sort_keys=sort_keys,
        ensure_ascii=ensure_ascii,
        separators=separators,
        default=_json_default,
    )

    p = Path(path)
    if atomic:
        _atomic_write_bytes(p, data.encode("utf-8" if not ensure_ascii else "ascii"))
    else:
        write_text(p, data, append=False, atomic=False)
    return p


def read_json(path: Union[str, Path]) -> Any:
    """Read JSON from path and return decoded object."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(
    path: Union[str, Path],
    obj: Any,
    *,
    sort_keys: bool = False,
    ensure_ascii: bool = False,
) -> Path:
    """
    Append one compact JSON object as a single line to a JSONL file.

    Not atomic by design (supports multi-process logging).
    """
    line = json.dumps(
        obj,
        ensure_ascii=ensure_ascii,
        separators=_JSON_COMPACT_SEPARATORS,
        sort_keys=sort_keys,
        default=_json_default,
    )
    return write_text(path, line, append=True, ensure_trailing_newline=True, atomic=False)


def iter_jsonl(path: Union[str, Path]) -> Generator[Any, None, None]:
    """Lazily iterate objects from a JSONL file (skips blank lines)."""
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)
