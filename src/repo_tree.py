#!/usr/bin/env python3
"""
repo_tree.py — Print a complete repository tree from the *current directory* downward.

This version is optimized for sharing in chats:
- Includes hidden files by default
- Skips noisy/huge directories by default: .git, artifacts, logs, USTC_TFC2016_malware
- Does not walk above the current directory
- Robust permission handling (fails fast and explains how to rerun)

Usage:
  python3 repo_tree.py
  python3 repo_tree.py --output repo_tree.txt
  python3 repo_tree.py --max_depth 6
  python3 repo_tree.py --exclude .git,artifacts,logs
  python3 repo_tree.py --follow_symlinks        # Not recommended unless you know there are no cycles
  python3 repo_tree.py --include_sizes          # Adds sizes for regular files
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set, Tuple


DEFAULT_EXCLUDES = {".git", "artifacts", "logs", "USTC_TFC2016_malware"}


@dataclass(frozen=True)
class Options:
    max_depth: Optional[int]
    follow_symlinks: bool
    include_sizes: bool
    output: Optional[str]
    excludes: Set[str]


def is_admin() -> bool:
    if os.name == "posix":
        return os.geteuid() == 0
    return False


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for u in units:
        if size < 1024.0 or u == units[-1]:
            return f"{size:.1f}{u}" if u != "B" else f"{int(size)}B"
        size /= 1024.0
    return f"{num_bytes}B"


def safe_stat(path: str, follow_symlinks: bool) -> Optional[os.stat_result]:
    try:
        return os.stat(path, follow_symlinks=follow_symlinks)
    except (FileNotFoundError, OSError):
        return None


def list_dir_entries(path: str) -> List[os.DirEntry]:
    try:
        with os.scandir(path) as it:
            return list(it)
    except PermissionError as e:
        raise PermissionError(f"Permission denied while scanning: {path}") from e
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Directory not found during scan (race): {path}") from e


def sort_entries(entries: Iterable[os.DirEntry]) -> List[os.DirEntry]:
    def key_fn(de: os.DirEntry) -> Tuple[int, str]:
        try:
            is_dir = de.is_dir(follow_symlinks=False)
        except OSError:
            is_dir = False
        return (0 if is_dir else 1, de.name.lower())

    return sorted(entries, key=key_fn)


def format_name(de: os.DirEntry, opts: Options) -> str:
    name = de.name

    try:
        is_link = de.is_symlink()
    except OSError:
        is_link = False

    size_suffix = ""
    if opts.include_sizes:
        try:
            if de.is_file(follow_symlinks=False):
                st = safe_stat(de.path, follow_symlinks=False)
                if st is not None:
                    size_suffix = f" ({human_size(st.st_size)})"
        except OSError:
            pass

    if is_link:
        try:
            target = os.readlink(de.path)
            st = safe_stat(de.path, follow_symlinks=True)
            if st is None:
                return f"{name} -> {target} [BROKEN]{size_suffix}"
            return f"{name} -> {target}{size_suffix}"
        except OSError:
            return f"{name} [SYMLINK]{size_suffix}"

    return f"{name}{size_suffix}"


def should_exclude(name: str, excludes: Set[str]) -> bool:
    # Exact match on directory/file name (not path) keeps behavior predictable.
    return name in excludes


def build_tree_lines(root: str, opts: Options) -> List[str]:
    lines: List[str] = []
    abs_root = os.path.abspath(root)
    root_label = os.path.basename(abs_root) or abs_root
    lines.append(f"{root_label}/  (root: {abs_root})")
    lines.append(f"(excludes: {', '.join(sorted(opts.excludes))})")

    def walk(dir_path: str, prefix: str, depth: int) -> None:
        if opts.max_depth is not None and depth > opts.max_depth:
            return

        entries = sort_entries(list_dir_entries(dir_path))

        # Filter excluded names at this level
        entries = [de for de in entries if not should_exclude(de.name, opts.excludes)]

        total = len(entries)
        for idx, de in enumerate(entries):
            is_last = (idx == total - 1)
            branch = "└── " if is_last else "├── "
            next_prefix = prefix + ("    " if is_last else "│   ")

            try:
                is_dir = de.is_dir(follow_symlinks=opts.follow_symlinks)
            except OSError:
                is_dir = False

            display = format_name(de, opts)
            if is_dir:
                lines.append(f"{prefix}{branch}{display}/")
                if de.is_symlink() and not opts.follow_symlinks:
                    continue
                walk(de.path, next_prefix, depth + 1)
            else:
                lines.append(f"{prefix}{branch}{display}")

    walk(root, prefix="", depth=1)
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print a full tree of the current directory (including hidden entries), with excludes."
    )
    parser.add_argument("--max_depth", type=int, default=None,
                        help="Maximum depth to print (1 means only direct children). Default: unlimited.")
    parser.add_argument("--follow_symlinks", action="store_true",
                        help="Follow symlinks for directory traversal. Default: off (safer).")
    parser.add_argument("--include_sizes", action="store_true",
                        help="Include human_readable sizes for files. Default: off.")
    parser.add_argument("--output", type=str, default=None,
                        help="Write output to a file instead of stdout.")
    parser.add_argument("--exclude", type=str, default="",
                        help="Comma_separated names to exclude (e.g., .git,artifacts,logs).")

    args = parser.parse_args()

    user_excludes = {x.strip() for x in args.exclude.split(",") if x.strip()}
    excludes = set(DEFAULT_EXCLUDES) | user_excludes

    opts = Options(
        max_depth=args.max_depth,
        follow_symlinks=args.follow_symlinks,
        include_sizes=args.include_sizes,
        output=args.output,
        excludes=excludes,
    )

    root = os.getcwd()
    try:
        lines = build_tree_lines(root, opts)
    except PermissionError as e:
        msg = [
            "ERROR: Permission denied while listing the repository tree.",
            f"Reason: {e}",
            "",
            "Recommendation:",
            "  Re_run with elevated privileges to include restricted paths:",
            "    sudo python3 repo_tree.py",
            "",
            "If you do NOT want admin privileges, remove/relocate restricted files",
            "or ensure the current user has read/execute permissions.",
        ]
        print("\n".join(msg), file=sys.stderr)
        return 2

    output_text = "\n".join(lines) + "\n"
    if opts.output:
        try:
            with open(opts.output, "w", encoding="utf_8") as f:
                f.write(output_text)
            print(f"Wrote repository tree to: {opts.output}")
        except PermissionError:
            print(f"ERROR: Cannot write to output file '{opts.output}' (permission denied).", file=sys.stderr)
            return 3
    else:
        sys.stdout.write(output_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
