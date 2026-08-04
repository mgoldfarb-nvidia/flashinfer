"""Filesystem helpers for the flashinfer-cubin build cache."""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path


def copy_tree_contents(source: Path, destination: Path) -> None:
    """Atomically copy regular files without sharing inodes with the source."""
    if not source.exists():
        return

    for source_path in tuple(source.rglob("*")):
        relative_path = source_path.relative_to(source)
        if (
            not source_path.is_file()
            or source_path.is_symlink()
            or any(part.endswith((".lock", ".tmp")) for part in relative_path.parts)
        ):
            continue

        destination_path = destination / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if destination_path.exists() and source_path.samefile(destination_path):
            continue

        temporary_path = destination_path.with_name(
            f".{destination_path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            shutil.copy2(source_path, temporary_path)
            os.replace(temporary_path, destination_path)
        finally:
            temporary_path.unlink(missing_ok=True)


def prune_tree(root: Path, expected_files: set[Path]) -> None:
    """Remove files not selected by the current artifact manifest."""
    if not root.exists():
        return

    for path in tuple(root.rglob("*")):
        if (path.is_file() or path.is_symlink()) and path.relative_to(
            root
        ) not in expected_files:
            path.unlink()

    directories = (path for path in root.rglob("*") if path.is_dir())
    for directory in sorted(
        directories, key=lambda path: len(path.parts), reverse=True
    ):
        if not any(directory.iterdir()):
            directory.rmdir()
