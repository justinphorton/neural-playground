"""Shared artifact paths and crash-safe array writes."""

from __future__ import annotations

import os
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np


def trial_record_path(raw_trials_dir: Path, trajectory_id: str, slot: int,
                      repeat: int) -> Path:
    """Return the stable filename used for one trajectory noise repeat."""
    return Path(raw_trials_dir) / f"{trajectory_id}_slot{slot}_r{repeat:02d}.npz"


def ensure_stage_directory(run_dir: Path, stage: str) -> Path:
    """Create and return the standard artifact directories for a run stage."""
    path = Path(run_dir) / stage
    path.mkdir(parents=True, exist_ok=True)
    for name in ("reservoirs", "raw_trials", "models", "metrics", "logs"):
        (path / name).mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_npz_atomic(path: Path, **arrays) -> None:
    """Write a complete compressed NPZ before replacing its checkpoint path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def save_json_atomic(path: Path, value: dict) -> None:
    """Write a complete JSON checkpoint before replacing its destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(json.dumps(value, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
