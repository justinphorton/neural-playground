"""Read recorded trials from one stage or live artifact directory."""

from __future__ import annotations

from pathlib import Path

from .records import TrialRecord, load_trial


def load_records(run_dir: Path, slot: int, split: str | None = None) -> list[TrialRecord]:
    paths = sorted((Path(run_dir) / "raw_trials").glob(f"*_slot{slot}_r*.npz"))
    records = [load_trial(path) for path in paths]
    if split is not None:
        records = [record for record in records if record.metadata.get("split") == split]
    return records
