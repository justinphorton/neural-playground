"""Permanent, hash-addressed manifest for the current C4/E4 training set."""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path


_FILENAME = re.compile(
    r"^(?P<note>C4|E4)_(?P<family>[^_]+)_velocity(?P<velocity>\d+)_hold(?P<hold>\d+)ms\.wav$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RecordingManifestRow:
    recording_id: str
    file_hash: str
    path: str
    note: str
    piano_family: str
    velocity: int
    hold_duration_ms: int
    condition_group_id: str
    dataset_split: str = "training"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse(path: Path) -> tuple[str, str, int, int]:
    match = _FILENAME.match(path.name)
    if not match:
        raise ValueError(f"unexpected training WAV filename: {path.name}")
    return (
        match.group("note").upper(), match.group("family"),
        int(match.group("velocity")), int(match.group("hold")),
    )


def build_training_manifest(root: Path, include_notes: tuple[str, ...] = ("C4", "E4")) -> list[RecordingManifestRow]:
    root = Path(root)
    expected = {note.upper() for note in include_notes}
    if expected != {"C4", "E4"}:
        raise ValueError("training manifest must contain C4 and E4 only")
    rows: list[RecordingManifestRow] = []
    for note in sorted(expected):
        note_dir = root / note
        if not note_dir.is_dir():
            raise FileNotFoundError(f"missing note directory: {note_dir}")
        for path in sorted(note_dir.glob("*.wav")):
            parsed_note, family, velocity, hold = _parse(path)
            if parsed_note != note:
                raise ValueError(f"note folder/filename mismatch: {path}")
            digest = file_sha256(path)
            group = f"{family}__velocity{velocity:03d}__hold{hold:04d}ms"
            rows.append(RecordingManifestRow(
                recording_id=f"rec_{digest[:16]}", file_hash=digest,
                path=str(path.resolve()), note=note, piano_family=family,
                velocity=velocity, hold_duration_ms=hold,
                condition_group_id=group,
            ))
    if not rows:
        raise ValueError("no C4/E4 WAVs found")
    groups: dict[str, set[str]] = {}
    for row in rows:
        groups.setdefault(row.condition_group_id, set()).add(row.note)
    incomplete = [group for group, notes in groups.items() if notes != {"C4", "E4"}]
    if incomplete:
        raise ValueError(f"each condition group needs one C4 and one E4: {incomplete[:3]}")
    if len([row for row in rows if row.note == "C4"]) != len([row for row in rows if row.note == "E4"]):
        raise ValueError("C4 and E4 counts must match")
    return sorted(rows, key=lambda row: (row.condition_group_id, row.note, row.path))


def write_manifest(rows: list[RecordingManifestRow], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(rows[0]).keys()) if rows else list(RecordingManifestRow.__dataclass_fields__)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
