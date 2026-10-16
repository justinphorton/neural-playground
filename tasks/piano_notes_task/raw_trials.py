"""Lossless, non-pickle raw trial artifacts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RawTrial:
    trial_id: str
    recording_id: str
    label: str
    repeat: int
    reservoir_hash: str
    noise_seed: int
    neuron_spike_times_ms: np.ndarray
    neuron_spike_ids: np.ndarray
    event_times_ms: np.ndarray
    event_channel_ids: np.ndarray
    phase_timings_ms: tuple[float, float, float] = (500.0, 1000.0, 1000.0)
    metadata: dict = field(default_factory=dict)


def save_raw_trial(trial: RawTrial, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    metadata = dict(trial.metadata)
    metadata.update({
        "trial_id": trial.trial_id, "recording_id": trial.recording_id,
        "label": trial.label, "repeat": trial.repeat,
        "reservoir_hash": trial.reservoir_hash, "noise_seed": trial.noise_seed,
        "phase_timings_ms": list(trial.phase_timings_ms),
    })
    with tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            neuron_spike_times_ms=np.asarray(trial.neuron_spike_times_ms, dtype=np.float64),
            neuron_spike_ids=np.asarray(trial.neuron_spike_ids, dtype=np.int64),
            event_times_ms=np.asarray(trial.event_times_ms, dtype=np.float64),
            event_channel_ids=np.asarray(trial.event_channel_ids, dtype=np.int32),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    tmp.replace(path)
    sidecar = path.with_suffix(path.suffix + ".json")
    sidecar_tmp = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.tmp")
    sidecar_tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    sidecar_tmp.replace(sidecar)


def load_raw_trial(path: Path) -> RawTrial:
    with np.load(Path(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        return RawTrial(
            trial_id=metadata.pop("trial_id"), recording_id=metadata.pop("recording_id"),
            label=metadata.pop("label"), repeat=int(metadata.pop("repeat")),
            reservoir_hash=metadata.pop("reservoir_hash"), noise_seed=int(metadata.pop("noise_seed")),
            neuron_spike_times_ms=data["neuron_spike_times_ms"], neuron_spike_ids=data["neuron_spike_ids"],
            event_times_ms=data["event_times_ms"], event_channel_ids=data["event_channel_ids"],
            phase_timings_ms=tuple(metadata.pop("phase_timings_ms")), metadata=metadata,
        )
