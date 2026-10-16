"""Neural-only feature extraction from saved eight-electrode events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import re
import json
from pathlib import Path

import numpy as np


PRIMARY_MEMORY_WINDOWS_MS = tuple((float(start), float(start + 50)) for start in range(5, 955, 50))
DIAGNOSTIC_WINDOWS_MS = ((0.0, 5.0), (5.0, 25.0), (25.0, 50.0), (50.0, 100.0), (100.0, 200.0), (200.0, 350.0), (350.0, 500.0), (500.0, 750.0), (750.0, 1000.0))


@dataclass(frozen=True)
class FeatureDataset:
    X: np.ndarray
    labels: np.ndarray
    recording_ids: np.ndarray
    condition_groups: np.ndarray
    feature_mode: str
    window_ms: tuple[float, float]
    recording_electrodes: tuple[int, ...]
    is_control: np.ndarray | None = None


def _events(channels: Sequence[int], times_ms: Sequence[float], electrodes: tuple[int, ...], window_ms: tuple[float, float]):
    channels = np.asarray(channels)
    times = np.asarray(times_ms, dtype=float)
    start, end = map(float, window_ms)
    mask = (times >= start) & (times < end) & np.isin(channels, electrodes)
    return channels[mask], times[mask]


def extract_features(channels: Sequence[int], times_ms: Sequence[float], recording_electrodes: Sequence[int], window_ms: tuple[float, float], mode: str = "fine_temporal", bin_ms: float = 5.0) -> np.ndarray:
    electrodes = tuple(int(value) for value in recording_electrodes)
    if not electrodes:
        raise ValueError("recording_electrodes must be nonempty")
    if window_ms[1] <= window_ms[0]:
        raise ValueError("feature window must have positive duration")
    selected_channels, selected_times = _events(channels, times_ms, electrodes, window_ms)
    if mode == "spike_count":
        return np.asarray([np.sum(selected_channels == electrode) for electrode in electrodes], dtype=np.float32)
    if mode == "coarse_temporal":
        edges = np.linspace(window_ms[0], window_ms[1], 6)
    elif mode == "fine_temporal":
        n_bins = int(round((window_ms[1] - window_ms[0]) / bin_ms))
        if n_bins < 1 or not np.isclose(n_bins * bin_ms, window_ms[1] - window_ms[0]):
            raise ValueError("fine-temporal window must be an exact multiple of bin_ms")
        edges = window_ms[0] + np.arange(n_bins + 1) * bin_ms
    else:
        raise ValueError(f"unknown feature mode: {mode}")
    output = np.zeros((len(edges) - 1, len(electrodes)), dtype=np.float32)
    for electrode_index, electrode in enumerate(electrodes):
        values = selected_times[selected_channels == electrode]
        output[:, electrode_index] = np.histogram(values, bins=edges)[0]
    return output.reshape(-1)


def neural_feature_columns(recording_electrodes: Sequence[int], mode: str, window_ms: tuple[float, float] | None = None, bin_ms: float = 5.0) -> tuple[str, ...]:
    electrodes = tuple(int(value) for value in recording_electrodes)
    if mode == "spike_count":
        return tuple(f"electrode_{electrode}_count" for electrode in electrodes)
    if window_ms is None:
        raise ValueError("window_ms is required for temporal columns")
    bins = 5 if mode == "coarse_temporal" else int(round((window_ms[1] - window_ms[0]) / bin_ms))
    if mode not in {"coarse_temporal", "fine_temporal"}:
        raise ValueError(f"unknown feature mode: {mode}")
    return tuple(f"electrode_{electrode}_bin_{index}" for index in range(bins) for electrode in electrodes)


def assert_neural_only(columns: Sequence[str], recording_electrodes: Sequence[int]) -> None:
    allowed_electrodes = {str(int(value)) for value in recording_electrodes}
    pattern = re.compile(r"^electrode_(\d+)_(?:count|bin_\d+)$")
    for column in columns:
        match = pattern.match(str(column))
        if match is None or match.group(1) not in allowed_electrodes:
            raise ValueError("only explicitly allowlisted recording-electrode features may enter X")


def extract_phase_features(channels: Sequence[int], times_ms: Sequence[float], recording_electrodes: Sequence[int], *, baseline_ms: float = 500.0, stimulation_ms: float = 1000.0, mode: str = "spike_count", phase: str = "baseline") -> np.ndarray:
    """Extract diagnostics from baseline/during/post phases without mixing phases."""
    if phase == "baseline":
        window = (0.0, baseline_ms)
        offset = 0.0
    elif phase == "during":
        window = (0.0, stimulation_ms)
        offset = baseline_ms
    elif phase == "post":
        window = (0.0, 1000.0)
        offset = baseline_ms + stimulation_ms
    else:
        raise ValueError("phase must be baseline, during, or post")
    return extract_features(channels, np.asarray(times_ms, dtype=float) - offset, recording_electrodes, window, mode)


def save_feature_dataset(dataset: FeatureDataset, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "feature_mode": dataset.feature_mode, "window_ms": list(dataset.window_ms),
        "recording_electrodes": list(dataset.recording_electrodes),
        "has_controls": dataset.is_control is not None,
    }
    with path.open("wb") as handle:
        np.savez_compressed(handle, X=dataset.X, labels=dataset.labels, recording_ids=dataset.recording_ids, condition_groups=dataset.condition_groups, is_control=np.asarray(dataset.is_control if dataset.is_control is not None else np.zeros(len(dataset.labels), dtype=bool)), metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))


def load_feature_dataset(path: Path) -> FeatureDataset:
    with np.load(Path(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        return FeatureDataset(
            X=data["X"], labels=data["labels"], recording_ids=data["recording_ids"], condition_groups=data["condition_groups"],
            feature_mode=metadata["feature_mode"], window_ms=tuple(metadata["window_ms"]), recording_electrodes=tuple(metadata["recording_electrodes"]),
            is_control=data["is_control"] if metadata.get("has_controls", True) else None,
        )
