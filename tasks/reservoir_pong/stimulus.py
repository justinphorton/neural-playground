"""Spatial mapping, biphasic pulses, and blind-window spike features."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .physics import CellEntry


@dataclass(frozen=True)
class PulseEvent:
    onset_ms: float
    electrode_id: int
    amplitude_ua: float = 80.0
    phase_ms: float = 0.1
    col: int = 0
    row: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.onset_ms) or self.onset_ms < 0:
            raise ValueError("pulse onset must be finite and non-negative")
        if self.electrode_id < 0:
            raise ValueError("electrode_id must be non-negative")
        if not math.isfinite(self.amplitude_ua) or not 0 < self.amplitude_ua <= 1000:
            raise ValueError("pulse amplitude must be in (0, 1000] uA")
        if not math.isfinite(self.phase_ms) or self.phase_ms <= 0:
            raise ValueError("phase_ms must be positive and finite")

    @property
    def phase_amplitudes_ua(self) -> tuple[float, float]:
        return (self.amplitude_ua, -self.amplitude_ua)

    @property
    def phase_durations_ms(self) -> tuple[float, float]:
        return (self.phase_ms, self.phase_ms)


def select_grid(
    electrode_coordinates: np.ndarray,
    center_xy: tuple[float, float],
    side: int = 4,
) -> list[tuple[int, float, float]]:
    """Pick the closest complete square block of physical MEA electrodes."""
    coords = np.asarray(electrode_coordinates, dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 3 or side < 1:
        raise ValueError("electrode coordinates must be an Nx3+ array and side positive")
    xs, ys = np.unique(coords[:, 1]), np.unique(coords[:, 2])
    if len(xs) < side or len(ys) < side:
        raise ValueError(f"MEA has fewer than {side} x {side} distinct positions")
    cx, cy = center_xy
    near_x = np.sort(xs[np.argsort(np.abs(xs - cx), kind="stable")[:side]])
    near_y = np.sort(ys[np.argsort(np.abs(ys - cy), kind="stable")[:side]])
    at = {(float(row[1]), float(row[2])): int(row[0]) for row in coords}
    grid = []
    for y in near_y:
        for x in near_x:
            electrode = at.get((float(x), float(y)))
            if electrode is None:
                raise ValueError("the selected electrode positions are not a complete grid")
            grid.append((electrode, float(x), float(y)))
    return grid


def map_entries_to_electrodes(
    entries: list[CellEntry],
    grid: np.ndarray | list,
    cols: int,
    rows: int,
    amplitude_ua: float = 80.0,
    phase_ms: float = 0.1,
) -> list[PulseEvent]:
    """Map each court cell to its physical electrode, preserving event order."""
    positions = list(grid)
    if len(positions) != cols * rows:
        raise ValueError("grid must contain one electrode per court cell")
    result = []
    for entry in entries:
        if not (0 <= entry.col < cols and 0 <= entry.row < rows):
            raise ValueError("cell entry lies outside the court grid")
        position = positions[entry.row * cols + entry.col]
        channel = int(position if np.isscalar(position) else position[0])
        result.append(PulseEvent(entry.onset_ms, channel, amplitude_ua, phase_ms,
                                 entry.col, entry.row))
    return result


def make_waveform(
    events: list[PulseEvent],
    duration_ms: float,
    n_channels: int,
    dt_ms: float = 0.1,
    channel_ids: tuple[int, ...] | list[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Render non-overlapping rectangular, charge-balanced biphasic pulses."""
    if duration_ms <= 0 or n_channels <= 0 or dt_ms <= 0:
        raise ValueError("duration, channel count, and sample interval must be positive")
    ids = np.arange(n_channels) if channel_ids is None else np.asarray(channel_ids)
    if ids.shape != (n_channels,) or len(set(int(v) for v in ids)) != n_channels:
        raise ValueError("channel_ids must name each waveform column exactly once")
    index = {int(channel): i for i, channel in enumerate(ids)}
    n_steps = int(math.ceil(duration_ms / dt_ms - 1e-12))
    waveform = np.zeros((n_steps, n_channels), dtype=np.float32)
    for event in events:
        if event.electrode_id not in index:
            raise ValueError(f"electrode {event.electrode_id} is not present in this MEA")
        first = round(event.onset_ms / dt_ms)
        width = round(event.phase_ms / dt_ms)
        if width < 1 or not math.isclose(width * dt_ms, event.phase_ms, abs_tol=1e-8):
            raise ValueError("phase duration must be an integral number of stimulus samples")
        if not math.isclose(first * dt_ms, event.onset_ms, abs_tol=1e-8):
            raise ValueError("pulse onset must be quantized before waveform generation")
        if first < 0 or first + 2 * width > n_steps:
            raise ValueError("biphasic pulse extends outside its trial")
        ch = index[event.electrode_id]
        waveform[first:first + width, ch] += event.amplitude_ua
        waveform[first + width:first + 2 * width, ch] -= event.amplitude_ua
    return waveform


def extract_post_pulse_features(
    spike_channels: np.ndarray,
    spike_times_ms: np.ndarray,
    events: list[PulseEvent],
    recording_channels: tuple[int, ...] | list[int] | np.ndarray,
    blank_ms: float,
    window_ms: float,
) -> np.ndarray:
    """Count per-electrode spikes in [onset+blank, onset+blank+window)."""
    if window_ms <= 0:
        raise ValueError("window_ms must be positive")
    channels, times = apply_recording_guard(spike_channels, spike_times_ms, events, blank_ms)
    recording = np.asarray(recording_channels, dtype=np.int64)
    out = np.zeros((len(events), len(recording)), dtype=np.float32)
    for i, event in enumerate(events):
        start = event.onset_ms + blank_ms
        end = start + window_ms
        in_window = (times >= start) & (times < end)
        for j, channel in enumerate(recording):
            out[i, j] = np.count_nonzero(in_window & (channels == channel))
    return out


def apply_recording_guard(
    spike_channels: np.ndarray,
    spike_times_ms: np.ndarray,
    events: list[PulseEvent],
    blank_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove recorded events during each pulse and amplifier guard interval."""
    channels = np.asarray(spike_channels)
    times = np.asarray(spike_times_ms, dtype=float)
    if channels.ndim != 1 or times.ndim != 1 or channels.shape != times.shape:
        raise ValueError("spike channels and times must be matching vectors")
    if np.any(~np.isfinite(times)) or not np.isfinite(blank_ms) or blank_ms < 0:
        raise ValueError("spike times must be finite and blank_ms non-negative")
    keep = np.ones(len(times), dtype=bool)
    for event in events:
        keep &= ~((times >= event.onset_ms) & (times < event.onset_ms + blank_ms))
    return channels[keep], times[keep]
