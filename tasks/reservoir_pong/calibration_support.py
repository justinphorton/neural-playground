"""Shared raw-recording storage and held-out classification for calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .storage import save_npz_atomic


class CalibrationRawRecorder:
    """Keep flattened native spike recordings with trial boundary offsets."""

    def __init__(self) -> None:
        self.kind, self.seed, self.electrode = [], [], []
        self.previous, self.gap, self.amplitude, self.phase = [], [], [], []
        self.spike_ids, self.spike_times, self.spike_offsets = [], [], [0]
        self.channel_ids, self.channel_times, self.channel_offsets = [], [], [0]

    def add(self, result, kind: str, seed: int, electrode: int = -1,
            previous: int = -1, gap_ms: float = 0.0,
            amplitude_ua: float = 0.0, phase_ms: float = 0.0) -> None:
        self.kind.append(kind)
        self.seed.append(int(seed))
        self.electrode.append(int(electrode))
        self.previous.append(int(previous))
        self.gap.append(float(gap_ms))
        self.amplitude.append(float(amplitude_ua))
        self.phase.append(float(phase_ms))
        self.spike_ids.append(np.asarray(result.spike_gids, dtype=np.int64))
        self.spike_times.append(np.asarray(result.spike_times_ms, dtype=np.float64))
        self.spike_offsets.append(self.spike_offsets[-1] + len(result.spike_gids))
        self.channel_ids.append(np.asarray(result.spike_channels, dtype=np.int32))
        self.channel_times.append(np.asarray(result.spike_channel_times_ms, dtype=np.float64))
        self.channel_offsets.append(self.channel_offsets[-1] + len(result.spike_channels))

    def save(self, path: Path) -> None:
        save_npz_atomic(
            Path(path),
            kind=np.asarray(self.kind, dtype="U16"),
            seed=np.asarray(self.seed, dtype=np.int64),
            electrode=np.asarray(self.electrode, dtype=np.int32),
            previous_electrode=np.asarray(self.previous, dtype=np.int32),
            gap_ms=np.asarray(self.gap, dtype=np.float32),
            amplitude_ua=np.asarray(self.amplitude, dtype=np.float32),
            phase_ms=np.asarray(self.phase, dtype=np.float32),
            spike_offsets=np.asarray(self.spike_offsets, dtype=np.int64),
            spike_gids=np.concatenate(self.spike_ids) if self.spike_ids else np.zeros(0, dtype=np.int64),
            spike_times_ms=np.concatenate(self.spike_times) if self.spike_times else np.zeros(0),
            channel_offsets=np.asarray(self.channel_offsets, dtype=np.int64),
            spike_channels=np.concatenate(self.channel_ids) if self.channel_ids else np.zeros(0, dtype=np.int32),
            spike_channel_times_ms=(np.concatenate(self.channel_times)
                                    if self.channel_times else np.zeros(0)),
        )

def holdout_accuracy(features: np.ndarray, labels: np.ndarray, seed: int) -> tuple[float, np.ndarray]:
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    train, test = [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        indices = rng.permutation(indices)
        holdout = max(1, int(round(0.30 * len(indices))))
        test.extend(indices[:holdout].tolist())
        train.extend(indices[holdout:].tolist())
    model = make_pipeline(StandardScaler(with_mean=False), RidgeClassifier(alpha=1.0))
    model.fit(features[train], labels[train])
    predicted = model.predict(features[test])
    truth = labels[test]
    return float(np.mean(predicted == truth)), (predicted == truth)
