"""Versioned configuration and output layout for the fixed-reservoir experiment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    notes: tuple[str, ...] = ("C4", "E4")
    baseline_ms: int = 500
    stimulation_ms: int = 1000
    post_stimulus_ms: int = 1000
    sample_rate_hz: int = 10_000
    default_peak_current_ua: float = 25.0
    screening_amplitudes_ua: tuple[float, ...] = (1.5, 3.125, 6.25, 12.5, 25.0)
    fade_out_ms: float = 15.0
    workers: int = 8
    full_dataset_repeats: int = 3
    structural_reservoir_seed: int = 123
    reservoir_cells: int = 300
    stimulation_electrode_index: int | None = None
    recording_electrodes: tuple[int, ...] = ()
    feature_mode: str = "fine_temporal"
    primary_window_ms: float = 50.0
    primary_window_start_ms: float = 5.0
    feature_bin_ms: float = 5.0
    primary_feature_windows_ms: tuple[tuple[float, float], ...] = field(
        default_factory=lambda: tuple((float(start), float(start + 50)) for start in range(5, 955, 50))
    )
    diagnostic_feature_windows_ms: tuple[tuple[float, float], ...] = field(default_factory=lambda: (
        (0.0, 5.0), (5.0, 25.0), (25.0, 50.0), (50.0, 100.0),
        (100.0, 200.0), (200.0, 350.0), (350.0, 500.0),
        (500.0, 750.0), (750.0, 1000.0),
    ))
    classifier: dict[str, Any] = field(default_factory=lambda: {
        "scaler": "StandardScaler",
        "estimator": "LogisticRegression",
        "penalty": "l2",
        "C": 1.0,
        "class_weight": "balanced",
        "max_iter": 2000,
        "random_state": 0,
    })
    reservoir_artifact: str = "reservoir"
    software_version: str = "fixed-reservoir-c4-e4-v1"

    def __post_init__(self) -> None:
        if tuple(self.notes) != ("C4", "E4"):
            raise ValueError("the fixed experiment supports exactly C4 and E4")
        if self.baseline_ms < 0 or self.stimulation_ms <= 0 or self.post_stimulus_ms <= 0:
            raise ValueError("phase durations are invalid")
        if self.sample_rate_hz != 10_000:
            raise ValueError("the experiment uses a fixed 10 kHz stimulus rate")
        if self.fade_out_ms <= 0 or self.fade_out_ms >= self.stimulation_ms:
            raise ValueError("fade_out_ms must be inside the stimulation phase")
        if self.workers < 1 or self.full_dataset_repeats < 1:
            raise ValueError("workers and repeats must be positive")
        if any(float(value) <= 0 or float(value) > 25 for value in self.screening_amplitudes_ua):
            raise ValueError("screening amplitudes must be in (0, 25] uA")
        if self.default_peak_current_ua <= 0 or self.default_peak_current_ua > 25:
            raise ValueError("default_peak_current_ua must be in (0, 25] uA")
        if any(start < self.primary_window_start_ms or end > self.post_stimulus_ms for start, end in self.primary_feature_windows_ms):
            raise ValueError("primary feature windows must be post-stimulus and inside the recorded memory phase")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["notes"] = list(self.notes)
        payload["screening_amplitudes_ua"] = list(self.screening_amplitudes_ua)
        payload["recording_electrodes"] = list(self.recording_electrodes)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        values = dict(payload)
        for key in ("notes", "screening_amplitudes_ua", "recording_electrodes"):
            if key in values:
                values[key] = tuple(values[key])
        for key in ("primary_feature_windows_ms", "diagnostic_feature_windows_ms"):
            if key in values:
                values[key] = tuple(tuple(float(bound) for bound in window) for window in values[key])
        return cls(**values)

    @property
    def config_hash(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def write(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**self.to_dict(), "config_hash": self.config_hash}, indent=2) + "\n")

    @classmethod
    def read(cls, path: Path) -> "ExperimentConfig":
        payload = json.loads(Path(path).read_text())
        payload.pop("config_hash", None)
        return cls.from_dict(payload)


def load_config(path: Path) -> ExperimentConfig:
    return ExperimentConfig.read(path)


def save_config(config: ExperimentConfig, path: Path) -> None:
    config.write(path)


EXPERIMENT_SUBDIRECTORIES = (
    "reservoir", "raw_trials", "features", "models", "metrics", "plots", "logs",
)


def create_experiment_directory(
    root: Path,
    experiment_name: str,
    config: ExperimentConfig,
    timestamp: str | None = None,
) -> Path:
    """Create an immutable experiment directory; never overwrite an existing run."""
    root = Path(root)
    timestamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in experiment_name).strip("-")
    path = root / f"{timestamp}_{safe_name}_{config.config_hash[:12]}"
    path.mkdir(parents=True, exist_ok=False)
    for child in EXPERIMENT_SUBDIRECTORIES:
        (path / child).mkdir()
    config.write(path / "config.json")
    return path
