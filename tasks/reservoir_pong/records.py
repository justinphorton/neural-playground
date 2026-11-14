"""Versioned run configuration, provenance, and non-pickle trial artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .storage import save_npz_atomic


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 123
    n_cells: int = 600
    grid_cols: int = 4
    grid_rows: int = 4
    warmup_ms: float = 20.0
    amplitude_ua: float = 80.0
    phase_ms: float = 0.1
    stimulus_dt_ms: float = 0.1
    feature_blank_ms: float = 0.3
    feature_window_ms: float = 0.5
    guard_ms: float = 0.1
    minimum_gap_ms: float = 1.0
    maximum_gap_ms: float = 12.0
    fall_ms: float = 24.0
    commit_fraction: float = 0.5
    paddle_width: float = 0.12
    paddle_speed_per_fall: float = 0.8
    residual_ranges: tuple[tuple[float, float], ...] = (
        (0.10, 0.25), (0.40, 0.80), (1.00, 1.50),
    )
    selected_residual_range: tuple[float, float] = (0.40, 0.80)
    regularization_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    calibration_repeats: int = 20
    pilot_counts: tuple[int, int, int] = (120, 40, 80)
    final_counts: tuple[int, int, int] = (600, 200, 400)
    noise_repeats: int = 2
    reservoir_count: int = 3
    bootstrap_repeats: int = 2000
    software_version: str = "reservoir-pong-v1.3"

    def __post_init__(self) -> None:
        if self.n_cells < 32 or self.grid_cols < 1 or self.grid_rows < 1:
            raise ValueError("reservoir and court grid sizes are invalid")
        if self.warmup_ms < 0 or self.amplitude_ua <= 0 or self.phase_ms <= 0:
            raise ValueError("pulse and warm-up settings are invalid")
        if self.stimulus_dt_ms <= 0 or self.feature_blank_ms < 0 or self.feature_window_ms <= 0:
            raise ValueError("sampling and feature windows are invalid")
        if self.guard_ms < 0 or self.minimum_gap_ms <= 0:
            raise ValueError("pulse-gap settings are invalid")
        if self.maximum_gap_ms < self.minimum_gap_ms:
            raise ValueError("maximum_gap_ms is below minimum_gap_ms")
        if (len(self.selected_residual_range) != 2
                or self.selected_residual_range[0] <= 0
                or self.selected_residual_range[1] < self.selected_residual_range[0]):
            raise ValueError("selected_residual_range must be a positive ordered pair")
        if self.fall_ms <= 0 or not 0 < self.commit_fraction < 1:
            raise ValueError("fall and commitment settings are invalid")
        if not 0 < self.paddle_width <= 1 or not 0 < self.paddle_speed_per_fall <= 1:
            raise ValueError("paddle settings are invalid")
        if not self.regularization_grid or any(a <= 0 for a in self.regularization_grid):
            raise ValueError("regularization values must be positive")
        if self.calibration_repeats < 2 or self.noise_repeats < 1:
            raise ValueError("calibration and noise repeats must be positive")
        if self.reservoir_count < 1:
            raise ValueError("reservoir_count must be positive")
        if self.bootstrap_repeats < 1:
            raise ValueError("bootstrap_repeats must be positive")

    @property
    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_dict(self) -> dict:
        return asdict(self) | {"config_hash": self.config_hash}

    @classmethod
    def from_dict(cls, value: dict) -> "ExperimentConfig":
        fields = dict(value)
        fields.pop("config_hash", None)
        for key in ("residual_ranges",):
            if key in fields:
                fields[key] = tuple(tuple(float(v) for v in pair) for pair in fields[key])
        if "selected_residual_range" in fields:
            fields["selected_residual_range"] = tuple(float(v) for v in fields["selected_residual_range"])
        for key in ("regularization_grid", "pilot_counts", "final_counts"):
            if key in fields:
                fields[key] = tuple(fields[key])
        return cls(**fields)


@dataclass(frozen=True)
class BallSpec:
    """Declared in records and aliased in physics to keep one public shape."""
    trajectory_id: str
    x0: float
    vx: float
    fall_ms: float
    commit_fraction: float = 0.5
    split: str = "train"

    def __post_init__(self) -> None:
        import math
        if not self.trajectory_id:
            raise ValueError("trajectory_id cannot be empty")
        if not math.isfinite(self.x0) or not 0 <= self.x0 <= 1:
            raise ValueError("x0 must be finite and inside [0, 1]")
        if not math.isfinite(self.vx) or not math.isfinite(self.fall_ms) or self.fall_ms <= 0:
            raise ValueError("vx and fall_ms must be finite and fall_ms positive")
        if not 0 < self.commit_fraction < 1:
            raise ValueError("commit_fraction must be inside (0, 1)")
        if self.split not in {"train", "validation", "test", "pilot", "live"}:
            raise ValueError("split must be train, validation, test, pilot, or live")

    @property
    def commit_ms(self) -> float:
        return self.fall_ms * self.commit_fraction


@dataclass(frozen=True)
class ExperimentManifest:
    config_hash: str
    backend: str
    livn_version: str
    reservoir_hash: str
    selected_gids: tuple[int, ...]
    model_hash: str
    python_version: str = field(default_factory=platform.python_version)
    numpy_version: str = field(default_factory=lambda: importlib.metadata.version("numpy"))
    scipy_version: str = field(default_factory=lambda: importlib.metadata.version("scipy"))
    scikit_learn_version: str = field(default_factory=lambda: importlib.metadata.version("scikit-learn"))
    software_version: str = "reservoir-pong-v1.3"
    plasticity_enabled: bool = False


@dataclass
class TrialRecord:
    trajectory_id: str
    reservoir_hash: str
    config_hash: str
    noise_seed: int
    pulse_times_ms: np.ndarray
    pulse_electrode_ids: np.ndarray
    pulse_cells: np.ndarray
    spike_times_ms: np.ndarray
    spike_gids: np.ndarray
    spike_channels: np.ndarray
    spike_channel_times_ms: np.ndarray
    feature_counts: np.ndarray
    availability_ms: np.ndarray
    neural_predictions: np.ndarray
    raw_predictions: np.ndarray
    paddle_positions: np.ndarray
    paddle_times_ms: np.ndarray
    raw_paddle_positions: np.ndarray
    follow_paddle_positions: np.ndarray
    landing_x: float
    caught: bool
    raw_caught: bool = False
    follow_caught: bool = False
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.pulse_times_ms = np.asarray(self.pulse_times_ms, dtype=np.float64)
        self.pulse_electrode_ids = np.asarray(self.pulse_electrode_ids, dtype=np.int32)
        self.pulse_cells = np.asarray(self.pulse_cells, dtype=np.int16).reshape((-1, 2))
        self.spike_times_ms = np.asarray(self.spike_times_ms, dtype=np.float64)
        self.spike_gids = np.asarray(self.spike_gids, dtype=np.int64)
        self.spike_channels = np.asarray(self.spike_channels, dtype=np.int32)
        self.spike_channel_times_ms = np.asarray(self.spike_channel_times_ms, dtype=np.float64)
        self.feature_counts = np.asarray(self.feature_counts, dtype=np.float32)
        self.availability_ms = np.asarray(self.availability_ms, dtype=np.float64)
        self.neural_predictions = np.asarray(self.neural_predictions, dtype=np.float64)
        self.raw_predictions = np.asarray(self.raw_predictions, dtype=np.float64)
        self.paddle_positions = np.asarray(self.paddle_positions, dtype=np.float64)
        self.paddle_times_ms = np.asarray(self.paddle_times_ms, dtype=np.float64)
        self.raw_paddle_positions = np.asarray(self.raw_paddle_positions, dtype=np.float64)
        self.follow_paddle_positions = np.asarray(self.follow_paddle_positions, dtype=np.float64)
        if self.spike_times_ms.shape != self.spike_gids.shape:
            raise ValueError("neuron spike times and neuron IDs must match")
        if self.spike_channels.shape != self.spike_channel_times_ms.shape:
            raise ValueError("recording channel IDs and channel spike times must match")
        if not (len(self.pulse_times_ms) == len(self.pulse_electrode_ids) == len(self.pulse_cells)):
            raise ValueError("pulse times, electrode IDs, and court cells must match")
        if self.feature_counts.ndim != 2 or self.feature_counts.shape[0] != len(self.availability_ms):
            raise ValueError("feature rows must align with prediction availability times")
        if not (len(self.availability_ms) == len(self.neural_predictions) == len(self.raw_predictions)):
            raise ValueError("readout predictions must align with availability times")
        if len(self.paddle_positions) != len(self.paddle_times_ms):
            raise ValueError("neural paddle trace times and positions must match")
        if len(self.raw_paddle_positions) not in (0, len(self.paddle_times_ms)):
            raise ValueError("raw baseline paddle trace must align with neural trace")
        if len(self.follow_paddle_positions) not in (0, len(self.paddle_times_ms)):
            raise ValueError("follow-the-ball paddle trace must align with neural trace")

    def save(self, path: Path) -> None:
        save_trial(self, path)


def create_run_directory(
    root: Path,
    run_name: str,
    config: ExperimentConfig,
    run_id: str | None = None,
) -> Path:
    root = Path(root)
    stamp = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "-", run_name).strip("-") or "pong"
    path = root / f"{stamp}_{safe_name}_{config.config_hash[:12]}"
    path.mkdir(parents=True, exist_ok=False)
    (path / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    for name in ("reservoirs", "calibration", "raw_trials", "models", "metrics", "logs"):
        (path / name).mkdir()
    return path


_ARRAY_NAMES = (
    "pulse_times_ms", "pulse_electrode_ids", "pulse_cells", "spike_times_ms",
    "spike_gids", "spike_channels", "spike_channel_times_ms", "feature_counts", "availability_ms",
    "neural_predictions", "raw_predictions", "paddle_positions", "paddle_times_ms",
    "raw_paddle_positions", "follow_paddle_positions",
)


def save_trial(record: TrialRecord, path: Path) -> None:
    save_npz_atomic(
        path,
        **{name: getattr(record, name) for name in _ARRAY_NAMES},
        trajectory_id=np.asarray(record.trajectory_id),
        reservoir_hash=np.asarray(record.reservoir_hash),
        config_hash=np.asarray(record.config_hash),
        noise_seed=np.asarray(record.noise_seed, dtype=np.int64),
        landing_x=np.asarray(record.landing_x),
        caught=np.asarray(record.caught),
        raw_caught=np.asarray(record.raw_caught),
        follow_caught=np.asarray(record.follow_caught),
        metadata_json=np.asarray(json.dumps(record.metadata, sort_keys=True)),
    )


def load_trial(path: Path) -> TrialRecord:
    with np.load(Path(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        arrays = {
            name: (data[name].copy() if name in data.files else np.zeros(0))
            for name in _ARRAY_NAMES
        }
        return TrialRecord(
            trajectory_id=str(data["trajectory_id"].item()),
            reservoir_hash=str(data["reservoir_hash"].item()),
            config_hash=str(data["config_hash"].item()),
            noise_seed=int(data["noise_seed"].item()),
            landing_x=float(data["landing_x"].item()), caught=bool(data["caught"].item()),
            raw_caught=bool(data["raw_caught"].item()), follow_caught=bool(data["follow_caught"].item()),
            metadata=metadata,
            **arrays,
        )


def trial_checkpoint_matches(path: Path, config_hash: str, reservoir_hash: str,
                             readout_id: str | None = None) -> bool:
    try:
        with np.load(Path(path), allow_pickle=False) as data:
            matches = (str(data["config_hash"].item()) == config_hash
                       and str(data["reservoir_hash"].item()) == reservoir_hash)
            if readout_id is not None:
                metadata = json.loads(str(data["metadata_json"].item()))
                matches = matches and metadata.get("readout_id") == readout_id
            return matches
    except (OSError, ValueError, KeyError):
        return False


def stable_seed(*parts: object, base_seed: int = 123) -> int:
    digest = hashlib.sha256((str(base_seed) + ":" + ":".join(map(str, parts))).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def structural_hash(env, selected_gids: np.ndarray, electrode_grid) -> str:
    digest = hashlib.sha256()

    def update(value) -> None:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())

    update(np.asarray(selected_gids, dtype=np.int64))
    update(np.asarray(electrode_grid, dtype=float))
    for name in ("pre_gid", "syn_row", "post_pop", "pre_pop", "dest_sectype",
                 "mech_id", "delay", "weight", "wslot"):
        conn = getattr(env, "conn", None)
        if conn is not None and hasattr(conn, name):
            update(getattr(conn, name))
    update(np.asarray(env.io.electrode_coordinates))
    digest.update(type(env.model).__module__.encode())
    digest.update(type(env.model).__qualname__.encode())
    digest.update(importlib.metadata.version("livn").encode())
    digest.update(type(env).__module__.encode())
    return digest.hexdigest()
