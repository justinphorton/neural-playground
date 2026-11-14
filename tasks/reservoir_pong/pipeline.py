"""Dataset assembly, regularized model selection, and versioned readout files."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import vstack

from .readouts import build_prefix_design, select_ridge, trajectory_sample_weights
from .records import TrialRecord
from .stimulus import PulseEvent


@dataclass(frozen=True)
class Examples:
    neural: np.ndarray
    raw: object
    targets: np.ndarray
    groups: np.ndarray
    sample_weights: np.ndarray


@dataclass(frozen=True)
class SelectedReadouts:
    neural: object
    raw: object
    neural_alpha: float
    raw_alpha: float
    neural_validation_scores: dict[float, float]
    raw_validation_scores: dict[float, float]


def build_examples(
    records: list[TrialRecord],
    electrode_grid_ids: tuple[int, ...],
    dt_ms: float,
    warmup_ms: float,
    fall_ms: float,
    split: str | None = None,
    pulse_cutoff_policy: str = "strictly_before",
) -> Examples:
    neural_rows, raw_rows, targets, groups = [], [], [], []
    for record in records:
        if split is not None and record.metadata.get("split") != split:
            continue
        commit = float(record.metadata["commit_ms"])
        keep = np.flatnonzero(record.availability_ms <= commit + 1e-9)
        if len(keep) < 1:
            continue
        events = [PulseEvent(float(t + warmup_ms), int(ch), col=int(cell[0]), row=int(cell[1]))
                  for t, ch, cell in zip(record.pulse_times_ms,
                                         record.pulse_electrode_ids, record.pulse_cells)]
        decisions = record.availability_ms[keep] + warmup_ms
        raw = build_prefix_design(events, decisions, electrode_grid_ids,
                                  warmup_ms + fall_ms, dt_ms,
                                  cutoff_policy=pulse_cutoff_policy)
        neural_rows.append(record.feature_counts[keep])
        raw_rows.append(raw)
        targets.extend([record.landing_x] * len(keep))
        groups.extend([record.trajectory_id] * len(keep))
    if not neural_rows:
        raise ValueError(f"no usable {split or 'all'} examples were found")
    neural = np.vstack(neural_rows).astype(np.float32)
    raw = vstack(raw_rows, format="csr")
    group_array = np.asarray(groups)
    return Examples(neural, raw, np.asarray(targets, dtype=float), group_array,
                    trajectory_sample_weights(group_array))


def train_readouts(
    training: Examples,
    validation: Examples,
    alpha_grid: tuple[float, ...],
) -> SelectedReadouts:
    neural, neural_alpha, neural_scores = select_ridge(
        training.neural, training.targets, training.sample_weights,
        validation.neural, validation.targets, validation.sample_weights, alpha_grid,
    )
    raw, raw_alpha, raw_scores = select_ridge(
        training.raw, training.targets, training.sample_weights,
        validation.raw, validation.targets, validation.sample_weights, alpha_grid,
    )
    return SelectedReadouts(neural, raw, neural_alpha, raw_alpha,
                            neural_scores, raw_scores)


def save_readouts(
    path: Path,
    neural,
    raw,
    config_hash: str,
    reservoir_hash: str,
    feature_count: int,
    grid_ids: tuple[int, ...],
    **metadata,
) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "neural": neural,
        "raw": raw,
        "config_hash": config_hash,
        "reservoir_hash": reservoir_hash,
        "feature_count": int(feature_count),
        "grid_ids": tuple(int(x) for x in grid_ids),
        "metadata": metadata,
    }
    joblib.dump(payload, path, compress=3)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".json").write_text(json.dumps({
        "config_hash": config_hash,
        "reservoir_hash": reservoir_hash,
        "readout_id": digest,
        "feature_count": int(feature_count),
        "grid_ids": list(grid_ids),
        "metadata": metadata,
    }, indent=2, sort_keys=True, default=str) + "\n")
    return digest


def load_readouts(path: Path, config_hash: str, reservoir_hash: str) -> dict:
    value = joblib.load(Path(path))
    if value.get("config_hash") != config_hash:
        raise ValueError("readout configuration hash does not match the active run")
    if value.get("reservoir_hash") != reservoir_hash:
        raise ValueError("readout reservoir hash does not match the active reservoir")
    manifest_path = Path(path).with_suffix(Path(path).suffix + ".json")
    if not manifest_path.is_file():
        raise ValueError("readout artifact manifest is missing")
    manifest = json.loads(manifest_path.read_text())
    actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if manifest.get("readout_id") != actual:
        raise ValueError("readout artifact checksum does not match its manifest")
    if manifest.get("config_hash") != config_hash or manifest.get("reservoir_hash") != reservoir_hash:
        raise ValueError("readout manifest identity does not match the active experiment")
    value["readout_id"] = actual
    return value


def locked_prediction(readout, features: np.ndarray) -> np.ndarray:
    before = np.asarray(readout.named_steps["ridge"].coef_).copy()
    predictions = np.asarray(readout.predict(features), dtype=float).reshape(-1)
    after = np.asarray(readout.named_steps["ridge"].coef_)
    if not np.array_equal(before, after):
        raise RuntimeError("the supposedly frozen readout changed during prediction")
    return predictions
