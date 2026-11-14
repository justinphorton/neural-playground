"""Matched fresh-test controls that do not enter readout fitting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .evaluation import paired_trajectory_bootstrap
from .physics import BallSpec
from .pipeline import load_readouts, locked_prediction
from .readouts import paddle_trace
from .records import ExperimentConfig
from .reservoir import ReservoirRunner, build_reservoir
from .stimulus import PulseEvent, extract_post_pulse_features
from .storage import save_json_atomic, sha256_file
from .trial_store import load_records


def unstimulated_window_features(
    spike_channels: np.ndarray,
    spike_times_ms: np.ndarray,
    record,
    recording_channels: tuple[int, ...],
    config: ExperimentConfig,
) -> np.ndarray:
    """Count no-input spikes in the same guarded windows as the matched trial."""
    marker_events = [
        PulseEvent(
            float(time + config.warmup_ms), int(channel), config.amplitude_ua,
            config.phase_ms, int(cell[0]), int(cell[1]),
        )
        for time, channel, cell in zip(
            record.pulse_times_ms, record.pulse_electrode_ids, record.pulse_cells,
        )
    ]
    all_features = extract_post_pulse_features(
        spike_channels, spike_times_ms, marker_events, recording_channels,
        config.feature_blank_ms, config.feature_window_ms,
    )
    available = (np.asarray(record.pulse_times_ms, dtype=float)
                 + config.feature_blank_ms + config.feature_window_ms)
    precommit = available <= float(record.metadata["commit_ms"]) + 1e-9
    selected = all_features[precommit]
    if len(selected) != len(record.availability_ms):
        raise ValueError("no-stimulation feature windows do not match saved decision times")
    return selected


def _outcome_summary(rows: list[dict], catch_key: str, error_key: str,
                     reference_key: str, seed: int) -> dict:
    groups = np.asarray([row["trajectory_id"] for row in rows])
    outcomes = np.asarray([row[catch_key] for row in rows], dtype=float)
    reference = np.asarray([row[reference_key] for row in rows], dtype=float)
    errors = np.asarray([row[error_key] for row in rows], dtype=float)
    gains = paired_trajectory_bootstrap(
        outcomes, reference, groups, seed=seed, repeats=2000,
    )
    return {
        "records": len(rows),
        "trajectories": int(len(np.unique(groups))),
        "catch_rate": float(outcomes.mean()),
        "landing_mae": float(errors.mean()),
        "catch_gain_vs_reservoir_ridge": gains[0],
        "catch_gain_vs_reservoir_ridge_ci_95": [gains[1], gains[2]],
    }


def run_fresh_test_controls(
    stage_dir: Path,
    config: ExperimentConfig,
    records: list,
    progress=None,
) -> dict:
    """Score zero-feature and actual no-stimulation trials on matched test balls."""
    stage_dir = Path(stage_dir)
    rows: list[dict] = []
    no_stim_summaries = {}
    zero_summaries = {}
    model_ids, reservoir_hashes = {}, {}
    control_dir = stage_dir / "controls"
    control_dir.mkdir(parents=True, exist_ok=True)
    trial_control_dir = control_dir / "trials"
    trial_control_dir.mkdir(parents=True, exist_ok=True)
    by_slot = {
        slot: [record for record in records
               if int(record.metadata["reservoir_slot"]) == slot]
        for slot in range(config.reservoir_count)
    }
    total = sum(len(items) for items in by_slot.values())
    completed = 0
    for slot in range(config.reservoir_count):
        reservoir = build_reservoir(config, slot)
        try:
            manifest = json.loads(
                (stage_dir / "reservoirs" / f"slot{slot}.json").read_text(),
            )
            if reservoir.reservoir_hash != manifest["reservoir_hash"]:
                raise ValueError(f"subreservoir {slot} changed before control scoring")
            readout_path = stage_dir / "models" / f"readouts_slot{slot}.joblib"
            readouts = load_readouts(readout_path, config.config_hash,
                                     reservoir.reservoir_hash)
            readout_id = readouts["readout_id"]
            model_ids[str(slot)] = readout_id
            reservoir_hashes[str(slot)] = reservoir.reservoir_hash
            runner = ReservoirRunner(reservoir, config)
            for record in by_slot[slot]:
                checkpoint_path = trial_control_dir / (
                    f"{record.trajectory_id}_slot{slot}_r{int(record.metadata['repeat']):02d}.json"
                )
                identity = {
                    "config_hash": config.config_hash,
                    "reservoir_hash": reservoir.reservoir_hash,
                    "readout_id": readout_id,
                    "noise_seed": int(record.noise_seed),
                    "trajectory_id": record.trajectory_id,
                    "reservoir_slot": slot,
                    "repeat": int(record.metadata["repeat"]),
                }
                if checkpoint_path.is_file():
                    saved = json.loads(checkpoint_path.read_text())
                    if saved.get("checkpoint_identity") != identity:
                        raise ValueError(
                            f"no-stimulation checkpoint is incompatible: {checkpoint_path}"
                        )
                    rows.append(saved["result"])
                    completed += 1
                    if progress:
                        progress(completed, total, record.trajectory_id)
                    continue
                ball = BallSpec(
                    record.trajectory_id, float(record.metadata["x0"]),
                    float(record.metadata["vx"]), float(record.metadata["fall_ms"]),
                    config.commit_fraction, "test",
                )
                zeros = np.zeros_like(record.feature_counts, dtype=np.float32)
                zero_predictions = locked_prediction(readouts["neural"], zeros)
                _, zero_caught = paddle_trace(
                    ball, record.availability_ms, zero_predictions,
                    config.paddle_width, config.paddle_speed_per_fall,
                    config.stimulus_dt_ms,
                )

                duration_ms = (config.warmup_ms + ball.fall_ms
                               + config.feature_blank_ms + config.feature_window_ms + 0.1)
                no_stim = runner.simulate([], duration_ms, int(record.noise_seed))
                no_stim_features = unstimulated_window_features(
                    no_stim.spike_channels, no_stim.spike_channel_times_ms,
                    record, runner.channel_ids, config,
                )
                no_stim_predictions = locked_prediction(
                    readouts["neural"], no_stim_features,
                )
                _, no_stim_caught = paddle_trace(
                    ball, record.availability_ms, no_stim_predictions,
                    config.paddle_width, config.paddle_speed_per_fall,
                    config.stimulus_dt_ms,
                )
                row = {
                    "trajectory_id": record.trajectory_id,
                    "reservoir_slot": slot,
                    "repeat": int(record.metadata["repeat"]),
                    "noise_seed": int(record.noise_seed),
                    "no_stimulation_network_spikes": int(len(no_stim.spike_times_ms)),
                    "reservoir_caught": bool(record.caught),
                    "reservoir_landing_mae": abs(float(record.neural_predictions[-1])
                                                  - float(record.landing_x)),
                    "ridge_zero_spike_caught": bool(zero_caught),
                    "ridge_zero_spike_landing_mae": abs(float(zero_predictions[-1])
                                                        - float(record.landing_x)),
                    "no_stimulation_caught": bool(no_stim_caught),
                    "no_stimulation_landing_mae": abs(float(no_stim_predictions[-1])
                                                      - float(record.landing_x)),
                }
                save_json_atomic(checkpoint_path, {
                    "checkpoint_identity": identity,
                    "result": row,
                })
                rows.append(row)
                completed += 1
                if progress:
                    progress(completed, total, record.trajectory_id)
        finally:
            reservoir.close()

    zero_path = control_dir / "ridge_zero_spike.jsonl"
    no_stim_path = control_dir / "actual_no_stimulation.jsonl"
    zero_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                                  for row in rows))
    no_stim_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                                     for row in rows))
    for slot in range(config.reservoir_count):
        slot_rows = [row for row in rows if row["reservoir_slot"] == slot]
        zero_summaries[str(slot)] = _outcome_summary(
            slot_rows, "ridge_zero_spike_caught", "ridge_zero_spike_landing_mae",
            "reservoir_caught", 641 + slot,
        )
        no_stim_summaries[str(slot)] = _outcome_summary(
            slot_rows, "no_stimulation_caught", "no_stimulation_landing_mae",
            "reservoir_caught", 651 + slot,
        )
    zero_overall = _outcome_summary(
        rows, "ridge_zero_spike_caught", "ridge_zero_spike_landing_mae",
        "reservoir_caught", 661,
    )
    no_stim_overall = _outcome_summary(
        rows, "no_stimulation_caught", "no_stimulation_landing_mae",
        "reservoir_caught", 671,
    )
    no_stim_overall.update({
        "mean_unstimulated_network_spikes": float(np.mean([
            row["no_stimulation_network_spikes"] for row in rows
        ])),
        "p95_unstimulated_network_spikes": float(np.quantile([
            row["no_stimulation_network_spikes"] for row in rows
        ], 0.95)),
    })
    summary = {
        "status": "complete",
        "records": len(rows),
        "trajectories": len({row["trajectory_id"] for row in rows}),
        "config_hash": config.config_hash,
        "reservoir_hashes": reservoir_hashes,
        "readout_ids": model_ids,
        "reservoir_ridge_zero_spike": zero_overall,
        "actual_no_stimulation": no_stim_overall,
        "zero_spike_subreservoirs": zero_summaries,
        "no_stimulation_subreservoirs": no_stim_summaries,
        "record_hashes": {
            "ridge_zero_spike": sha256_file(zero_path),
            "actual_no_stimulation": sha256_file(no_stim_path),
        },
        "interpretation": "Controls replay the same test ball, decision clock, readout, and paddle physics; no-stimulation trials deliver zero pulses and use the original trial's noise seed.",
    }
    (control_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    return summary
