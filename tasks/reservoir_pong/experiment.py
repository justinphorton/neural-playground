"""Pong trial generation, simulation, and causal prediction frames."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .physics import (BallSpec, CellEntry, landing_position, pulse_events,
                      reflected_position, validate_pulse_spacing)
from .readouts import (build_prefix_design, follow_ball_targets, paddle_trace,
                       paddle_trace_times, pulse_policy_version, seeded_ball)
from .pipeline import locked_prediction
from .records import ExperimentConfig, TrialRecord, stable_seed
from .reservoir import ReservoirRunner, SimulationResult
from .stimulus import PulseEvent, extract_post_pulse_features, map_entries_to_electrodes


@dataclass(frozen=True)
class TrialTrace:
    ball: BallSpec
    events: tuple[PulseEvent, ...]
    simulation: SimulationResult
    feature_counts: np.ndarray
    availability_ms: np.ndarray
    electrode_grid: tuple[tuple[int, float, float], ...]
    recording_channels: tuple[int, ...]
    config_hash: str
    reservoir_hash: str


def _has_wall_bounce_after_commit(ball: BallSpec) -> bool:
    end = ball.x0 + ball.vx
    lo, hi = sorted((ball.x0, end))
    for wall in range(int(np.floor(lo)) - 1, int(np.ceil(hi)) + 2):
        fraction = (float(wall) - ball.x0) / ball.vx if ball.vx else np.inf
        if ball.commit_fraction + 1e-10 < fraction < 1.0 - 1e-10:
            return True
    return False


def _accepted_candidate(
    ball: BallSpec,
    config: ExperimentConfig,
    require_post_commit_bounce: bool,
    rejection_counts: dict[str, int] | None = None,
) -> list[CellEntry] | None:
    entries = pulse_events(ball, config.grid_cols, config.grid_rows, config.stimulus_dt_ms)
    try:
        validate_pulse_spacing(entries, config.minimum_gap_ms, config.maximum_gap_ms)
    except ValueError:
        if rejection_counts is not None:
            rejection_counts["spacing"] = rejection_counts.get("spacing", 0) + 1
        return None
    available = sum(
        event.onset_ms + config.feature_blank_ms + config.feature_window_ms
        <= ball.commit_ms + 1e-9
        for event in entries
    )
    if available < 3:
        if rejection_counts is not None:
            rejection_counts["observations"] = rejection_counts.get("observations", 0) + 1
        return None
    if require_post_commit_bounce and not _has_wall_bounce_after_commit(ball):
        if rejection_counts is not None:
            rejection_counts["post_commit_bounce"] = rejection_counts.get("post_commit_bounce", 0) + 1
        return None
    return entries


def generate_accepted_balls(
    config: ExperimentConfig,
    residual_range: tuple[float, float],
    count: int,
    split: str,
    seed: int | None = None,
    require_post_commit_bounce: bool = True,
    opposite_only: bool = False,
    max_attempts_per_ball: int = 2000,
    stats: dict | None = None,
) -> list[tuple[BallSpec, list[CellEntry]]]:
    if count <= 0 or residual_range[0] <= 0 or residual_range[1] < residual_range[0]:
        raise ValueError("ball count or residual range is invalid")
    seed = config.seed if seed is None else seed
    accepted = []
    attempts = 0
    rejection_counts: dict[str, int] = {}
    max_attempts = count * max_attempts_per_ball
    while len(accepted) < count and attempts < max_attempts:
        trajectory_id = f"{split}-{seed}-{attempts:08d}"
        ball = seeded_ball(trajectory_id, split, residual_range, config.fall_ms,
                           stable_seed(split, seed, base_seed=config.seed),
                           config.commit_fraction)
        events = _accepted_candidate(ball, config, require_post_commit_bounce, rejection_counts)
        if events is not None:
            commit_x = reflected_position(ball.x0 + ball.vx * ball.commit_fraction)
            landing_x = landing_position(ball)
            is_opposite = (commit_x < 0.5) != (landing_x < 0.5)
            if not opposite_only or is_opposite:
                accepted.append((ball, events))
            else:
                rejection_counts["non_opposite_supplement"] = (
                    rejection_counts.get("non_opposite_supplement", 0) + 1
                )
        attempts += 1
    if stats is not None:
        stats.update({"attempts": attempts, "accepted": len(accepted),
                      "rejected": attempts - len(accepted),
                      "rejection_rate": (attempts - len(accepted)) / attempts if attempts else 0.0,
                      "rejection_reasons": rejection_counts})
    if len(accepted) < count:
        raise RuntimeError(
            f"accepted {len(accepted)}/{count} {split} trajectories after {attempts} attempts; "
            "the calibrated timing and game constraints are incompatible"
        )
    return accepted


def make_trial_trace(
    ball: BallSpec,
    entries: list[CellEntry],
    reservoir,
    runner: ReservoirRunner,
    config: ExperimentConfig,
    noise_seed: int,
) -> TrialTrace:
    validate_pulse_spacing(entries, config.minimum_gap_ms, config.maximum_gap_ms)
    events = map_entries_to_electrodes(entries, reservoir.electrode_grid,
                                       config.grid_cols, config.grid_rows,
                                       config.amplitude_ua, config.phase_ms)
    absolute_events = tuple(
        PulseEvent(e.onset_ms + config.warmup_ms, e.electrode_id, e.amplitude_ua,
                   e.phase_ms, e.col, e.row) for e in events
    )
    duration = config.warmup_ms + ball.fall_ms + config.feature_blank_ms + config.feature_window_ms + 0.1
    simulation = runner.simulate(absolute_events, duration, noise_seed)
    features = extract_post_pulse_features(
        simulation.spike_channels,
        simulation.spike_channel_times_ms,
        list(absolute_events),
        runner.channel_ids,
        config.feature_blank_ms,
        config.feature_window_ms,
    )
    availability = np.asarray([
        e.onset_ms + config.feature_blank_ms + config.feature_window_ms - config.warmup_ms
        for e in absolute_events
    ], dtype=float)
    return TrialTrace(ball, tuple(events), simulation, features, availability,
                      tuple(tuple(row) for row in reservoir.electrode_grid),
                      tuple(int(channel) for channel in runner.channel_ids),
                      config.config_hash, reservoir.reservoir_hash)


def trace_to_record(
    trace: TrialTrace,
    config: ExperimentConfig,
    slot: int,
    repeat: int = 0,
    readouts: dict | None = None,
) -> TrialRecord:
    """Record a simulation and, when fitted models are supplied, score its three controllers."""
    ball = trace.ball
    before_commit = trace.availability_ms <= ball.commit_ms + 1e-9
    times = trace.availability_ms[before_commit]
    features = trace.feature_counts[before_commit]
    if len(times) < 3:
        raise ValueError("ball has fewer than three usable pre-commit observations")

    neural_predictions = raw_predictions = np.zeros(len(times))
    paddle = raw_paddle = follow_paddle = paddle_times = np.zeros(0)
    caught = raw_caught = follow_caught = False
    pulse_cutoff_policy = "strictly_before"
    if readouts is not None:
        pulse_cutoff_policy = readouts.get("metadata", {}).get(
            "pulse_cutoff_policy", "at_or_before",
        )
        events = [PulseEvent(e.onset_ms + config.warmup_ms, e.electrode_id) for e in trace.events]
        channels = readouts.get("metadata", {}).get("recording_channels")
        if channels is not None and tuple(channels) != trace.recording_channels:
            raise ValueError("readout channel order does not match the recording")
        grid_ids = tuple(int(row[0]) for row in trace.electrode_grid)
        if "grid_ids" in readouts and tuple(readouts["grid_ids"]) != grid_ids:
            raise ValueError("readout grid does not match the delivered electrodes")
        raw = build_prefix_design(
            events, times + config.warmup_ms, grid_ids,
            config.warmup_ms + ball.fall_ms, config.stimulus_dt_ms,
            cutoff_policy=pulse_cutoff_policy,
        )
        neural_predictions = locked_prediction(readouts["neural"], features)
        raw_predictions = locked_prediction(readouts["raw"], raw)
        follow_predictions = follow_ball_targets(ball, times)
        motion = (config.paddle_width, config.paddle_speed_per_fall, config.stimulus_dt_ms)
        paddle, caught = paddle_trace(ball, times, neural_predictions, *motion)
        raw_paddle, raw_caught = paddle_trace(ball, times, raw_predictions, *motion)
        follow_paddle, follow_caught = paddle_trace(ball, times, follow_predictions, *motion)
        paddle_times = paddle_trace_times(ball.fall_ms, config.stimulus_dt_ms)

    return TrialRecord(
        trajectory_id=trace.ball.trajectory_id,
        reservoir_hash=trace.reservoir_hash,
        config_hash=trace.config_hash,
        noise_seed=trace.simulation.noise_seed,
        pulse_times_ms=np.asarray([e.onset_ms for e in trace.events]),
        pulse_electrode_ids=np.asarray([e.electrode_id for e in trace.events]),
        pulse_cells=np.asarray([(e.col, e.row) for e in trace.events]),
        spike_times_ms=trace.simulation.spike_times_ms,
        spike_gids=trace.simulation.spike_gids,
        spike_channels=trace.simulation.spike_channels,
        spike_channel_times_ms=trace.simulation.spike_channel_times_ms,
        feature_counts=features,
        availability_ms=times,
        neural_predictions=neural_predictions,
        raw_predictions=raw_predictions,
        paddle_positions=paddle,
        paddle_times_ms=paddle_times,
        raw_paddle_positions=raw_paddle,
        follow_paddle_positions=follow_paddle,
        landing_x=landing_position(trace.ball),
        caught=caught,
        raw_caught=raw_caught,
        follow_caught=follow_caught,
        metadata={
            "reservoir_slot": int(slot),
            "readouts_available": readouts is not None,
            "readout_id": readouts["readout_id"] if readouts else None,
            "split": trace.ball.split,
            "x0": trace.ball.x0,
            "vx": trace.ball.vx,
            "fall_ms": trace.ball.fall_ms,
            "commit_ms": trace.ball.commit_ms,
            "warmup_ms": config.warmup_ms,
            "repeat": int(repeat),
            "feature_window_ms": config.feature_window_ms,
            "feature_blank_ms": config.feature_blank_ms,
            "pulse_amplitude_ua": config.amplitude_ua,
            "pulse_phase_ms": config.phase_ms,
            "pulse_phase_amplitudes_ua": [config.amplitude_ua, -config.amplitude_ua],
            "pulse_phase_durations_ms": [config.phase_ms, config.phase_ms],
            "electrode_grid": [list(row) for row in trace.electrode_grid],
            "recording_channels": list(trace.recording_channels),
            "label_scope": "landing_position",
            "pulse_cutoff_policy": pulse_cutoff_policy,
            "decision_policy_version": pulse_policy_version(pulse_cutoff_policy),
        },
    )
