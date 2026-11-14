"""Reset, blanking, zero-input, and artifact controls for calibration."""

from __future__ import annotations

import numpy as np

from .calibration_support import CalibrationRawRecorder, holdout_accuracy
from .records import ExperimentConfig, stable_seed
from .reservoir import Reservoir, ReservoirRunner
from .stimulus import PulseEvent, extract_post_pulse_features


def synthetic_artifact_blanking_control(
    phase_ms: float,
    guard_ms: float,
    window_ms: float,
    recording_channels: tuple[int, ...],
) -> dict:
    """Verify the imposed blind interval excludes synthetic artifact events."""
    if not recording_channels:
        raise ValueError("at least one recording channel is required")
    onset = 5.0
    blank_ms = 2.0 * phase_ms + guard_ms
    event = PulseEvent(onset, recording_channels[0], 80.0, phase_ms)
    artifact_channels = np.asarray(
        recording_channels[:3] if len(recording_channels) >= 3
        else (recording_channels[0],) * 3,
        dtype=np.int32,
    )
    artifact_times = onset + np.asarray(
        (0.01, phase_ms, max(0.0, blank_ms - 1e-6)), dtype=float,
    )
    artifact_features = extract_post_pulse_features(
        artifact_channels, artifact_times, [event], recording_channels,
        blank_ms, window_ms,
    )

    # Include boundary spikes so the check also pins the half-open feature window:
    # the left edge counts; the right edge does not.
    boundary_channels = np.asarray(
        (recording_channels[0], recording_channels[-1], recording_channels[0]),
        dtype=np.int32,
    )
    boundary_times = onset + np.asarray(
        (blank_ms, blank_ms + window_ms - 1e-6, blank_ms + window_ms), dtype=float,
    )
    valid_features = extract_post_pulse_features(
        boundary_channels, boundary_times, [event], recording_channels,
        blank_ms, window_ms,
    )
    artifact_sum = int(artifact_features.sum())
    valid_count = int(valid_features.sum())
    passed = artifact_sum == 0 and valid_count == 2
    if not passed:
        raise RuntimeError("synthetic artifact/feature-window blanking control failed")
    return {
        "passed": passed,
        "artifact_events_injected": len(artifact_times),
        "artifact_feature_sum": artifact_sum,
        "post_blank_spikes_counted": valid_count,
        "blank_ms": blank_ms,
        "window_ms": window_ms,
        "interpretation": "synthetic events check recording-rule behavior, not biological artifact recovery",
    }


def _longer_blanking_sensitivity(
    recorder: CalibrationRawRecorder,
    reservoir: Reservoir,
    config: ExperimentConfig,
    phase_ms: float,
    amplitude_ua: float,
    window_ms: float,
) -> list[dict]:
    rows = []
    base_blank = 2.0 * phase_ms + config.guard_ms
    selected = [i for i, kind in enumerate(recorder.kind)
                if kind == "spatial"
                and np.isclose(recorder.amplitude[i], amplitude_ua)
                and np.isclose(recorder.phase[i], phase_ms)]
    labels, times = [], []
    for i in selected:
        labels.append(int(recorder.electrode[i]))
        times.append((recorder.channel_ids[i], recorder.channel_times[i]))
    if not selected:
        return rows
    grid_channels = tuple(int(row[0]) for row in reservoir.electrode_grid)
    class_labels = np.asarray([grid_channels.index(label) for label in labels])
    for extra in (0.0, 0.1, 0.2, 0.5, 1.0):
        features = []
        for channel_ids, spike_times in times:
            count_vector = np.zeros(len(reservoir.env.io.channel_ids), dtype=np.float32)
            start, end = 5.0 + base_blank + extra, 5.0 + base_blank + extra + window_ms
            in_window = (spike_times >= start) & (spike_times < end)
            for j, channel_id in enumerate(reservoir.env.io.channel_ids):
                count_vector[j] = np.count_nonzero(in_window & (channel_ids == channel_id))
            features.append(count_vector)
        accuracy, _ = holdout_accuracy(np.asarray(features), class_labels, seed=557)
        rows.append({"extra_blank_ms": extra, "blank_ms": base_blank + extra,
                     "window_ms": window_ms, "location_accuracy": accuracy,
                     "recordings": len(features),
                     "label": "post hoc longer blanking; imposed simulation rule"})
    return rows


def run_calibration_controls(
    reservoir: Reservoir,
    config: ExperimentConfig,
    recorder: CalibrationRawRecorder,
    phase_ms: float,
    amplitude_ua: float,
    window_ms: float,
    repeats: int = 20,
) -> dict:
    runner = ReservoirRunner(reservoir, config)
    channels = tuple(int(ch) for ch in runner.channel_ids)
    synthetic_artifact_control = synthetic_artifact_blanking_control(
        phase_ms, config.guard_ms, window_ms, channels,
    )

    empty = extract_post_pulse_features(
        np.zeros(0, dtype=np.int32), np.zeros(0),
        [PulseEvent(1.0, channels[0], amplitude_ua, phase_ms)],
        channels, 2.0 * phase_ms + config.guard_ms, window_ms,
    )
    if np.any(empty):
        raise RuntimeError("zero-spike feature control returned a nonzero count")

    baseline_spikes = []
    for repeat in range(repeats):
        seed = stable_seed("no-stimulation-control", reservoir.slot, repeat,
                          base_seed=config.seed)
        result = runner.simulate([], 10.0, seed)
        recorder.add(result, "no_stimulation_control", seed)
        baseline_spikes.append(len(result.spike_times_ms))

    grid = [int(row[0]) for row in reservoir.electrode_grid]
    previous_labels = [grid[0], grid[3], grid[12], grid[15]]
    probe_channel = grid[5]
    prior_features, prior_classes = [], []
    max_reset_difference = 0.0
    for repeat in range(repeats):
        probe_seed = stable_seed("carryover-probe", reservoir.slot, repeat,
                                 base_seed=config.seed)
        outputs = []
        for label, previous in enumerate(previous_labels):
            prior_seed = stable_seed("previous-ball", reservoir.slot, label, repeat,
                                     base_seed=config.seed)
            prior_event = PulseEvent(5.0, previous, amplitude_ua, phase_ms)
            runner.simulate([prior_event], 10.0, prior_seed)
            probe_event = PulseEvent(5.0, probe_channel, amplitude_ua, phase_ms)
            probe = runner.simulate([probe_event], 10.0, probe_seed)
            features = extract_post_pulse_features(
                probe.spike_channels, probe.spike_channel_times_ms, [probe_event],
                channels, 2.0 * phase_ms + config.guard_ms, window_ms,
            )[0]
            outputs.append(features)
            prior_features.append(features)
            prior_classes.append(label)
        reference = outputs[0]
        max_reset_difference = max(max_reset_difference,
                                   float(np.max(np.abs(np.asarray(outputs) - reference))))
    carryover_accuracy, _ = holdout_accuracy(
        np.asarray(prior_features), np.asarray(prior_classes), seed=991,
    )
    train_counts, tail_counts = [], []
    minimum_gap = 2.0 * phase_ms + config.guard_ms + window_ms
    train_electrode = grid[5]
    for repeat in range(repeats):
        onset = 5.0
        train_events = [PulseEvent(onset + i * minimum_gap, train_electrode,
                                   amplitude_ua, phase_ms) for i in range(12)]
        tail_start = (train_events[-1].onset_ms + 2.0 * phase_ms
                      + config.guard_ms + window_ms)
        duration = tail_start + 5.0
        seed = stable_seed("repeated-pulse-train", reservoir.slot, repeat,
                           base_seed=config.seed)
        train = runner.simulate(train_events, duration, seed)
        recorder.add(train, "pulse_train", seed, electrode=train_electrode,
                     gap_ms=minimum_gap, amplitude_ua=amplitude_ua, phase_ms=phase_ms)
        features = extract_post_pulse_features(
            train.spike_channels, train.spike_channel_times_ms, train_events,
            channels, 2.0 * phase_ms + config.guard_ms, window_ms,
        )
        train_counts.append(features.sum(axis=1))
        tail_counts.append(int(np.count_nonzero(
            (train.spike_times_ms >= tail_start)
            & (train.spike_times_ms < tail_start + 5.0)
        )))
    train_counts = np.asarray(train_counts, dtype=float)
    first_response = float(train_counts[:, :3].mean())
    last_response = float(train_counts[:, -3:].mean())
    return {
        "no_stimulation": {
            "trials": len(baseline_spikes),
            "mean_network_spikes": float(np.mean(baseline_spikes)),
            "p99_network_spikes": float(np.quantile(baseline_spikes, 0.99)),
        },
        "zero_spike_feature_control": {"passed": True, "feature_sum": float(empty.sum())},
        "synthetic_artifact_blanking_control": synthetic_artifact_control,
        "repeated_pulse_train": {
            "trials": repeats,
            "pulses_per_train": 12,
            "gap_ms": minimum_gap,
            "first_three_mean_spikes": first_response,
            "last_three_mean_spikes": last_response,
            "late_to_early_response_ratio": last_response / first_response if first_response else None,
            "tail_spikes_after_last_counting_window_mean": float(np.mean(tail_counts)),
            "tail_spikes_after_last_counting_window_p99": float(np.quantile(tail_counts, 0.99)),
        },
        "previous_ball_label_control": {
            "labels": ["upper-left", "upper-right", "lower-left", "lower-right"],
            "probe_electrode": probe_channel,
            "accuracy": carryover_accuracy,
            "chance": 0.25,
            "maximum_same_seed_feature_difference": max_reset_difference,
            "repeats_per_previous_label": repeats,
            "interpretation": "Probe response follows a between-ball reset; label decoding checks carryover.",
        },
        "longer_blanking_sensitivity": _longer_blanking_sensitivity(
            recorder, reservoir, config, phase_ms, amplitude_ua, window_ms,
        ),
    }
