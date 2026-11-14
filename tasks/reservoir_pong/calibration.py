"""Label-free stimulation and state-retention gates for simulated reservoirs."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .calibration_report import write_calibration_report
from .calibration_support import CalibrationRawRecorder, holdout_accuracy
from .records import ExperimentConfig, stable_seed
from .reservoir import ReservoirRunner, build_reservoir
from .stimulus import PulseEvent, extract_post_pulse_features


@dataclass(frozen=True)
class CalibrationCandidate:
    phase_ms: float
    amplitude_ua: float
    window_ms: float
    passed: bool
    response_rate: float
    location_accuracy: float
    background_p99: float = 0.0
    electrode_response_rates: tuple[float, ...] = ()


@dataclass(frozen=True)
class MemoryGapResult:
    gap_ms: float
    accuracy: float
    lower_95: float
    passed: bool
    phase_ms: float = 0.0
    amplitude_ua: float = 0.0
    window_ms: float = 0.0


@dataclass(frozen=True)
class ReservoirCalibration:
    slot: int
    reservoir_hash: str
    background_mean: float
    background_p99: float
    candidates: tuple[CalibrationCandidate, ...]
    memory: tuple[MemoryGapResult, ...]
    usable_maximum_gap_ms: float | None
    status: str
    controls: dict = field(default_factory=dict)
    electrode_grid: tuple[tuple[int, float, float], ...] = ()



def choose_calibration(candidates: list[CalibrationCandidate]) -> CalibrationCandidate:
    passing = [item for item in candidates if item.passed]
    if not passing:
        raise RuntimeError("no stimulation configuration passed spatial calibration")
    return min(passing, key=lambda c: (c.phase_ms, c.amplitude_ua, c.window_ms))


def contiguous_memory_limit(
    measured: list[MemoryGapResult], minimum_gap_ms: float
) -> float | None:
    valid = sorted((m for m in measured if m.gap_ms >= minimum_gap_ms - 1e-9), key=lambda m: m.gap_ms)
    if not valid or not valid[0].passed:
        return None
    maximum = valid[0].gap_ms
    for item in valid[1:]:
        if not item.passed:
            break
        maximum = item.gap_ms
    return maximum



def _bootstrap_lower(correct: np.ndarray, rng_seed: int, repeats: int = 2000) -> float:
    correct = np.asarray(correct, dtype=float)
    if len(correct) == 0:
        return 0.0
    rng = np.random.default_rng(rng_seed)
    indices = rng.integers(0, len(correct), size=(repeats, len(correct)))
    return float(np.quantile(correct[indices].mean(axis=1), 0.025))


def _pulse_features(runner, electrode: int | None, amplitude: float,
                    phase_ms: float, seed: int, windows=(0.5, 1.0, 2.0),
                    guard_ms: float = 0.1, recorder: CalibrationRawRecorder | None = None,
                    kind: str = "background") -> np.ndarray:
    onset = 5.0
    events = [] if electrode is None else [PulseEvent(onset, electrode, amplitude, phase_ms)]
    result = runner.simulate(events, duration_ms=10.0, noise_seed=seed)
    if recorder is not None:
        recorder.add(result, kind, seed, -1 if electrode is None else electrode,
                     amplitude_ua=amplitude, phase_ms=phase_ms)
    marker = [PulseEvent(onset, runner.channel_ids[0] if electrode is None else electrode,
                         amplitude, phase_ms)]
    return np.asarray([
        extract_post_pulse_features(
            result.spike_channels, result.spike_channel_times_ms, marker,
            runner.channel_ids, blank_ms=2.0 * phase_ms + guard_ms, window_ms=window,
        )[0]
        for window in windows
    ])


def _measure_spatial_candidate(
    reservoir,
    runner: ReservoirRunner,
    phase_ms: float,
    amplitude: float,
    repeats: int,
    noise_seeds: list[int],
    background: np.ndarray,
    guard_ms: float,
    recorder: CalibrationRawRecorder | None = None,
) -> list[CalibrationCandidate]:
    channel_ids = [int(row[0]) for row in reservoir.electrode_grid]
    windows = (0.5, 1.0, 2.0)
    raw = np.zeros((len(windows), len(channel_ids), repeats, len(runner.channel_ids)), dtype=float)
    for label, channel in enumerate(channel_ids):
        for repeat, seed in enumerate(noise_seeds):
            onset = 5.0
            event = PulseEvent(onset, channel, amplitude, phase_ms, label % 4, label // 4)
            sim = runner.simulate([event], 10.0, seed)
            if recorder is not None:
                recorder.add(sim, "spatial", seed, electrode=channel,
                             amplitude_ua=amplitude, phase_ms=phase_ms)
            raw[:, label, repeat] = np.asarray([
                extract_post_pulse_features(
                    sim.spike_channels, sim.spike_channel_times_ms, [event],
                    runner.channel_ids, blank_ms=2.0 * phase_ms + guard_ms, window_ms=window,
                )[0] for window in windows
            ])
    candidates = []
    labels = np.repeat(np.arange(len(channel_ids)), repeats)
    for i, window in enumerate(windows):
        baseline_limit = float(np.quantile(background[:, i, :].sum(axis=1), 0.99))
        per_electrode_response_rate = np.mean(raw[i].sum(axis=2) > baseline_limit, axis=1)
        response_rate = float(np.min(per_electrode_response_rate))
        features = raw[i].reshape((-1, len(runner.channel_ids)))
        accuracy, _ = holdout_accuracy(features, labels, seed=307)
        passed = response_rate >= 0.80 and accuracy > 0.50
        candidates.append(CalibrationCandidate(phase_ms, amplitude, window, passed,
                                               response_rate, accuracy, baseline_limit,
                                               tuple(float(x) for x in per_electrode_response_rate)))
    return candidates


def _calibrate_spatial_one(
    reservoir,
    config: ExperimentConfig,
    repeats: int,
    slot: int,
    recorder: CalibrationRawRecorder | None = None,
    required_candidate: CalibrationCandidate | None = None,
) -> tuple[list[CalibrationCandidate], dict[float, np.ndarray]]:
    runner = ReservoirRunner(reservoir, config)
    seeds = [stable_seed("spatial", slot, i, base_seed=config.seed) for i in range(repeats)]
    background_by_phase = {}
    candidates = []
    for phase_ms in (0.1, 0.2, 0.5):
        background = np.asarray([_pulse_features(
            runner, None, config.amplitude_ua, phase_ms, seed,
            guard_ms=config.guard_ms, recorder=recorder,
        ) for seed in seeds])
        background_by_phase[phase_ms] = background
        for amplitude in (40.0, 80.0, 120.0):
            measured = _measure_spatial_candidate(
                reservoir, runner, phase_ms, amplitude, repeats, seeds, background,
                config.guard_ms, recorder,
            )
            candidates.extend(measured)
            if required_candidate is not None and (phase_ms, amplitude) == (
                    required_candidate.phase_ms, required_candidate.amplitude_ua):
                return candidates, background_by_phase
            if required_candidate is None and any(candidate.passed for candidate in measured):
                return candidates, background_by_phase
    return candidates, background_by_phase


def _memory_calibration(
    reservoir, config: ExperimentConfig, phase_ms: float, amplitude: float,
    window_ms: float, slot: int, repeats: int,
    recorder: CalibrationRawRecorder | None = None,
) -> list[MemoryGapResult]:
    runner = ReservoirRunner(reservoir, config)
    grid = [int(row[0]) for row in reservoir.electrode_grid]
    previous_channels = [grid[0], grid[3], grid[12], grid[15]]
    probe = grid[5]
    gap_values = (1.0, 2.0, 3.0, 6.0, 10.0, 20.0, 40.0)
    results = []
    min_gap = 2 * phase_ms + config.guard_ms + window_ms
    for gap in gap_values:
        if gap < min_gap - 1e-9:
            results.append(MemoryGapResult(gap, 0.0, 0.0, False))
            continue
        counts, labels = [], []
        for label, previous in enumerate(previous_channels):
            for repeat in range(repeats):
                seed = stable_seed("memory", slot, gap, label, repeat, base_seed=config.seed)
                first, second = PulseEvent(5.0, previous, amplitude, phase_ms), PulseEvent(
                    5.0 + gap, probe, amplitude, phase_ms,
                )
                sim = runner.simulate([first, second], 8.0 + gap, seed)
                if recorder is not None:
                    recorder.add(sim, "memory", seed, electrode=probe, previous=previous,
                                 gap_ms=gap, amplitude_ua=amplitude, phase_ms=phase_ms)
                counts.append(extract_post_pulse_features(
                    sim.spike_channels, sim.spike_channel_times_ms, [second],
                    runner.channel_ids, 2.0 * phase_ms + config.guard_ms, window_ms,
                )[0])
                labels.append(label)
        accuracy, correct = holdout_accuracy(np.asarray(counts), np.asarray(labels), seed=slot + 900)
        lower = _bootstrap_lower(correct, rng_seed=slot + int(gap * 13))
        passed = accuracy >= 0.50 and lower > 0.25
        results.append(MemoryGapResult(gap, accuracy, lower, passed))
    return results



def calibrate_run(
    run_dir: Path,
    config: ExperimentConfig,
    repeats: int | None = None,
) -> tuple[ExperimentConfig, list[ReservoirCalibration]]:
    """Find the shortest common spatial-and-memory calibration across all patches."""
    run_dir = Path(run_dir)
    (run_dir / "calibration").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    repeats = int(repeats or config.calibration_repeats)
    if repeats < 8:
        raise ValueError("calibration requires at least eight repeats per spatial location")
    print(
        f"[calibration] building {config.reservoir_count} fixed {config.n_cells}-cell "
        f"subreservoirs; {repeats} repeats per grid location",
        flush=True,
    )

    reservoirs = []
    recorders = []
    runners = []
    candidates_by_slot = [[] for _ in range(config.reservoir_count)]
    backgrounds_by_slot = [{} for _ in range(config.reservoir_count)]
    memories_by_slot = [[] for _ in range(config.reservoir_count)]
    selected = None
    selected_memories = None
    selected_maxima = None
    failure_kind = "failed_spatial_gate"

    try:
        for slot in range(config.reservoir_count):
            reservoir = build_reservoir(config, slot)
            reservoirs.append(reservoir)
            recorder = CalibrationRawRecorder()
            recorders.append(recorder)
            runners.append(ReservoirRunner(reservoir, config))
            manifest = {
                "config_hash": config.config_hash,
                "livn_version": version("livn"),
                "python_version": platform.python_version(),
                "numpy_version": version("numpy"),
                "scipy_version": version("scipy"),
                "scikit_learn_version": version("scikit-learn"),
                "software_version": config.software_version,
                "backend": type(reservoir.env).__module__,
                "reservoir_hash": reservoir.reservoir_hash,
                "model_hash": reservoir.model_hash,
                "selected_gids": reservoir.selected_gids.astype(int).tolist(),
                "electrode_grid": reservoir.electrode_grid,
                "grid_center_source": "mean physical coordinate of the selected neuron IDs",
                "plasticity_enabled": reservoir.plasticity_enabled,
                "is_subreservoir_of_one_simulated_culture": True,
            }
            (run_dir / "calibration" / f"reservoir_slot{slot}.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )

        for phase_ms in (0.1, 0.2, 0.5):
            for slot, runner in enumerate(runners):
                seeds = [stable_seed("spatial", slot, i, base_seed=config.seed)
                         for i in range(repeats)]
                backgrounds_by_slot[slot][phase_ms] = np.asarray([
                    _pulse_features(
                        runner, None, config.amplitude_ua, phase_ms, seed,
                        guard_ms=config.guard_ms, recorder=recorders[slot],
                    ) for seed in seeds
                ])
            for amplitude in (40.0, 80.0, 120.0):
                measured_by_slot = []
                for slot, (reservoir, runner) in enumerate(zip(reservoirs, runners)):
                    seeds = [stable_seed("spatial", slot, i, base_seed=config.seed)
                             for i in range(repeats)]
                    measured = _measure_spatial_candidate(
                        reservoir, runner, phase_ms, amplitude, repeats, seeds,
                        backgrounds_by_slot[slot][phase_ms], config.guard_ms,
                        recorders[slot],
                    )
                    candidates_by_slot[slot].extend(measured)
                    measured_by_slot.append(measured)
                print(
                    f"[calibration] {phase_ms:g} ms/phase, {amplitude:g} uA: "
                    + ", ".join(
                        f"slot {slot} response={min(c.response_rate for c in measured):.2f} "
                        f"location_id={max(c.location_accuracy for c in measured):.2f}"
                        for slot, measured in enumerate(measured_by_slot)
                    ),
                    flush=True,
                )
                for window_ms in (0.5, 1.0, 2.0):
                    selected_candidates = [next(item for item in measured
                                                 if np.isclose(item.window_ms, window_ms))
                                           for measured in measured_by_slot]
                    if not all(item.passed for item in selected_candidates):
                        continue
                    failure_kind = "failed_memory_gate"
                    print(
                        f"[calibration] spatial gate passed for {phase_ms:g} ms/phase, "
                        f"{amplitude:g} uA, {window_ms:g} ms window; measuring memory",
                        flush=True,
                    )
                    trial_memories, maxima = [], []
                    for slot, (reservoir, recorder) in enumerate(zip(reservoirs, recorders)):
                        calibrated = replace(
                            config, phase_ms=phase_ms, amplitude_ua=amplitude,
                            feature_window_ms=window_ms,
                            feature_blank_ms=2 * phase_ms + config.guard_ms,
                        )
                        memory = _memory_calibration(
                            reservoir, calibrated, phase_ms, amplitude,
                            window_ms, slot, repeats, recorder,
                        )
                        memory = [replace(row, phase_ms=phase_ms,
                                          amplitude_ua=amplitude, window_ms=window_ms)
                                  for row in memory]
                        memories_by_slot[slot].extend(memory)
                        trial_memories.append(memory)
                        minimum = 2 * phase_ms + config.guard_ms + window_ms
                        maxima.append(contiguous_memory_limit(memory, minimum))
                        passing = [item for item in memory if item.passed]
                        print(
                            f"[calibration] slot {slot} memory: "
                            f"{len(passing)}/{len(memory)} tested gaps passed; "
                            f"longest contiguous={maxima[-1]}",
                            flush=True,
                        )
                    if all(value is not None for value in maxima):
                        selected = selected_candidates[0]
                        selected_memories = trial_memories
                        selected_maxima = maxima
                        failure_kind = "passed"
                        break
                if selected is not None:
                    break
            if selected is not None:
                break

        controls_by_slot = [{} for _ in reservoirs]
        if selected is not None:
            for slot, (reservoir, recorder) in enumerate(zip(reservoirs, recorders)):
                calibrated = replace(
                    config, phase_ms=selected.phase_ms,
                    amplitude_ua=selected.amplitude_ua,
                    feature_window_ms=selected.window_ms,
                    feature_blank_ms=2 * selected.phase_ms + config.guard_ms,
                )
                try:
                    from .controls import run_calibration_controls
                    controls_by_slot[slot] = run_calibration_controls(
                        reservoir, calibrated, recorder,
                        selected.phase_ms, selected.amplitude_ua,
                        selected.window_ms, repeats=repeats,
                    )
                except Exception as error:
                    controls_by_slot[slot] = {"status": "failed", "error":
                                              f"{type(error).__name__}: {error}"}
                    failure_kind = "failed_controls"
        else:
            from .controls import synthetic_artifact_blanking_control
            for slot, runner in enumerate(runners):
                controls_by_slot[slot] = {
                    "status": "not_run_no_common_configuration",
                    "synthetic_artifact_blanking_control": synthetic_artifact_blanking_control(
                        phase_ms=0.1, guard_ms=config.guard_ms, window_ms=0.5,
                        recording_channels=runner.channel_ids,
                    ),
                }

        reports = []
        for slot, reservoir in enumerate(reservoirs):
            phase = selected.phase_ms if selected is not None else 0.1
            background = backgrounds_by_slot[slot].get(phase)
            if background is None:
                background = np.zeros((0, 3, len(runners[slot].channel_ids)))
            if selected is not None:
                memory = selected_memories[slot]
                maximum = selected_maxima[slot]
                status = "passed" if controls_by_slot[slot].get("status", "passed") != "failed" else "failed_controls"
            else:
                memory = memories_by_slot[slot]
                maximum = None
                status = failure_kind
            reports.append(ReservoirCalibration(
                slot=slot,
                reservoir_hash=reservoir.reservoir_hash,
                background_mean=float(background.mean()) if background.size else 0.0,
                background_p99=float(np.quantile(background.sum(axis=1), 0.99)) if background.size else 0.0,
                candidates=tuple(candidates_by_slot[slot]),
                memory=tuple(memory),
                usable_maximum_gap_ms=maximum,
                status=status,
                controls=controls_by_slot[slot],
                electrode_grid=tuple(tuple(row) for row in reservoir.electrode_grid),
            ))
    finally:
        for recorder, reservoir in zip(recorders, reservoirs):
            try:
                recorder.save(run_dir / "calibration" / f"raw_slot{reservoir.slot}.npz")
            finally:
                reservoir.close()

    if selected is None or any(report.status != "passed" for report in reports):
        updated = config
        gate = {"status": "failed", "reason": failure_kind,
                "selected": asdict(selected) if selected is not None else None,
                "reservoirs": [asdict(report) for report in reports]}
    else:
        minimum = 2 * selected.phase_ms + config.guard_ms + selected.window_ms
        maximum = min(report.usable_maximum_gap_ms for report in reports)
        fall_ms = config.grid_rows * (minimum + maximum) / 2.0
        updated = replace(
            config, phase_ms=selected.phase_ms, amplitude_ua=selected.amplitude_ua,
            feature_window_ms=selected.window_ms,
            feature_blank_ms=2.0 * selected.phase_ms + config.guard_ms,
            minimum_gap_ms=minimum, maximum_gap_ms=maximum, fall_ms=fall_ms,
        )
        gate = {"status": "passed", "selected": asdict(selected),
                "minimum_gap_ms": minimum, "maximum_gap_ms": maximum,
                "fall_ms": fall_ms, "reservoirs": [asdict(report) for report in reports]}
    print(
        f"[calibration] {gate['status']}: {gate.get('reason', 'shared configuration and controls complete')}",
        flush=True,
    )
    (run_dir / "calibration" / "summary.json").write_text(json.dumps(gate, indent=2) + "\n")
    (run_dir / "config.json").write_text(json.dumps(updated.to_dict(), indent=2) + "\n")
    write_calibration_report(run_dir, gate, reports, selected, updated)
    with (run_dir / "logs" / "observations.jsonl").open("a") as stream:
        stream.write(json.dumps({
            "phase": "calibration", "status": gate["status"],
            "config_hash": updated.config_hash,
            "selected": asdict(selected) if selected else None,
            "reservoirs": [{"slot": row.slot, "hash": row.reservoir_hash,
                            "status": row.status,
                            "usable_maximum_gap_ms": row.usable_maximum_gap_ms}
                           for row in reports],
        }, sort_keys=True) + "\n")
    return updated, reports
