"""Software-only task-design gate using the exact Pong pulse stream."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import vstack

from .experiment import generate_accepted_balls
from .physics import landing_position
from .pipeline import Examples
from .path_hypotheses import build_path_hypotheses, geometry_target_trace
from .readouts import (
    build_prefix_design,
    follow_ball_targets,
    paddle_trace,
    select_ridge,
    trajectory_sample_weights,
)
from .records import ExperimentConfig, stable_seed
from .stimulus import PulseEvent, map_entries_to_electrodes


@dataclass(frozen=True)
class PreflightTrajectory:
    ball: object
    events: tuple[PulseEvent, ...]
    availability_ms: np.ndarray
    raw_prefixes: object


def _trajectory_records(config: ExperimentConfig, residual_range: tuple[float, float],
                        split: str, count: int, seed: int,
                        stats: dict) -> list[PreflightTrajectory]:
    accepted = generate_accepted_balls(
        config, residual_range, count, split, seed,
        require_post_commit_bounce=True, stats=stats,
    )
    grid = [(index, float(index % config.grid_cols), float(index // config.grid_cols))
            for index in range(config.grid_cols * config.grid_rows)]
    electrode_ids = tuple(range(config.grid_cols * config.grid_rows))
    trajectories = []
    for ball, entries in accepted:
        events = tuple(map_entries_to_electrodes(
            entries, grid, config.grid_cols, config.grid_rows,
            config.amplitude_ua, config.phase_ms,
        ))
        available = np.asarray([
            event.onset_ms + config.feature_blank_ms + config.feature_window_ms
            for event in events
        ], dtype=float)
        keep = available <= ball.commit_ms + 1e-9
        available = available[keep]
        if len(available) < 3:
            raise RuntimeError("accepted preflight ball lost its three committed observations")
        absolute_events = [PulseEvent(
            event.onset_ms + config.warmup_ms, event.electrode_id,
            event.amplitude_ua, event.phase_ms, event.col, event.row,
        ) for event in events]
        absolute_available = available + config.warmup_ms
        raw = build_prefix_design(
            absolute_events, absolute_available, electrode_ids,
            config.warmup_ms + ball.fall_ms, config.stimulus_dt_ms,
        )
        trajectories.append(PreflightTrajectory(ball, events, available, raw))
    return trajectories


def _examples(trajectories: list[PreflightTrajectory]) -> Examples:
    raw = vstack([trial.raw_prefixes for trial in trajectories], format="csr")
    targets, groups = [], []
    for trial in trajectories:
        targets.extend([landing_position(trial.ball)] * len(trial.availability_ms))
        groups.extend([trial.ball.trajectory_id] * len(trial.availability_ms))
    groups = np.asarray(groups)
    return Examples(np.zeros((len(targets), 0), dtype=np.float32), raw,
                    np.asarray(targets, dtype=float), groups,
                    trajectory_sample_weights(groups))


def _evaluate(model, trajectories: list[PreflightTrajectory],
              mode: str, config: ExperimentConfig) -> tuple[float, float]:
    caught, errors = [], []
    for trial in trajectories:
        if mode == "oracle":
            prediction = np.full(len(trial.availability_ms), landing_position(trial.ball))
        elif mode == "follow_ball":
            prediction = follow_ball_targets(trial.ball, trial.availability_ms)
        else:
            if mode != "raw":
                raise ValueError(f"unsupported preflight controller {mode!r}")
            values = model.predict(trial.raw_prefixes)
            prediction = np.asarray(values, dtype=float).reshape(-1)
        _, did_catch = paddle_trace(
            trial.ball, trial.availability_ms, prediction,
            config.paddle_width, config.paddle_speed_per_fall,
            config.stimulus_dt_ms,
        )
        caught.append(did_catch)
        errors.append(abs(float(prediction[-1]) - landing_position(trial.ball)))
    return float(np.mean(caught)), float(np.mean(errors))


def _evaluate_geometry(trajectories, hypotheses, config):
    caught, errors = [], []
    for trial in trajectories:
        prediction = geometry_target_trace(
            trial.events, trial.availability_ms, hypotheses,
            config.stimulus_dt_ms, config.grid_cols,
        )
        _, did_catch = paddle_trace(
            trial.ball, trial.availability_ms, prediction,
            config.paddle_width, config.paddle_speed_per_fall,
            config.stimulus_dt_ms,
        )
        caught.append(did_catch)
        errors.append(abs(float(prediction[-1]) - landing_position(trial.ball)))
    return float(np.mean(caught)), float(np.mean(errors))


def _evaluate_candidate(
    config: ExperimentConfig,
    residual_range: tuple[float, float],
    counts: tuple[int, int, int],
    seed: int,
) -> dict:
    train_count, validation_count, test_count = counts
    generation_stats = {}
    train_trials = _trajectory_records(
        config, residual_range, "train", train_count,
        stable_seed("preflight", "train", residual_range, base_seed=seed),
        generation_stats.setdefault("train", {}),
    )
    validation_trials = _trajectory_records(
        config, residual_range, "validation", validation_count,
        stable_seed("preflight", "validation", residual_range, base_seed=seed),
        generation_stats.setdefault("validation", {}),
    )
    test_trials = _trajectory_records(
        config, residual_range, "test", test_count,
        stable_seed("preflight", "test", residual_range, base_seed=seed),
        generation_stats.setdefault("test", {}),
    )
    training, validation = _examples(train_trials), _examples(validation_trials)
    raw_model, raw_alpha, raw_scores = select_ridge(
        training.raw, training.targets, training.sample_weights,
        validation.raw, validation.targets, validation.sample_weights,
        config.regularization_grid,
    )
    hypotheses = build_path_hypotheses(config, residual_range)
    oracle_rate, _ = _evaluate(None, test_trials, "oracle", config)
    follow_rate, _ = _evaluate(None, test_trials, "follow_ball", config)
    linear_rate, linear_mae = _evaluate(raw_model, test_trials, "raw", config)
    geometry_rate, geometry_mae = _evaluate_geometry(test_trials, hypotheses, config)
    passes = oracle_rate >= 0.95 and geometry_rate >= 0.70 and geometry_rate >= linear_rate + 0.10
    return {
        "residual_range": list(residual_range),
        "counts": {"train": train_count, "validation": validation_count, "test": test_count},
        "generation": generation_stats,
        "oracle_catch_rate": oracle_rate,
        "follow_ball_catch_rate": follow_rate,
        "raw_pulse_ridge_catch_rate": linear_rate,
        "raw_pulse_landing_mae": linear_mae,
        "geometry_pulse_prefix_catch_rate": geometry_rate,
        "geometry_pulse_prefix_landing_mae": geometry_mae,
        "raw_ridge_alpha": raw_alpha,
        "raw_ridge_validation_mse": raw_scores,
        "passes_task_design_gate": passes,
    }


def run_preflight(
    run_dir: Path,
    config: ExperimentConfig,
    counts: tuple[int, int, int] = (120, 40, 80),
    seed: int | None = None,
    candidate_ranges: tuple[tuple[float, float], ...] | None = None,
) -> tuple[ExperimentConfig, dict]:
    """Screen declared residual bands; assignment runs can freeze one band."""
    run_dir = Path(run_dir)
    seed = config.seed if seed is None else seed
    candidates = candidate_ranges or ((0.40, 0.80), (0.25, 0.50), (0.60, 1.00))
    if not candidates:
        raise ValueError("preflight needs at least one residual range")
    for low, high in candidates:
        if low <= 0 or high < low:
            raise ValueError("preflight residual ranges must be positive and ordered")
    reports = []
    for band in candidates:
        report = _evaluate_candidate(config, band, counts, seed)
        reports.append(report)
        if report["passes_task_design_gate"]:
            break
    selected = next((tuple(row["residual_range"]) for row in reports
                     if row["passes_task_design_gate"]), None)
    result = {
        "status": "passed" if selected is not None else "failed_task_design_gate",
        "seed": int(seed),
        "candidate_ranges": [list(pair) for pair in candidates],
        "selected_residual_range": list(selected) if selected else None,
        "candidate_results": reports,
        "interpretation": (
            "Software-only geometry decoder using delivered pulse prefixes; this is not a neural-reservoir result."
        ),
    }
    if selected is not None:
        config = ExperimentConfig.from_dict(config.to_dict() | {
            "selected_residual_range": list(selected),
        })
        (run_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    (run_dir / "metrics" / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Software-only Pong preflight", "",
        f"Status: **{result['status']}**. {result['interpretation']}", "",
        "| Residual range | Oracle catch | Follow ball | Raw-pulse Ridge | Pulse geometry | Gate |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for item in reports:
        lines.append(
            f"| {item['residual_range'][0]:.2f}–{item['residual_range'][1]:.2f} | "
            f"{item['oracle_catch_rate']:.3f} | {item['follow_ball_catch_rate']:.3f} | "
            f"{item['raw_pulse_ridge_catch_rate']:.3f} | "
            f"{item['geometry_pulse_prefix_catch_rate']:.3f} | "
            f"{'pass' if item['passes_task_design_gate'] else 'fail'} |"
        )
    lines.extend(["", "The pulse-geometry controller is a software-only diagnostic of the encoded pulse history. "
                  "It is not a neural-reservoir result. Timing was held fixed while screening the "
                  "prespecified ranges.", ""])
    (run_dir / "metrics" / "preflight.md").write_text("\n".join(lines))
    log_path = run_dir / "logs" / "observations.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as stream:
        stream.write(json.dumps({"phase": "preflight", "status": result["status"],
                                 "config_hash": config.config_hash,
                                 "selected_residual_range": result["selected_residual_range"],
                                 "candidate_results": reports}, sort_keys=True) + "\n")
    return config, result
