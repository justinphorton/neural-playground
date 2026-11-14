"""Candidate Pong paths and a pulse-only geometry diagnostic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .experiment import _accepted_candidate
from .physics import BallSpec, landing_position
from .records import ExperimentConfig


@dataclass(frozen=True)
class PathHypothesis:
    landing_x: float
    pulse_ticks: tuple[int, ...]
    pulse_cells: tuple[int, ...]


def build_path_hypotheses(
    config: ExperimentConfig,
    residual_range: tuple[float, float] | None = None,
    resolution: float = 0.01,
) -> tuple[PathHypothesis, ...]:
    """Enumerate timing-valid paths from the declared court and speed range."""
    if resolution <= 0:
        raise ValueError("path resolution must be positive")
    low, high = tuple(residual_range or config.selected_residual_range)
    if low <= 0 or high < low:
        raise ValueError("residual range must be positive and ordered")
    x_values = np.arange(0.10, 0.90 + resolution * 0.5, resolution)
    residuals = np.arange(low, high + resolution * 0.5, resolution)
    templates = []
    for x0 in x_values:
        for residual in residuals:
            for sign in (-1.0, 1.0):
                vx = sign * float(residual) / (1.0 - config.commit_fraction)
                ball = BallSpec("template", float(x0), vx, config.fall_ms,
                                config.commit_fraction, "pilot")
                entries = _accepted_candidate(ball, config, True)
                if entries is None:
                    continue
                templates.append(PathHypothesis(
                    landing_position(ball),
                    tuple(int(round(event.onset_ms / config.stimulus_dt_ms))
                          for event in entries),
                    tuple(event.row * config.grid_cols + event.col for event in entries),
                ))
    if not templates:
        raise RuntimeError("no timing-valid Pong path hypotheses were generated")
    return tuple(templates)


def decode_pulse_prefix(
    hypotheses: tuple[PathHypothesis, ...] | list[PathHypothesis],
    observed_ticks: tuple[int, ...] | list[int],
    observed_cells: tuple[int, ...] | list[int],
    decision_tick: int,
    tolerance_ticks: int = 1,
) -> float:
    """Estimate landing from pulses strictly earlier than ``decision_tick``."""
    if len(observed_ticks) != len(observed_cells):
        raise ValueError("observed pulse times and cells must have equal length")
    if decision_tick < 0 or tolerance_ticks < 0:
        raise ValueError("decision and timing tolerance must be non-negative")
    visible = [
        (int(tick), int(cell))
        for tick, cell in zip(observed_ticks, observed_cells)
        if int(tick) < decision_tick
    ]
    if not hypotheses:
        raise ValueError("at least one path hypothesis is required")
    if not visible:
        return float(np.mean([item.landing_x for item in hypotheses]))

    observed_t = np.asarray([item[0] for item in visible], dtype=float)
    observed_c = np.asarray([item[1] for item in visible], dtype=int)
    scores = np.empty(len(hypotheses), dtype=float)
    for index, hypothesis in enumerate(hypotheses):
        expected = [
            (tick, cell) for tick, cell in zip(
                hypothesis.pulse_ticks, hypothesis.pulse_cells,
            ) if tick < decision_tick
        ]
        paired = min(len(expected), len(visible))
        if paired:
            expected_t = np.asarray([item[0] for item in expected[:paired]], dtype=float)
            expected_c = np.asarray([item[1] for item in expected[:paired]], dtype=int)
            timing = np.maximum(np.abs(expected_t - observed_t[:paired]) - tolerance_ticks, 0.0)
            cells = expected_c != observed_c[:paired]
            scores[index] = float(np.sum(timing * timing) + 36 * np.count_nonzero(cells))
        else:
            scores[index] = 0.0
        scores[index] += 36.0 * abs(len(expected) - len(visible))

    weights = np.exp(-0.5 * np.minimum(scores - scores.min(), 140.0))
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        return float(hypotheses[int(np.argmin(scores))].landing_x)
    landings = np.asarray([item.landing_x for item in hypotheses], dtype=float)
    return float(np.dot(weights, landings) / total)


def geometry_target_trace(
    events,
    decision_times_ms: np.ndarray,
    hypotheses: tuple[PathHypothesis, ...] | list[PathHypothesis],
    dt_ms: float,
    grid_cols: int = 4,
    tolerance_ticks: int = 1,
) -> np.ndarray:
    """Apply the pulse-only decoder at each decision using its visible prefix."""
    if dt_ms <= 0 or grid_cols <= 0:
        raise ValueError("stimulus time step and grid width must be positive")
    decisions = np.asarray(decision_times_ms, dtype=float)
    if decisions.ndim != 1 or np.any(~np.isfinite(decisions)):
        raise ValueError("decision times must be a finite vector")
    ordered = sorted(events, key=lambda item: item.onset_ms)
    pulse_ticks = [int(round(item.onset_ms / dt_ms)) for item in ordered]
    pulse_cells = [int(item.row * grid_cols + item.col) for item in ordered]
    targets = []
    for decision in decisions:
        decision_tick = int(np.floor(decision / dt_ms + 1e-9))
        targets.append(decode_pulse_prefix(
            hypotheses, pulse_ticks, pulse_cells, decision_tick, tolerance_ticks,
        ))
    return np.asarray(targets, dtype=float)
