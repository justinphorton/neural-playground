"""Trajectory-level shuffled-label null for frozen test evaluation."""

from __future__ import annotations

import numpy as np

from .physics import BallSpec
from .readouts import paddle_trace, select_ridge
from .records import ExperimentConfig, TrialRecord


def permute_trajectory_targets(targets: np.ndarray, groups: np.ndarray,
                               rng: np.random.Generator) -> np.ndarray:
    targets = np.asarray(targets, dtype=float)
    groups = np.asarray(groups)
    if targets.shape != groups.shape or targets.ndim != 1 or not len(groups):
        raise ValueError("trajectory targets and groups must be aligned non-empty vectors")
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("at least two trajectories are required for a label permutation")
    value_by_group = {}
    for group in unique:
        values = np.unique(targets[groups == group])
        if len(values) != 1:
            raise ValueError("each trajectory must have one repeated target")
        value_by_group[group] = values[0]
    shuffled = rng.permutation(np.asarray([value_by_group[group] for group in unique]))
    mapping = dict(zip(unique.tolist(), shuffled.tolist()))
    return np.asarray([mapping[group] for group in groups], dtype=float)


def _record_catch_rate(model, records: list[TrialRecord], config: ExperimentConfig) -> float:
    outcomes = []
    for record in records:
        if record.metadata.get("split") != "test":
            continue
        ball = BallSpec(
            record.trajectory_id,
            float(record.metadata["x0"]),
            float(record.metadata["vx"]),
            float(record.metadata["fall_ms"]),
            config.commit_fraction,
            "test",
        )
        prediction = np.asarray(model.predict(record.feature_counts), dtype=float).reshape(-1)
        _, caught = paddle_trace(ball, record.availability_ms, prediction,
                                 config.paddle_width, config.paddle_speed_per_fall,
                                 config.stimulus_dt_ms)
        outcomes.append(caught)
    if not outcomes:
        raise ValueError("permutation control needs held-out test records")
    return float(np.mean(outcomes))


def run_permutation_null(
    training,
    validation,
    test_records: list[TrialRecord],
    alpha_grid: tuple[float, ...],
    config: ExperimentConfig,
    observed_catch_rate: float,
    permutations: int = 100,
    seed: int | None = None,
) -> dict:
    if permutations < 20:
        raise ValueError("at least 20 permutations are required for an empirical null")
    rng = np.random.default_rng(config.seed if seed is None else seed)
    values = []
    for _ in range(permutations):
        shuffled_targets = permute_trajectory_targets(training.targets, training.groups, rng)
        model, alpha, validation_scores = select_ridge(
            training.neural, shuffled_targets, training.sample_weights,
            validation.neural, validation.targets, validation.sample_weights, alpha_grid,
        )
        values.append(_record_catch_rate(model, test_records, config))
    null = np.asarray(values, dtype=float)
    p_value = (1 + int(np.count_nonzero(null >= observed_catch_rate))) / (permutations + 1)
    return {
        "method": "trajectory-level training-label permutation; validation selects alpha",
        "permutations": permutations,
        "seed": int(config.seed if seed is None else seed),
        "observed_neural_catch_rate": float(observed_catch_rate),
        "null_mean_catch_rate": float(null.mean()),
        "null_95_interval": [float(x) for x in np.quantile(null, [0.025, 0.975])],
        "empirical_one_sided_p": float(p_value),
        "null_catch_rates": null.tolist(),
        "interpretation": "All frames and repeats of a ball share one shuffled target; test records are never permuted.",
    }
