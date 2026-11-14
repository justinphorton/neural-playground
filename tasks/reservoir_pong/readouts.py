"""Leakage-safe continuous readouts and pulse-history baselines."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .physics import BallSpec, landing_position, trajectory_position
from .records import stable_seed
from .stimulus import PulseEvent

STRICT_PULSE_POLICY_VERSION = "strict-pulse-prefix-v2"
LEGACY_PULSE_POLICY_VERSION = "legacy-inclusive-v1"


def pulse_policy_version(policy: str) -> str:
    if policy == "strictly_before":
        return STRICT_PULSE_POLICY_VERSION
    if policy == "at_or_before":
        return LEGACY_PULSE_POLICY_VERSION
    raise ValueError("cutoff policy must be 'strictly_before' or 'at_or_before'")


def build_prefix_design(
    events: list[PulseEvent],
    availability_ms: list[float] | np.ndarray,
    electrode_ids: tuple[int, ...] | list[int],
    duration_ms: float,
    dt_ms: float,
    cutoff_policy: str = "strictly_before",
) -> csr_matrix:
    """Sparse electrode-by-time one-hot rows for each available pulse prefix."""
    if duration_ms <= 0 or dt_ms <= 0:
        raise ValueError("duration and time step must be positive")
    if cutoff_policy not in {"strictly_before", "at_or_before"}:
        raise ValueError("cutoff_policy must be 'strictly_before' or 'at_or_before'")
    channel_index = {int(ch): i for i, ch in enumerate(electrode_ids)}
    if len(channel_index) != len(electrode_ids):
        raise ValueError("electrode IDs must be unique")
    n_bins = int(np.ceil(duration_ms / dt_ms - 1e-12))
    row_idx, col_idx = [], []
    times = np.asarray(availability_ms, dtype=float)
    if np.any(~np.isfinite(times)):
        raise ValueError("availability times must be finite")
    def tick(value: float, label: str) -> int:
        result = int(np.rint(value / dt_ms))
        if not np.isclose(value, result * dt_ms, rtol=0, atol=max(1e-8, dt_ms * 1e-7)):
            raise ValueError(f"{label} must lie on the stimulus tick grid")
        return result

    event_ticks = [tick(float(event.onset_ms), "pulse onset") for event in events]
    for row, decision_time in enumerate(times):
        decision_tick = tick(float(decision_time), "decision time")
        for event, onset_tick in zip(events, event_ticks):
            visible = (onset_tick < decision_tick if cutoff_policy == "strictly_before"
                       else onset_tick <= decision_tick)
            if not visible:
                break
            if event.electrode_id not in channel_index:
                raise ValueError(f"electrode {event.electrode_id} is outside the input grid")
            bin_index = onset_tick
            if not 0 <= bin_index < n_bins:
                raise ValueError("pulse onset lies outside raw-input baseline duration")
            row_idx.append(row)
            col_idx.append(bin_index * len(channel_index) + channel_index[event.electrode_id])
    return csr_matrix((np.ones(len(row_idx), dtype=np.float32), (row_idx, col_idx)),
                      shape=(len(times), n_bins * len(channel_index)))


def trajectory_sample_weights(group_ids: np.ndarray) -> np.ndarray:
    groups = np.asarray(group_ids)
    if groups.ndim != 1 or len(groups) == 0:
        raise ValueError("group_ids must be a non-empty vector")
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return weights * (len(weights) / len(unique))


def fit_ridge(x, y, alpha: float, sample_weight: np.ndarray | None = None):
    if alpha <= 0:
        raise ValueError("Ridge alpha must be positive")
    model = make_pipeline(StandardScaler(with_mean=not issparse(x)),
                          Ridge(alpha=alpha, solver="lsqr"))
    if sample_weight is None:
        model.fit(x, y)
    else:
        model.fit(x, y, ridge__sample_weight=np.asarray(sample_weight, dtype=float))
    return model


def select_ridge(
    x_train,
    y_train: np.ndarray,
    train_weights: np.ndarray,
    x_validation,
    y_validation: np.ndarray,
    validation_weights: np.ndarray,
    alpha_grid: tuple[float, ...],
) -> tuple[object, float, dict[float, float]]:
    scores = {}
    models = {}
    for alpha in alpha_grid:
        model = fit_ridge(x_train, y_train, alpha, train_weights)
        predictions = np.asarray(model.predict(x_validation)).reshape(-1)
        score = float(np.average((predictions - y_validation) ** 2, weights=validation_weights))
        scores[float(alpha)] = score
        models[float(alpha)] = model
    best = min(scores, key=lambda a: (scores[a], a))
    return models[best], best, scores


def legal_paddle_target(prediction: float, paddle_width: float = 0.12) -> float:
    """Clip an aim point to legal paddle-center coordinates, not ball coordinates."""
    if not 0 < paddle_width <= 1 or not np.isfinite(prediction):
        raise ValueError("prediction and paddle width are invalid")
    half_width = paddle_width / 2
    return float(np.clip(prediction, half_width, 1 - half_width))


def paddle_trace(
    ball: BallSpec,
    availability_ms: np.ndarray,
    targets: np.ndarray,
    paddle_width: float = 0.12,
    speed_per_fall: float = 0.8,
    integration_step_ms: float = 0.1,
) -> tuple[np.ndarray, bool]:
    """Simulate the speed-limited paddle and freeze its target at commitment."""
    times = np.asarray(availability_ms, dtype=float)
    predictions = np.asarray(targets, dtype=float)
    if (times.shape != predictions.shape or np.any(~np.isfinite(times))
            or np.any(~np.isfinite(predictions)) or np.any(np.diff(times) < 0)):
        raise ValueError("prediction times and values must be ordered equal-length vectors")
    if not 0 < paddle_width <= 1 or speed_per_fall <= 0 or integration_step_ms <= 0:
        raise ValueError("paddle geometry and speed are invalid")
    if np.any(times < -1e-9) or np.any(times > ball.fall_ms + 1e-9):
        raise ValueError("prediction times must lie within the ball fall")
    max_speed = speed_per_fall / ball.fall_ms
    commit = ball.commit_ms
    x_paddle = 0.5
    locked_target: float | None = None
    target: float | None = None
    next_prediction = 0
    sample_times = paddle_trace_times(ball.fall_ms, integration_step_ms)
    breakpoints = np.unique(np.concatenate((
        sample_times, times, np.asarray([commit, ball.fall_ms]),
    )))
    sample_positions: dict[float, float] = {}
    previous_time = 0.0
    for t in breakpoints:
        elapsed = float(t - previous_time)
        if elapsed > 0:
            desired = locked_target if locked_target is not None else target
            if desired is not None:
                max_delta = max_speed * elapsed
                x_paddle += float(np.clip(desired - x_paddle, -max_delta, max_delta))
                x_paddle = float(np.clip(x_paddle, paddle_width / 2,
                                         1 - paddle_width / 2))
        while next_prediction < len(times) and times[next_prediction] <= t + 1e-9:
            if times[next_prediction] <= commit + 1e-9 and locked_target is None:
                target = legal_paddle_target(predictions[next_prediction], paddle_width)
            next_prediction += 1
        if t >= commit - 1e-9 and locked_target is None:
            locked_target = target if target is not None else x_paddle
        if np.any(np.isclose(sample_times, t, rtol=0, atol=1e-9)):
            sample_positions[float(t)] = x_paddle
        previous_time = float(t)
    trace = np.asarray([sample_positions[float(t)] for t in sample_times], dtype=float)
    actual = float(landing_position(ball))
    caught = abs(trace[-1] - actual) <= paddle_width / 2
    return trace, caught


def paddle_trace_times(fall_ms: float, integration_step_ms: float = 0.1) -> np.ndarray:
    if fall_ms <= 0 or integration_step_ms <= 0:
        raise ValueError("fall and integration step must be positive")
    sample_count = int(np.floor(fall_ms / integration_step_ms + 1e-10))
    sample_times = np.arange(sample_count + 1, dtype=float) * integration_step_ms
    if sample_times[-1] < fall_ms - 1e-9:
        sample_times = np.append(sample_times, fall_ms)
    else:
        sample_times[-1] = fall_ms
    return sample_times


def follow_ball_targets(ball: BallSpec, availability_ms: np.ndarray) -> np.ndarray:
    return np.asarray([trajectory_position(ball, np.clip(t / ball.fall_ms, 0, 1))[0]
                       for t in np.asarray(availability_ms, dtype=float)], dtype=float)


def seeded_ball(
    trajectory_id: str,
    split: str,
    residual_range: tuple[float, float],
    fall_ms: float,
    seed: int,
    commit_fraction: float = 0.5,
) -> BallSpec:
    rng = np.random.default_rng(stable_seed(trajectory_id, split, base_seed=seed))
    x0 = float(rng.uniform(0.1, 0.9))
    residual = float(rng.uniform(*residual_range))
    sign = -1.0 if rng.integers(0, 2) == 0 else 1.0
    vx = sign * residual / (1.0 - commit_fraction)
    return BallSpec(trajectory_id, x0, vx, fall_ms, commit_fraction, split)
