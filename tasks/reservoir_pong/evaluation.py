"""Paired, trajectory-level Pong metrics and scientific success gates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .physics import reflected_position
from .records import TrialRecord


def opposite_side_mask(
    x0: np.ndarray,
    vx: np.ndarray,
    fall_ms: np.ndarray,
    commit_fraction: float = 0.5,
) -> np.ndarray:
    x0, vx, fall = np.asarray(x0, dtype=float), np.asarray(vx, dtype=float), np.asarray(fall_ms, dtype=float)
    if not (x0.shape == vx.shape == fall.shape):
        raise ValueError("trajectory parameter arrays must have matching shapes")
    committed = np.asarray([reflected_position(a + b * commit_fraction) for a, b in zip(x0, vx)])
    landed = np.asarray([reflected_position(a + b) for a, b in zip(x0, vx)])
    return (committed < 0.5) != (landed < 0.5)


def count_opposite_trajectory_ids(
    records: list[TrialRecord], commit_fraction: float = 0.5,
) -> int:
    """Count unique opposite-side trajectories in a set of saved records."""
    if not records:
        return 0
    opposite = opposite_side_mask(
        np.asarray([float(record.metadata["x0"]) for record in records]),
        np.asarray([float(record.metadata["vx"]) for record in records]),
        np.asarray([float(record.metadata["fall_ms"]) for record in records]),
        commit_fraction,
    )
    return int(len({record.trajectory_id for record, keep
                    in zip(records, opposite) if keep}))


def paired_trajectory_bootstrap(
    neural_caught: np.ndarray,
    baseline_caught: np.ndarray,
    trajectory_ids: np.ndarray,
    seed: int = 123,
    repeats: int = 5000,
) -> tuple[float, float, float]:
    neural, baseline, groups = np.asarray(neural_caught, dtype=float), np.asarray(baseline_caught, dtype=float), np.asarray(trajectory_ids)
    if not (neural.shape == baseline.shape == groups.shape) or not len(groups):
        raise ValueError("paired outcomes and trajectory groups must be non-empty and aligned")
    unique = np.unique(groups)
    gains = np.asarray([np.mean(neural[groups == group] - baseline[groups == group]) for group in unique])
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(gains), size=(repeats, len(gains)))
    estimates = gains[samples].mean(axis=1)
    return float(gains.mean()), float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def success_decision(metrics: dict) -> tuple[bool, list[str]]:
    reasons = []
    if int(metrics.get("opposite_trajectory_count", 0)) < 100:
        reasons.append("fewer than 100 opposite-side test trajectories")
    if float(metrics.get("mean_catch_gain", -1.0)) < 0.05:
        reasons.append("mean opposite-side catch gain is below 5 percentage points")
    if float(metrics.get("gain_ci_lower", -1.0)) <= 0:
        reasons.append("paired 95% interval does not establish a positive gain")
    gains = metrics.get("reservoir_gains", [])
    if len(gains) != 3 or any(float(value) <= 0 for value in gains):
        reasons.append("one or more reservoir subpatches do not show positive improvement")
    if any(int(value) < 100 for value in metrics.get("opposite_trajectories_by_reservoir", {}).values()):
        reasons.append("one or more subpatches have fewer than 100 opposite-side test trajectories")
    if float(metrics.get("neural_landing_mae", float("inf"))) > float(metrics.get("raw_landing_mae", -float("inf"))):
        reasons.append("overall landing-position error is worse than the raw-pulse baseline")
    return not reasons, reasons


def normalized_remaining_error_slices(
    records: list[TrialRecord],
    bins: tuple[float, ...] = (0.5, 0.625, 0.75, 0.875, 1.0),
    bootstrap_repeats: int = 2000,
) -> dict[str, dict]:
    """Report unclipped prediction error by remaining fraction, clustered by ball."""
    edges = np.asarray(bins, dtype=float)
    if (edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0)
            or edges[0] < 0 or edges[-1] > 1):
        raise ValueError("remaining-distance bins must be increasing fractions in [0, 1]")
    rows: dict[int, dict[str, list[float]]] = {
        index: {} for index in range(len(edges) - 1)
    }
    decision_counts = {index: 0 for index in rows}
    record_counts = {index: 0 for index in rows}
    for record in records:
        times = np.asarray(record.availability_ms, dtype=float)
        fall = float(record.metadata["fall_ms"])
        if fall <= 0 or len(times) != len(record.neural_predictions) or len(times) != len(record.raw_predictions):
            raise ValueError("prediction windows must align with a positive fall duration")
        remaining = (fall - times) / fall
        neural_error = np.abs(np.asarray(record.neural_predictions, dtype=float)
                              - float(record.landing_x))
        raw_error = np.abs(np.asarray(record.raw_predictions, dtype=float)
                           - float(record.landing_x))
        for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
            selected = ((remaining >= lower - 1e-9)
                        & ((remaining < upper - 1e-9)
                           | ((index == len(edges) - 2)
                              & (remaining <= upper + 1e-9))))
            if not np.any(selected):
                continue
            key = str(record.trajectory_id)
            entry = rows[index].setdefault(key, [0.0, 0.0, 0.0])
            entry[0] += float(neural_error[selected].sum())
            entry[1] += float(raw_error[selected].sum())
            entry[2] += int(np.count_nonzero(selected))
            decision_counts[index] += int(np.count_nonzero(selected))
            record_counts[index] += 1

    result = {}
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        grouped = rows[index]
        if not grouped:
            result[f"{lower:g}_{upper:g}"] = {
                "remaining_fraction": [float(lower), float(upper)],
                "trajectories": 0, "records": 0, "decisions": 0,
                "neural_mae": None, "raw_mae": None,
                "raw_minus_neural_mae": None,
                "raw_minus_neural_mae_ci_95": [None, None],
            }
            continue
        trajectory_ids = np.asarray(sorted(grouped))
        neural = np.asarray([grouped[key][0] / grouped[key][2]
                             for key in trajectory_ids])
        raw = np.asarray([grouped[key][1] / grouped[key][2]
                          for key in trajectory_ids])
        paired = paired_trajectory_bootstrap(
            raw, neural, trajectory_ids, seed=1310 + index,
            repeats=bootstrap_repeats,
        )
        result[f"{lower:g}_{upper:g}"] = {
            "remaining_fraction": [float(lower), float(upper)],
            "trajectories": int(len(trajectory_ids)),
            "records": int(record_counts[index]),
            "decisions": int(decision_counts[index]),
            "neural_mae": float(neural.mean()),
            "raw_mae": float(raw.mean()),
            "raw_minus_neural_mae": paired[0],
            "raw_minus_neural_mae_ci_95": [paired[1], paired[2]],
        }
    return result


def merge_opposite_analysis(
    metrics: dict,
    analysis_records: list[TrialRecord],
    commit_fraction: float = 0.5,
    bootstrap_repeats: int = 5000,
) -> dict:
    """Add opposite-side results without changing first-block overall metrics."""
    if not analysis_records:
        raise ValueError("opposite-side analysis requires saved trial records")
    slots = np.asarray([int(record.metadata["reservoir_slot"])
                        for record in analysis_records])
    x0 = np.asarray([float(record.metadata["x0"]) for record in analysis_records])
    vx = np.asarray([float(record.metadata["vx"]) for record in analysis_records])
    fall = np.asarray([float(record.metadata["fall_ms"]) for record in analysis_records])
    groups = np.asarray([record.trajectory_id for record in analysis_records])
    opposite = opposite_side_mask(x0, vx, fall, commit_fraction)
    neural = np.asarray([record.caught for record in analysis_records], dtype=float)
    raw = np.asarray([record.raw_caught for record in analysis_records], dtype=float)
    opposite_groups = groups[opposite]
    overall = paired_trajectory_bootstrap(
        neural[opposite], raw[opposite], opposite_groups,
        seed=172, repeats=bootstrap_repeats,
    ) if np.any(opposite) else (0.0, -1.0, 1.0)
    gains, by_slot = [], {}
    reservoirs = dict(metrics.get("reservoirs", {}))
    for slot in sorted(np.unique(slots)):
        mask = (slots == slot) & opposite
        unique = np.unique(groups[mask])
        by_slot[str(int(slot))] = int(len(unique))
        if np.any(mask):
            gain = paired_trajectory_bootstrap(
                neural[mask], raw[mask], groups[mask], seed=300 + int(slot),
                repeats=bootstrap_repeats,
            )
        else:
            gain = (0.0, -1.0, 1.0)
        gains.append(gain[0])
        item = dict(reservoirs.get(str(int(slot)), {}))
        item["opposite_n"] = int(np.count_nonzero(mask))
        item["opposite_trajectories"] = int(len(unique))
        item["opposite_catch_gain"] = gain[0]
        item["opposite_gain_ci_95"] = [gain[1], gain[2]]
        reservoirs[str(int(slot))] = item
    metrics.update({
        "opposite_count": int(np.count_nonzero(opposite)),
        "opposite_trajectory_count": int(len(np.unique(opposite_groups))),
        "opposite_trajectories_by_reservoir": by_slot,
        "mean_catch_gain": overall[0],
        "gain_ci_lower": overall[1],
        "gain_ci_upper": overall[2],
        "reservoir_gains": gains,
        "reservoirs": reservoirs,
    })
    success, reasons = success_decision(metrics)
    metrics["scientific_success"] = success
    metrics["success_reasons"] = reasons
    return metrics


def summarize_records(records: list[TrialRecord], commit_fraction: float = 0.5,
                      bootstrap_repeats: int = 5000) -> dict:
    if not records:
        raise ValueError("no held-out trial records were supplied")
    slot = np.asarray([int(r.metadata["reservoir_slot"]) for r in records])
    groups = np.asarray([r.trajectory_id for r in records])
    neural_caught = np.asarray([r.caught for r in records], dtype=float)
    raw_caught = np.asarray([r.raw_caught for r in records], dtype=float)
    follow_caught = np.asarray([r.follow_caught for r in records], dtype=float)
    neural_mae = np.asarray([abs(float(r.neural_predictions[-1]) - r.landing_x) for r in records])
    raw_mae = np.asarray([abs(float(r.raw_predictions[-1]) - r.landing_x) for r in records])

    overall_estimate, overall_low, overall_high = paired_trajectory_bootstrap(
        neural_caught, raw_caught, groups, seed=171, repeats=bootstrap_repeats,
    )
    per_reservoir = {}
    for index in sorted(np.unique(slot)):
        mask = slot == index
        per_reservoir[str(int(index))] = {
            "n": int(mask.sum()),
            "catch_rate": float(neural_caught[mask].mean()),
            "raw_catch_rate": float(raw_caught[mask].mean()),
            "follow_catch_rate": float(follow_caught[mask].mean()),
            "landing_mae": float(neural_mae[mask].mean()),
            "raw_landing_mae": float(raw_mae[mask].mean()),
        }
    result = {
        "n_trials": len(records),
        "n_trajectories": int(len(np.unique(groups))),
        "neural_catch_rate": float(neural_caught.mean()),
        "raw_catch_rate": float(raw_caught.mean()),
        "follow_catch_rate": float(follow_caught.mean()),
        "neural_landing_mae": float(neural_mae.mean()),
        "raw_landing_mae": float(raw_mae.mean()),
        "catch_gain": overall_estimate,
        "catch_gain_ci_95": [overall_low, overall_high],
        "reservoirs": per_reservoir,
    }
    result["normalized_remaining_error_slices"] = normalized_remaining_error_slices(
        records, bootstrap_repeats=bootstrap_repeats,
    )
    return merge_opposite_analysis(result, records, commit_fraction, bootstrap_repeats)


def write_metrics(metrics: dict, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
