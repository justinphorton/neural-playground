"""Resumable trajectory collection, frozen model fitting, and held-out runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .experiment import generate_accepted_balls, make_trial_trace, trace_to_record
from .pipeline import build_examples, load_readouts, save_readouts, train_readouts
from .readouts import pulse_policy_version
from .records import (
    ExperimentConfig,
    stable_seed,
    trial_checkpoint_matches,
)
from .reservoir import ReservoirRunner, build_reservoir
from .manifests import write_reservoir_manifest as _write_manifest
from .storage import ensure_stage_directory, trial_record_path
from .trial_store import load_records


def collect_slot(
    run_dir: Path,
    config: ExperimentConfig,
    slot: int,
    split: str,
    count: int,
    residual_range: tuple[float, float],
    repeats: int | None = None,
    seed: int | None = None,
    progress=None,
    *,
    output_dir: Path | None = None,
    opposite_only: bool = False,
    readout_id: str | None = None,
) -> list[Path]:
    repeats = config.noise_repeats if repeats is None else repeats
    if repeats < 1:
        raise ValueError("noise repeats must be positive")
    generation_stats = {}
    balls = generate_accepted_balls(
        config, residual_range, count, split,
        config.seed if seed is None else seed,
        opposite_only=opposite_only,
        stats=generation_stats,
    )
    run_dir = Path(run_dir)
    output = Path(output_dir or run_dir)
    (output / "metrics").mkdir(parents=True, exist_ok=True)
    (output / "metrics" / f"generation_{split}_slot{slot}.json").write_text(
        json.dumps(generation_stats, indent=2, sort_keys=True) + "\n"
    )
    reservoir = build_reservoir(config, slot)
    try:
        _write_manifest(run_dir, reservoir, config)
        runner = ReservoirRunner(reservoir, config)
        readouts = None
        if readout_id is not None:
            readouts = load_readouts(
                run_dir / "models" / f"readouts_slot{slot}.joblib",
                config.config_hash, reservoir.reservoir_hash,
            )
            if readouts["readout_id"] != readout_id:
                raise ValueError("readout changed after the evaluation plan was frozen")
        output_paths = []
        completed = 0
        for ball, entries in balls:
            for repeat in range(repeats):
                path = trial_record_path(output / "raw_trials", ball.trajectory_id, slot, repeat)
                if path.exists():
                    if not trial_checkpoint_matches(path, config.config_hash,
                                                    reservoir.reservoir_hash, readout_id):
                        raise RuntimeError(f"stale or incompatible checkpoint at {path}; use a new run")
                    output_paths.append(path)
                    completed += 1
                    continue
                noise_seed = stable_seed("trial", ball.trajectory_id, repeat, base_seed=config.seed)
                trace = make_trial_trace(ball, entries, reservoir, runner, config, noise_seed)
                record = trace_to_record(trace, config, slot, repeat, readouts)
                record.save(path)
                with (run_dir / "logs" / f"slot{slot}_{split}.jsonl").open("a") as stream:
                    stream.write(json.dumps({
                        "trajectory_id": ball.trajectory_id, "repeat": repeat,
                        "noise_seed": noise_seed, "file": path.name,
                        "reservoir_hash": reservoir.reservoir_hash,
                        "spikes": len(trace.simulation.spike_times_ms),
                    }) + "\n")
                output_paths.append(path)
                completed += 1
                if progress:
                    progress(completed, len(balls) * repeats, ball.trajectory_id)
        return output_paths
    finally:
        reservoir.close()


def fit_slot_readouts(
    run_dir: Path,
    config: ExperimentConfig,
    slot: int,
    residual_range: tuple[float, float],
) -> dict:
    """Fit the two Ridge readouts and freeze the training-mean control."""
    pulse_cutoff_policy = "strictly_before"
    training = load_records(run_dir, slot, "train")
    held_out = load_records(run_dir, slot, "validation")
    manifest = json.loads((Path(run_dir) / "reservoirs" / f"slot{slot}.json").read_text())
    grid_ids = tuple(int(row[0]) for row in manifest["grid"])
    train = build_examples(training, grid_ids,
        config.stimulus_dt_ms, config.warmup_ms, config.fall_ms, split="train",
        pulse_cutoff_policy=pulse_cutoff_policy)
    validation = build_examples(held_out, grid_ids,
        config.stimulus_dt_ms, config.warmup_ms, config.fall_ms, split="validation",
        pulse_cutoff_policy=pulse_cutoff_policy)
    if set(train.groups) & set(validation.groups):
        raise ValueError("train and validation share trajectory IDs")
    landings = {record.trajectory_id: record.landing_x for record in training}
    training_mean = float(np.mean(list(landings.values())))
    model = train_readouts(train, validation, config.regularization_grid)
    model_path = Path(run_dir) / "models" / f"readouts_slot{slot}.joblib"
    readout_id = save_readouts(
        model_path, model.neural, model.raw, config.config_hash,
        manifest["reservoir_hash"], model.neural.n_features_in_, grid_ids,
        neural_alpha=model.neural_alpha, raw_alpha=model.raw_alpha,
        neural_validation_scores=model.neural_validation_scores,
        raw_validation_scores=model.raw_validation_scores,
        residual_range=list(residual_range),
        constant_training_mean=training_mean,
        recording_channels=manifest.get("recording_channels"),
        pulse_cutoff_policy=pulse_cutoff_policy,
        decision_policy_version=pulse_policy_version(pulse_cutoff_policy),
    )
    return {
        "slot": slot,
        "config_hash": config.config_hash,
        "reservoir_hash": manifest["reservoir_hash"],
        "neural_alpha": model.neural_alpha,
        "raw_alpha": model.raw_alpha,
        "neural_validation_mse": model.neural_validation_scores,
        "raw_validation_mse": model.raw_validation_scores,
        "training_rows": len(train.targets),
        "validation_rows": len(validation.targets),
        "training_trajectories": int(len(np.unique(train.groups))),
        "validation_trajectories": int(len(np.unique(validation.groups))),
        "model_path": str(model_path),
        "readout_id": readout_id,
        "constant_training_mean": training_mean,
        "pulse_cutoff_policy": pulse_cutoff_policy,
        "decision_policy_version": pulse_policy_version(pulse_cutoff_policy),
    }


def run_training_stage(
    root: Path,
    config: ExperimentConfig,
    stage: str = "pilot",
    progress=None,
) -> dict:
    """Collect train/validation trajectories and freeze three readout pairs."""
    if stage not in {"pilot", "final"}:
        raise ValueError("training stage must be 'pilot' or 'final'")
    stage_dir = ensure_stage_directory(root, stage)
    if (stage_dir / "metrics" / "evaluation_plan.json").exists():
        raise ValueError("readouts are frozen; use evaluate to resume or start a new run")
    train_count, validation_count, _ = config.pilot_counts if stage == "pilot" else config.final_counts
    residual = tuple(config.selected_residual_range)
    models = []
    assignments: dict[str, list[str]] = {"train": [], "validation": []}
    for split, count in (("train", train_count), ("validation", validation_count)):
        split_seed = stable_seed(stage, split, "trajectories", base_seed=config.seed)
        for slot in range(config.reservoir_count):
            collect_slot(
                stage_dir, config, slot, split, count, residual,
                seed=split_seed,
                progress=progress,
            )
        assignments[split] = sorted({record.trajectory_id for record in load_records(stage_dir, 0, split)})
    for slot in range(config.reservoir_count):
        models.append(fit_slot_readouts(stage_dir, config, slot, residual))
    result = {
        "stage": stage,
        "config_hash": config.config_hash,
        "residual_range": list(residual),
        "models": models,
        "split_trajectory_counts": {name: len(ids) for name, ids in assignments.items()},
    }
    (stage_dir / "metrics" / "training.json").write_text(json.dumps(result, indent=2) + "\n")
    (stage_dir / "metrics" / "split_assignments.json").write_text(
        json.dumps(assignments, indent=2, sort_keys=True) + "\n"
    )
    with (Path(root) / "logs" / "observations.jsonl").open("a") as stream:
        stream.write(json.dumps({"phase": "training", "stage": stage,
                                 "config_hash": config.config_hash,
                                 "residual_range": list(residual),
                                 "models": models}, sort_keys=True) + "\n")
    return result


def run_pilot(root: Path, config: ExperimentConfig, progress=None) -> dict:
    stage = Path(root) / "pilot"
    if (stage / "metrics" / "evaluation_plan.json").exists():
        training = json.loads((stage / "metrics" / "training.json").read_text())
    else:
        training = run_training_stage(root, config, "pilot", progress)
    from .evaluation_run import run_evaluation_stage
    evaluation = run_evaluation_stage(root, config, "pilot", minimum_opposite=0, progress=progress)
    return {"training": training, "evaluation": evaluation}
