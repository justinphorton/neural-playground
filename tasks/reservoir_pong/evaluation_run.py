"""Held-out trajectory evaluation and frozen-controller scoring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .evaluation import merge_opposite_analysis, opposite_side_mask, summarize_records
from .evaluation_report import write_evaluation_artifacts
from .pipeline import build_examples, load_readouts
from .records import ExperimentConfig, load_trial, stable_seed
from .storage import ensure_stage_directory, save_json_atomic
from .training import collect_slot
from .trial_store import load_records


def _opposite_ids(records: list, commit_fraction: float) -> set[str]:
    if not records:
        return set()
    mask = opposite_side_mask(
        np.asarray([float(record.metadata["x0"]) for record in records]),
        np.asarray([float(record.metadata["vx"]) for record in records]),
        np.asarray([float(record.metadata["fall_ms"]) for record in records]),
        commit_fraction,
    )
    return {record.trajectory_id for record, keep in zip(records, mask) if keep}


def _plan_shared_opposite_supplement(
    base_by_slot: dict[int, list],
    commit_fraction: float,
    minimum_opposite: int,
    expected_trajectory_count: int,
) -> tuple[set[str], int]:
    """Require the primary block and its opposite-side slice to match by ball ID."""
    base_ids = {slot: {record.trajectory_id for record in records}
                for slot, records in base_by_slot.items()}
    if any(len(ids) != expected_trajectory_count for ids in base_ids.values()):
        raise ValueError("primary test block has an unexpected trajectory count")
    if any(ids != base_ids[0] for ids in base_ids.values()):
        raise ValueError("primary test trajectories differ across subreservoirs")
    opposite_by_slot = {
        slot: _opposite_ids(records, commit_fraction)
        for slot, records in base_by_slot.items()
    }
    if any(ids != opposite_by_slot[0] for ids in opposite_by_slot.values()):
        raise ValueError("opposite-side primary test cases differ across subreservoirs")
    missing = max(0, minimum_opposite - len(opposite_by_slot[0]))
    return opposite_by_slot[0], missing


def _validate_shared_opposite_supplement(
    records_by_slot: dict[int, list],
    base_opposite_ids: set[str],
    commit_fraction: float,
    minimum_opposite: int,
) -> dict[int, set[str]]:
    supplement_ids = {
        slot: {record.trajectory_id for record in records}
        for slot, records in records_by_slot.items()
    }
    if any(ids != supplement_ids[0] for ids in supplement_ids.values()):
        raise ValueError("opposite-side supplemental trajectories differ across subreservoirs")
    for slot, records in records_by_slot.items():
        ids = supplement_ids[slot]
        if _opposite_ids(records, commit_fraction) != ids:
            raise ValueError("opposite-side supplement contains a non-opposite trajectory")
        if ids & base_opposite_ids:
            raise ValueError("opposite-side supplement overlaps the primary test block")
        if len(ids | base_opposite_ids) < minimum_opposite:
            raise ValueError("shared opposite-side analysis block is below its target count")
    return supplement_ids


def run_evaluation_stage(
    root: Path,
    config: ExperimentConfig,
    stage: str = "final",
    minimum_opposite: int | None = None,
    progress=None,
    test_seed: int | None = None,
) -> dict:
    """Collect the fixed overall block and a separate opposite-side supplement."""
    stage_dir = ensure_stage_directory(root, stage)
    _, _, test_count = config.pilot_counts if stage == "pilot" else config.final_counts
    if minimum_opposite is None:
        minimum_opposite = 0 if stage == "pilot" else 100
    split_seed = (stable_seed(stage, "test", "trajectories", base_seed=config.seed)
                  if test_seed is None else int(test_seed))
    evaluation_plan_path = stage_dir / "metrics" / "evaluation_plan.json"
    model_meta = {
        str(slot): json.loads((stage_dir / "models" / f"readouts_slot{slot}.joblib.json").read_text())
        for slot in range(config.reservoir_count)
    }
    readout_ids = {slot: model["readout_id"] for slot, model in model_meta.items()}
    plan = {
        "protocol": "ridge-only-v1",
        "frozen_readout_ids": readout_ids,
        "stage": stage,
        "config_hash": config.config_hash,
        "test_seed": split_seed,
        "trajectory_count": test_count,
        "noise_repeats": config.noise_repeats,
        "minimum_opposite_supplement_per_reservoir": minimum_opposite,
    }
    if evaluation_plan_path.is_file():
        existing_plan = json.loads(evaluation_plan_path.read_text())
        if existing_plan.get("protocol") != plan["protocol"]:
            raise ValueError("legacy evaluation is saved for replay; use a new run to evaluate")
        if existing_plan != plan:
            raise ValueError("held-out evaluation plan is frozen; use a new run for different settings")
    else:
        save_json_atomic(evaluation_plan_path, plan)
    def collect_test(slot, count, seed, *, opposite_only=False):
        paths = collect_slot(
            stage_dir, config, slot, "test", count, tuple(config.selected_residual_range),
            seed=seed, progress=progress, opposite_only=opposite_only,
            output_dir=stage_dir / "supplemental" if opposite_only else stage_dir,
            readout_id=readout_ids[str(slot)],
        )
        return [load_trial(path) for path in paths]

    base_by_slot = {slot: collect_test(slot, test_count, split_seed)
                    for slot in range(config.reservoir_count)}
    records = [record for rows in base_by_slot.values() for record in rows]
    metrics = summarize_records(records, config.commit_fraction, config.bootstrap_repeats)
    # From center, a constant target is limited by paddle width and total travel.
    half_width = config.paddle_width / 2
    targets = np.asarray([
        model_meta[str(r.metadata["reservoir_slot"])]["metadata"]["constant_training_mean"]
        for r in records
    ])
    travel = min(config.paddle_speed_per_fall, 0.5 - half_width)
    impact = np.clip(targets, 0.5 - travel, 0.5 + travel)
    metrics["constant_catch_rate"] = float(np.mean(
        np.abs(impact - [r.landing_x for r in records]) <= half_width
    ))
    if stage == "final":
        from .evaluation_controls import run_fresh_test_controls
        metrics["matched_test_controls"] = run_fresh_test_controls(
            stage_dir, config, records, progress=progress,
        )
    supplemental_records = []
    supplemental_counts = {}
    shared_supplement_ids: set[str] = set()
    shared_base_opposite_ids: set[str] = set()
    supplement_seed = None
    if stage == "final" and minimum_opposite:
        base_opposite_ids, missing = _plan_shared_opposite_supplement(
            base_by_slot, config.commit_fraction, minimum_opposite, test_count,
        )
        shared_base_opposite_ids = base_opposite_ids
        if missing:
            supplement_seed = stable_seed(
                stage, "test", "opposite-shared", split_seed, minimum_opposite,
                base_seed=config.seed,
            )
        supplements_by_slot = {
            slot: collect_test(slot, missing, supplement_seed, opposite_only=True) if missing else []
            for slot in base_by_slot
        }
        supplement_ids_by_slot = _validate_shared_opposite_supplement(
            supplements_by_slot, base_opposite_ids, config.commit_fraction,
            minimum_opposite,
        )
        shared_supplement_ids = supplement_ids_by_slot[0]
        for slot, current_supplement in supplements_by_slot.items():
            opposite_count = len(supplement_ids_by_slot[slot])
            supplemental_records.extend(current_supplement)
            supplemental_counts[str(slot)] = opposite_count
        supplement_plan = {
            "config_hash": config.config_hash,
            "test_seed": split_seed,
            "minimum_opposite_per_subreservoir": minimum_opposite,
            "primary_test_trajectory_ids": sorted({
                record.trajectory_id
                for rows in base_by_slot.values() for record in rows
            }),
            "primary_opposite_trajectory_ids": sorted(shared_base_opposite_ids),
            "supplement_seed": supplement_seed,
            "shared_opposite_supplement_trajectory_ids": sorted(shared_supplement_ids),
            "overall_metrics_exclude_supplement": True,
        }
        supplement_plan_path = stage_dir / "metrics" / "opposite_supplement_plan.json"
        if supplement_plan_path.is_file():
            if json.loads(supplement_plan_path.read_text()) != supplement_plan:
                raise ValueError("shared opposite-side supplement plan is frozen")
        else:
            save_json_atomic(supplement_plan_path, supplement_plan)
    if supplemental_records:
        metrics = merge_opposite_analysis(
            metrics, records + supplemental_records, config.commit_fraction,
            config.bootstrap_repeats,
        )
    metrics.update({
        "stage": stage,
        "config_hash": config.config_hash,
        "reservoir_hashes": {
            str(slot): json.loads((stage_dir / "reservoirs" / f"slot{slot}.json").read_text())[
                "reservoir_hash"
            ] for slot in range(config.reservoir_count)
        },
        "readout_ids": readout_ids,
        "overall_test_trajectory_count": len({record.trajectory_id for record in records}),
        "supplemental_opposite_trajectory_counts": supplemental_counts,
        "shared_supplemental_opposite_trajectory_ids": sorted(shared_supplement_ids),
        "overall_test_excludes_supplemental": True,
        "test_seed": split_seed,
        "test_block_noise_repeats": config.noise_repeats,
    })
    if stage == "final":
        from .permutation import run_permutation_null
        permutation_results = {}
        for slot in range(config.reservoir_count):
            slot_manifest = json.loads(
                (stage_dir / "reservoirs" / f"slot{slot}.json").read_text()
            )
            grid_ids = tuple(int(row[0]) for row in slot_manifest["grid"])
            readouts = load_readouts(
                stage_dir / "models" / f"readouts_slot{slot}.joblib",
                config.config_hash, slot_manifest["reservoir_hash"],
            )
            cutoff_policy = readouts.get("metadata", {}).get(
                "pulse_cutoff_policy", "at_or_before",
            )
            training_examples = build_examples(
                load_records(stage_dir, slot, "train"), grid_ids, config.stimulus_dt_ms, config.warmup_ms,
                config.fall_ms, split="train", pulse_cutoff_policy=cutoff_policy,
            )
            validation_examples = build_examples(
                load_records(stage_dir, slot, "validation"), grid_ids, config.stimulus_dt_ms, config.warmup_ms,
                config.fall_ms, split="validation", pulse_cutoff_policy=cutoff_policy,
            )
            permutation_results[str(slot)] = run_permutation_null(
                training_examples, validation_examples, base_by_slot[slot],
                config.regularization_grid, config,
                metrics["reservoirs"][str(slot)]["catch_rate"],
                permutations=100,
                seed=stable_seed("permutation", slot, base_seed=config.seed),
            )
        metrics["permutation_controls"] = permutation_results
    write_evaluation_artifacts(stage_dir, stage, metrics)
    with (Path(root) / "logs" / "observations.jsonl").open("a") as stream:
        stream.write(json.dumps({"phase": "evaluation", "stage": stage,
                                 "scientific_success": metrics["scientific_success"],
                                 "metrics": metrics}, sort_keys=True) + "\n")
    return metrics
