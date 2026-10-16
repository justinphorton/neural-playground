"""Command-line orchestration for the fixed-reservoir experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset_manifest import build_training_manifest
from .experiment_config import ExperimentConfig
from .fixed_reservoir import (
    AnalysisStage,
    SimulationStage,
    FeatureStage,
    build_trial_specs,
    collect_trials_parallel,
    initialize_experiment,
    preflight_worker_hashes,
    TrialSpec,
)
from .feature_extraction import load_feature_dataset, save_feature_dataset
from .raw_trials import load_raw_trial
from .reservoir_artifact import ReservoirManifest
from .screening import build_screen_schedule


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    init = subparsers.add_parser("initialize", help="write config, manifest, audit, and canonical reservoir")
    init.add_argument("--root", type=Path, default=Path("experiments"))
    init.add_argument("--training-data", type=Path, default=Path(__file__).with_name("training_data"))
    init.add_argument("--name", default="fixed-c4-e4-memory")
    preflight = subparsers.add_parser("preflight", help="launch eight workers that only load the reservoir")
    preflight.add_argument("artifact", type=Path)
    preflight.add_argument("--workers", type=int, default=8)
    smoke = subparsers.add_parser("smoke", help="run one C4, one E4, and one zero-current trial")
    smoke.add_argument("experiment", type=Path)
    smoke.add_argument("--amplitude", type=float, default=1.5)
    screen = subparsers.add_parser("screen-schedule", help="write the preregistered 384-trial screen schedule")
    screen.add_argument("experiment", type=Path)
    screen.add_argument("--second-pass", type=float, nargs="*", default=())
    screen_run = subparsers.add_parser("screen-run", help="execute the saved amplitude-screen schedule")
    screen_run.add_argument("experiment", type=Path)
    screen_run.add_argument("--schedule", type=Path, default=None)
    screen_run.add_argument("--workers", type=int, default=None)
    collect = subparsers.add_parser("collect", help="collect an explicit full-data or screen schedule")
    collect.add_argument("experiment", type=Path)
    collect.add_argument("--amplitude", type=float, required=True)
    collect.add_argument("--repeats", type=int, default=None)
    collect.add_argument("--controls", action="store_true")
    collect.add_argument("--workers", type=int, default=None)
    extract = subparsers.add_parser("extract", help="extract neural-only features from saved raw trials")
    extract.add_argument("experiment", type=Path)
    extract.add_argument("--start-ms", type=float, default=5.0)
    extract.add_argument("--end-ms", type=float, default=55.0)
    extract.add_argument("--mode", default="fine_temporal", choices=("spike_count", "coarse_temporal", "fine_temporal"))
    train = subparsers.add_parser("train", help="fit the all-data development classifier from saved features")
    train.add_argument("experiment", type=Path)
    train.add_argument("--features", type=Path, required=True)
    train.add_argument("--amplitude", type=float, required=True)
    train.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.command == "initialize":
        experiment, rows, reservoir = initialize_experiment(args.root, args.training_data, ExperimentConfig(), args.name)
        print(json.dumps({"experiment": str(experiment), "recordings": len(rows), "reservoir_hash": reservoir.reservoir_hash}, indent=2))
        return 0
    if args.command == "preflight":
        hashes = preflight_worker_hashes(args.artifact, args.workers)
        print(json.dumps({"workers": len(hashes), "reservoir_hash": hashes[0]}, indent=2))
        return 0
    if args.command == "smoke":
        experiment = args.experiment
        config = ExperimentConfig.read(experiment / "config.json")
        rows = build_training_manifest(Path(__file__).with_name("training_data"))
        chosen = rows[:1] + [row for row in rows if row.note == "E4"][:1]
        manifest = ReservoirManifest.read(experiment / "reservoir" / "reservoir_manifest.json")
        note_specs = build_trial_specs(chosen, manifest.reservoir_hash, args.amplitude, 1)
        control_specs = [spec for spec in build_trial_specs(chosen, manifest.reservoir_hash, 0.0, 1, include_controls=True) if spec.is_control]
        specs = note_specs + control_specs[:1]
        stage = SimulationStage(config, experiment, experiment / "reservoir")
        try:
            for spec in specs:
                trial = stage.run_trial(spec, experiment / "raw_trials" / f"{spec.trial_id}.npz")
                print(spec.trial_id, len(trial.event_times_ms), "events")
        finally:
            stage.close()
        return 0
    if args.command == "screen-schedule":
        config = ExperimentConfig.read(args.experiment / "config.json")
        rows = build_training_manifest(Path(__file__).with_name("training_data"))
        manifest = ReservoirManifest.read(args.experiment / "reservoir" / "reservoir_manifest.json")
        specs, selection = build_screen_schedule(rows, manifest.reservoir_hash, config.screening_amplitudes_ua, args.second_pass)
        payload = {"selection": {"recording_ids": list(selection.recording_ids), "condition_groups": list(selection.condition_groups)}, "trials": [spec.__dict__ for spec in specs]}
        path = args.experiment / "metrics" / "screen_schedule.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(path)
        return 0
    if args.command == "screen-run":
        experiment = args.experiment
        config = ExperimentConfig.read(experiment / "config.json")
        reservoir = ReservoirManifest.read(experiment / "reservoir" / "reservoir_manifest.json")
        schedule_path = args.schedule or experiment / "metrics" / "screen_schedule.json"
        payload = json.loads(schedule_path.read_text())
        specs = [TrialSpec(**item) for item in payload.get("trials", [])]
        if any(spec.expected_reservoir_hash != reservoir.reservoir_hash for spec in specs):
            raise ValueError("screen schedule references a different reservoir hash")
        results = collect_trials_parallel(specs, config, experiment, experiment / "reservoir", args.workers)
        print(json.dumps({"schedule": str(schedule_path), "trials": len(specs), "results": results}, indent=2))
        return 0
    if args.command == "collect":
        config = ExperimentConfig.read(args.experiment / "config.json")
        rows = build_training_manifest(Path(__file__).with_name("training_data"))
        manifest = ReservoirManifest.read(args.experiment / "reservoir" / "reservoir_manifest.json")
        specs = build_trial_specs(rows, manifest.reservoir_hash, args.amplitude, args.repeats or config.full_dataset_repeats, include_controls=args.controls)
        print(collect_trials_parallel(specs, config, args.experiment, args.experiment / "reservoir", args.workers))
        return 0
    if args.command == "extract":
        config = ExperimentConfig.read(args.experiment / "config.json")
        reservoir = ReservoirManifest.read(args.experiment / "reservoir" / "reservoir_manifest.json")
        paths = sorted((args.experiment / "raw_trials").glob("*.npz"))
        trials = [load_raw_trial(path) for path in paths]
        dataset = FeatureStage(config.recording_electrodes or reservoir.recording_electrodes).from_trials(trials, (args.start_ms, args.end_ms), args.mode)
        output = args.experiment / "features" / f"{args.mode}_{args.start_ms:g}_{args.end_ms:g}.npz"
        save_feature_dataset(dataset, output)
        print(output)
        return 0
    if args.command == "train":
        config = ExperimentConfig.read(args.experiment / "config.json")
        reservoir = ReservoirManifest.read(args.experiment / "reservoir" / "reservoir_manifest.json")
        dataset = load_feature_dataset(args.features)
        analysis = AnalysisStage()
        model = analysis.fit_all(dataset)
        output = args.output or args.experiment / "models" / "fixed_readout.joblib"
        analysis.save_model(model, dataset, output, config=config, reservoir_hash=reservoir.reservoir_hash, amplitude_ua=args.amplitude)
        print(output)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
