"""Post-process a completed selected-amplitude collection and fit the readout."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .analysis import grouped_folds, permutation_p_value
from .experiment_config import ExperimentConfig
from .feature_extraction import PRIMARY_MEMORY_WINDOWS_MS, save_feature_dataset
from .fixed_reservoir import AnalysisStage, FeatureStage
from .raw_trials import load_raw_trial
from .reservoir_artifact import ReservoirManifest


def _selected_trials(experiment: Path, amplitude: float):
    trials = []
    for path in sorted((experiment / "raw_trials").glob("*.npz")):
        trial = load_raw_trial(path)
        if trial.label in {"C4", "E4"} and np.isclose(float(trial.metadata.get("amplitude_ua", -1.0)), amplitude):
            trials.append(trial)
    return trials


def wait_for_collection(experiment: Path, amplitude: float, expected: int = 768, poll_seconds: int = 30):
    while True:
        count = len(_selected_trials(experiment, amplitude))
        if count >= expected:
            return
        time.sleep(poll_seconds)


def run(experiment: Path, amplitude: float) -> Path:
    experiment = Path(experiment)
    config = ExperimentConfig.read(experiment / "config.json")
    reservoir = ReservoirManifest.read(experiment / "reservoir" / "reservoir_manifest.json")
    trials = _selected_trials(experiment, amplitude)
    if len(trials) != 768:
        raise RuntimeError(f"expected 768 selected-amplitude note trials, found {len(trials)}")
    if len({trial.noise_seed for trial in trials}) != len(trials):
        raise RuntimeError("selected-amplitude trials contain duplicate noise seeds")
    if len({trial.reservoir_hash for trial in trials}) != 1 or trials[0].reservoir_hash != reservoir.reservoir_hash:
        raise RuntimeError("selected-amplitude trials do not share the canonical reservoir hash")
    if {trial.label for trial in trials} != {"C4", "E4"}:
        raise RuntimeError("selected-amplitude collection is missing a note class")

    electrodes = tuple(reservoir.recording_electrodes)
    stage = FeatureStage(electrodes)
    analysis = AnalysisStage()
    metrics = {
        "amplitude_ua": amplitude,
        "n_trials": len(trials),
        "reservoir_hash": reservoir.reservoir_hash,
        "windows": [],
    }
    best = None
    for window in PRIMARY_MEMORY_WINDOWS_MS:
        dataset = stage.from_trials(trials, window, "fine_temporal")
        output = experiment / "features" / f"fine_temporal_{window[0]:g}_{window[1]:g}.npz"
        save_feature_dataset(dataset, output)
        result = analysis.evaluate(dataset)
        folds = grouped_folds(dataset.labels, dataset.recording_ids, dataset.condition_groups, 5)
        p_value = permutation_p_value(dataset.X, dataset.labels, dataset.condition_groups, folds, result.balanced_accuracy, permutations=100, seed=0)
        row = {
            "window_ms": list(window),
            "balanced_accuracy": result.balanced_accuracy,
            "accuracy": result.accuracy,
            "roc_auc": result.roc_auc,
            "permutation_p_value_100": p_value,
        }
        metrics["windows"].append(row)
        if best is None or (p_value < 0.05 and best[0] >= 0.05) or (p_value < 0.05 and best[0] < 0.05 and result.balanced_accuracy > best[1]) or (p_value >= 0.05 and best[0] >= 0.05 and result.balanced_accuracy > best[1]):
            best = (p_value, result.balanced_accuracy, window, output, dataset)

    if best is None:
        raise RuntimeError("no primary feature window was evaluated")
    _, _, window, feature_path, dataset = best
    model = analysis.fit_all(dataset)
    model_path = experiment / "models" / "fixed_readout.joblib"
    analysis.save_model(model, dataset, model_path, config=config, reservoir_hash=reservoir.reservoir_hash, amplitude_ua=amplitude)
    metrics["selected_window_ms"] = list(window)
    metrics["model_path"] = str(model_path)
    metrics["feature_path"] = str(feature_path)
    (experiment / "metrics" / "development_full_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return model_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--amplitude", type=float, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    if args.wait:
        wait_for_collection(args.experiment, args.amplitude)
    print(run(args.experiment, args.amplitude))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
