import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from tasks.piano_notes_task.experiment_config import ExperimentConfig, create_experiment_directory
from tasks.piano_notes_task.dataset_manifest import build_training_manifest
from tasks.piano_notes_task.stimulus import build_trial_current, apply_fade_out
from tasks.piano_notes_task.feature_extraction import (
    PRIMARY_MEMORY_WINDOWS_MS,
    FeatureDataset,
    extract_features,
    load_feature_dataset,
    neural_feature_columns,
    save_feature_dataset,
)
from tasks.piano_notes_task.raw_trials import RawTrial, save_raw_trial, load_raw_trial
from tasks.piano_notes_task.fixed_reservoir_experiment import main as experiment_main
from tasks.piano_notes_task.reservoir_artifact import ReservoirManifest


class ExperimentPipelineTests(unittest.TestCase):
    def test_config_defaults_are_locked(self):
        config = ExperimentConfig()
        self.assertEqual(config.notes, ("C4", "E4"))
        self.assertEqual((config.baseline_ms, config.stimulation_ms, config.post_stimulus_ms), (500, 1000, 1000))
        self.assertEqual(config.sample_rate_hz, 10_000)
        self.assertEqual(config.screening_amplitudes_ua, (1.5, 3.125, 6.25, 12.5, 25.0))
        self.assertEqual(config.fade_out_ms, 15)
        self.assertEqual(config.full_dataset_repeats, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config.write(path)
            self.assertEqual(ExperimentConfig.read(path).config_hash, config.config_hash)

    def test_experiment_directory_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = create_experiment_directory(Path(directory), "demo", ExperimentConfig(), timestamp="20260101T000000Z")
            self.assertTrue((path / "config.json").is_file())
            with self.assertRaises(FileExistsError):
                create_experiment_directory(Path(directory), "demo", ExperimentConfig(), timestamp="20260101T000000Z")

    def test_manifest_rejects_d4_and_pairs_c4_e4(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for note in ("C4", "E4", "D4"):
                (root / note).mkdir()
            samples = np.zeros(10_000, dtype=np.int16)
            for note in ("C4", "E4", "D4"):
                wavfile.write(root / note / f"{note}_AcousticGrand_velocity080_hold0600ms.wav", 10_000, samples)
            manifest = build_training_manifest(root, include_notes=("C4", "E4"))
            self.assertEqual(len(manifest), 2)
            self.assertEqual({row.note for row in manifest}, {"C4", "E4"})
            self.assertEqual(manifest[0].condition_group_id, manifest[1].condition_group_id)

    def test_taper_and_trial_current_have_exact_zero_phases(self):
        waveform = np.ones(10_000, dtype=np.float32)
        tapered = apply_fade_out(waveform, 10_000, 15)
        self.assertEqual(float(tapered[-1]), 0.0)
        current, diagnostics = build_trial_current(waveform, peak_current_ua=25, baseline_ms=500, post_stimulus_ms=1000)
        self.assertEqual(current.shape, (25_000, 1))
        self.assertTrue(np.all(current[:5_000] == 0))
        self.assertTrue(np.all(current[15_000:] == 0))
        self.assertLessEqual(diagnostics["achieved_peak_ua"], 25.0 + 1e-6)

    def test_features_only_use_recording_electrodes_and_primary_windows_do_not_overlap_stimulation(self):
        channels = np.array([10, 11, 99, 12], dtype=np.int32)
        times = np.array([5, 10, 20, 800], dtype=float)
        features = extract_features(channels, times, recording_electrodes=(10, 11, 12), window_ms=(5, 55), mode="spike_count")
        self.assertEqual(features.shape, (3,))
        self.assertEqual(neural_feature_columns((10, 11, 12), "spike_count"), ("electrode_10_count", "electrode_11_count", "electrode_12_count"))
        self.assertEqual(PRIMARY_MEMORY_WINDOWS_MS[0], (5, 55))
        self.assertTrue(all(start >= 5 and end <= 1000 for start, end in PRIMARY_MEMORY_WINDOWS_MS))

    def test_raw_trial_round_trip_without_pickle(self):
        trial = RawTrial(
            trial_id="t1", recording_id="r1", label="C4", repeat=0,
            reservoir_hash="abc", noise_seed=4,
            neuron_spike_times_ms=np.array([1.0]), neuron_spike_ids=np.array([3], dtype=np.int64),
            event_times_ms=np.array([5.0]), event_channel_ids=np.array([10], dtype=np.int32),
            metadata={"config_hash": "cfg"},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trial.npz"
            save_raw_trial(trial, path)
            self.assertTrue(path.with_suffix(path.suffix + ".json").is_file())
            loaded = load_raw_trial(path)
        self.assertEqual(loaded.trial_id, trial.trial_id)
        np.testing.assert_array_equal(loaded.event_channel_ids, trial.event_channel_ids)
        self.assertEqual(loaded.metadata, trial.metadata)

    def test_feature_dataset_round_trip_without_pickle(self):
        dataset = FeatureDataset(np.ones((2, 8)), np.array(["C4", "E4"]), np.array(["r1", "r2"]), np.array(["g1", "g2"]), "spike_count", (5, 55), (1, 2), np.array([False, False]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.npz"
            save_feature_dataset(dataset, path)
            loaded = load_feature_dataset(path)
        np.testing.assert_array_equal(loaded.X, dataset.X)
        self.assertEqual(loaded.feature_mode, dataset.feature_mode)

    def test_train_command_writes_provenance_tagged_model(self):
        dataset = FeatureDataset(
            np.vstack([np.zeros((5, 8)), np.ones((5, 8))]),
            np.array(["C4"] * 5 + ["E4"] * 5),
            np.array([f"r{i}" for i in range(10)]),
            np.array([f"g{i}" for i in range(10)]),
            "spike_count", (5, 55), tuple(range(8)), np.zeros(10, dtype=bool),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = root / "experiment"
            (experiment / "reservoir").mkdir(parents=True)
            (experiment / "models").mkdir()
            config = ExperimentConfig(recording_electrodes=tuple(range(8)))
            config.write(experiment / "config.json")
            ReservoirManifest(
                selected_gids=(1,), selected_gids_by_population={"EXC": (1,)},
                source_file_hashes={}, livn_version="test", backend="test", fixed_simulation_seed=123,
                model_parameters={}, electrode_geometry=[], stimulation_electrode=0,
                recording_electrodes=tuple(range(8)), reservoir_hash="reservoir", n_cells=1,
            ).write(experiment / "reservoir" / "reservoir_manifest.json")
            feature_path = experiment / "features.npz"
            save_feature_dataset(dataset, feature_path)
            experiment_main(["train", str(experiment), "--features", str(feature_path), "--amplitude", "6.25"])
            self.assertTrue((experiment / "models" / "fixed_readout.joblib").is_file())
            self.assertEqual(json.loads((experiment / "models" / "fixed_readout.joblib.json").read_text())["status"], "development")

    def test_screen_run_command_reads_schedule_from_a_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = root / "experiment"
            (experiment / "reservoir").mkdir(parents=True)
            (experiment / "metrics").mkdir()
            config = ExperimentConfig()
            config.write(experiment / "config.json")
            ReservoirManifest(
                selected_gids=(1,), selected_gids_by_population={"EXC": (1,)},
                source_file_hashes={}, livn_version="test", backend="test", fixed_simulation_seed=123,
                model_parameters={}, electrode_geometry=[], stimulation_electrode=0,
                recording_electrodes=tuple(range(8)), reservoir_hash="reservoir", n_cells=1,
            ).write(experiment / "reservoir" / "reservoir_manifest.json")
            (experiment / "metrics" / "screen_schedule.json").write_text(json.dumps({"trials": []}))

            self.assertEqual(experiment_main(["screen-run", str(experiment)]), 0)


if __name__ == "__main__":
    unittest.main()
