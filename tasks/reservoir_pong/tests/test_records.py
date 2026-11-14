import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tasks.reservoir_pong.records import (
    ExperimentConfig,
    ExperimentManifest,
    TrialRecord,
    create_run_directory,
    load_trial,
    save_trial,
    trial_checkpoint_matches,
)


class RecordTests(unittest.TestCase):
    def test_config_hash_round_trips_and_changes_with_parameters(self):
        config = ExperimentConfig()
        restored = ExperimentConfig.from_dict(config.to_dict())
        self.assertEqual(config.config_hash, restored.config_hash)
        self.assertNotEqual(config.config_hash, ExperimentConfig(n_cells=601).config_hash)

    def test_run_directory_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = create_run_directory(Path(directory), "pong_test", ExperimentConfig(), "fixed")
            self.assertTrue((path / "config.json").is_file())
            with self.assertRaises(FileExistsError):
                create_run_directory(Path(directory), "pong_test", ExperimentConfig(), "fixed")

    def test_npz_trial_checkpoint_checks_config_and_reservoir_identity(self):
        record = TrialRecord(
            trajectory_id="trajectory-1", reservoir_hash="reservoir-a", config_hash="config-a",
            noise_seed=45, pulse_times_ms=np.array([20.0]), pulse_electrode_ids=np.array([3]),
            pulse_cells=np.array([[0, 0]]), spike_times_ms=np.array([20.5]),
            spike_gids=np.array([7]), spike_channels=np.array([3]),
            spike_channel_times_ms=np.array([20.5]),
            feature_counts=np.array([[1.0, 0.0]]), availability_ms=np.array([20.8]),
            neural_predictions=np.array([0.6]), raw_predictions=np.array([0.4]),
            paddle_positions=np.array([0.5]), paddle_times_ms=np.array([0.1]),
            raw_paddle_positions=np.array([0.5]), follow_paddle_positions=np.array([0.5]),
            landing_x=0.7, caught=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.npz"
            record.metadata["readout_id"] = "player-a"
            save_trial(record, path)
            loaded = load_trial(path)
            self.assertTrue(np.array_equal(loaded.feature_counts, record.feature_counts))
            self.assertTrue(trial_checkpoint_matches(path, "config-a", "reservoir-a"))
            self.assertTrue(trial_checkpoint_matches(path, "config-a", "reservoir-a",
                                                     readout_id="player-a"))
            self.assertFalse(trial_checkpoint_matches(path, "config-b", "reservoir-a"))
            self.assertFalse(trial_checkpoint_matches(path, "config-a", "reservoir-b"))
            self.assertFalse(trial_checkpoint_matches(path, "config-a", "reservoir-a",
                                                      readout_id="player-b"))

    def test_failed_checkpoint_write_preserves_the_previous_complete_record(self):
        record = TrialRecord(
            trajectory_id="atomic-1", reservoir_hash="r", config_hash="c",
            noise_seed=4, pulse_times_ms=np.zeros(0), pulse_electrode_ids=np.zeros(0),
            pulse_cells=np.zeros((0, 2)), spike_times_ms=np.zeros(0),
            spike_gids=np.zeros(0), spike_channels=np.zeros(0),
            spike_channel_times_ms=np.zeros(0), feature_counts=np.zeros((0, 1)),
            availability_ms=np.zeros(0), neural_predictions=np.zeros(0),
            raw_predictions=np.zeros(0), paddle_positions=np.zeros(0),
            paddle_times_ms=np.zeros(0), raw_paddle_positions=np.zeros(0),
            follow_paddle_positions=np.zeros(0), landing_x=0.5, caught=False,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.npz"
            save_trial(record, path)
            previous = path.read_bytes()

            def fail_after_partial_write(target, **_arrays):
                target.write(b"partial archive")
                raise OSError("simulated interrupted checkpoint")

            with patch("tasks.reservoir_pong.storage.np.savez_compressed",
                       side_effect=fail_after_partial_write):
                with self.assertRaisesRegex(OSError, "interrupted checkpoint"):
                    save_trial(record, path)

            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_manifest_records_versions_and_model_identity(self):
        config = ExperimentConfig()
        manifest = ExperimentManifest(config.config_hash, "livn.backend.native.env", "2.0.0",
                                      "b" * 64, (1, 2, 3), "4" * 64)
        self.assertEqual(manifest.config_hash, config.config_hash)
        self.assertFalse(manifest.plasticity_enabled)

if __name__ == "__main__":
    unittest.main()
