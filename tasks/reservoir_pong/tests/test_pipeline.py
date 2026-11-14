import tempfile
import unittest
from pathlib import Path

import numpy as np

from tasks.reservoir_pong.pipeline import (
    build_examples,
    load_readouts,
    locked_prediction,
    save_readouts,
)
from tasks.reservoir_pong.records import TrialRecord
from tasks.reservoir_pong.readouts import fit_ridge
from tasks.reservoir_pong.training import load_records


def sample_record(index):
    return TrialRecord(
        trajectory_id=f"train-{index}", reservoir_hash="rhash", config_hash="chash",
        noise_seed=index, pulse_times_ms=np.array([0.0, 3.0]),
        pulse_electrode_ids=np.array([10, 11]), pulse_cells=np.array([[0, 0], [1, 0]]),
        spike_times_ms=np.array([20.5]), spike_gids=np.array([3]),
        spike_channels=np.array([10]), spike_channel_times_ms=np.array([20.5]),
        feature_counts=np.array([[index, 1.0], [index + 1, 2.0]]),
        availability_ms=np.array([1.0, 4.0]),
        neural_predictions=np.zeros(2), raw_predictions=np.zeros(2),
        paddle_positions=np.array([]), paddle_times_ms=np.array([]),
        raw_paddle_positions=np.array([]), follow_paddle_positions=np.array([]),
        landing_x=index / 10, caught=False,
        metadata={"split": "train", "commit_ms": 5.0, "warmup_ms": 20.0,
                  "fall_ms": 10.0, "x0": 0.4, "vx": 0.2, "repeat": 0},
    )


class PipelineTests(unittest.TestCase):
    def test_example_matrix_uses_only_precommit_frames_and_grid_pulses(self):
        records = [sample_record(i) for i in range(10)]
        examples = build_examples(records, electrode_grid_ids=(10, 11), dt_ms=0.1,
                                  warmup_ms=20.0, fall_ms=10.0)
        self.assertEqual(examples.neural.shape, (20, 2))
        self.assertEqual(examples.raw.shape, (20, 300 * 2))
        self.assertEqual(len(np.unique(examples.groups)), 10)
        self.assertTrue(np.all(examples.sample_weights >= 0))

    def test_new_training_examples_exclude_pulse_at_the_feature_cutoff(self):
        record = sample_record(3)
        record.pulse_times_ms = np.array([0.0, 3.0, 4.0])
        record.pulse_electrode_ids = np.array([10, 11, 10])
        record.pulse_cells = np.array([[0, 0], [1, 0], [0, 0]])
        examples = build_examples(
            [record], electrode_grid_ids=(10, 11), dt_ms=0.1,
            warmup_ms=20.0, fall_ms=10.0,
        )
        self.assertEqual(examples.raw.getrow(1).nnz, 2)

    def test_legacy_training_prefix_can_keep_the_old_inclusive_cutoff(self):
        record = sample_record(3)
        record.pulse_times_ms = np.array([0.0, 3.0, 4.0])
        record.pulse_electrode_ids = np.array([10, 11, 10])
        record.pulse_cells = np.array([[0, 0], [1, 0], [0, 0]])
        examples = build_examples(
            [record], electrode_grid_ids=(10, 11), dt_ms=0.1,
            warmup_ms=20.0, fall_ms=10.0,
            pulse_cutoff_policy="at_or_before",
        )
        self.assertEqual(examples.raw.getrow(1).nnz, 3)

    def test_readout_artifact_checks_configuration_and_reservoir(self):
        x = np.arange(20, dtype=float).reshape(-1, 1)
        model = fit_ridge(x, x[:, 0] / 20 + 0.2, alpha=1.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "readouts.joblib"
            save_readouts(path, model, model, config_hash="c", reservoir_hash="r",
                          feature_count=1, grid_ids=(10,),
                          pulse_cutoff_policy="strictly_before",
                          decision_policy_version="strict-pulse-prefix-v2")
            loaded = load_readouts(path, config_hash="c", reservoir_hash="r")
            self.assertEqual(loaded["metadata"]["pulse_cutoff_policy"], "strictly_before")
            self.assertAlmostEqual(float(loaded["neural"].predict([[10]])[0]),
                                   float(model.predict([[10]])[0]))
            with self.assertRaises(ValueError):
                load_readouts(path, config_hash="other", reservoir_hash="r")

    def test_prediction_does_not_mutate_frozen_readout(self):
        x = np.arange(20, dtype=float).reshape(-1, 1)
        model = fit_ridge(x, x[:, 0] / 20 + 0.2, alpha=1.0)
        before = model.named_steps["ridge"].coef_.copy()
        predictions = locked_prediction(model, x[:3])
        self.assertEqual(predictions.shape, (3,))
        self.assertTrue(np.array_equal(before, model.named_steps["ridge"].coef_))

    def test_training_loader_finds_the_checkpoint_filename_written_by_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_trials = root / "raw_trials"
            raw_trials.mkdir()
            record = sample_record(3)
            record.save(raw_trials / "train-3_slot0_r00.npz")
            loaded = load_records(root, slot=0, split="train")
            self.assertEqual([item.trajectory_id for item in loaded], ["train-3"])


if __name__ == "__main__":
    unittest.main()
