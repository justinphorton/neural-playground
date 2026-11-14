from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from tasks.reservoir_pong.records import ExperimentConfig
from tasks.reservoir_pong.evaluation_controls import unstimulated_window_features


class FreshTestControlTests(unittest.TestCase):
    def test_no_stimulation_features_use_same_half_open_guarded_windows(self):
        config = ExperimentConfig(
            warmup_ms=0.0, fall_ms=10.0, feature_blank_ms=0.3,
            feature_window_ms=0.5,
        )
        record = SimpleNamespace(
            pulse_times_ms=np.asarray([1.0]),
            pulse_electrode_ids=np.asarray([10]),
            pulse_cells=np.asarray([[0, 0]]),
            availability_ms=np.asarray([1.8]),
            metadata={"commit_ms": 5.0},
        )
        features = unstimulated_window_features(
            np.asarray([10, 10, 10, 10]),
            np.asarray([1.2, 1.3, 1.79, 1.8]),
            record, (10,), config,
        )
        self.assertEqual(features.shape, (1, 1))
        self.assertEqual(float(features[0, 0]), 2.0)


if __name__ == "__main__":
    unittest.main()
