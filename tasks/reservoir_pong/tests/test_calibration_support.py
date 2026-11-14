import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tasks.reservoir_pong.calibration_support import CalibrationRawRecorder, holdout_accuracy


class CalibrationSupportTests(unittest.TestCase):
    def test_raw_recorder_round_trips_flattened_spikes_and_offsets(self):
        recorder = CalibrationRawRecorder()
        result = SimpleNamespace(
            spike_gids=np.asarray([4, 5]), spike_times_ms=np.asarray([1.0, 2.0]),
            spike_channels=np.asarray([10]), spike_channel_times_ms=np.asarray([1.5]),
        )
        recorder.add(result, "spatial", 7, electrode=12)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.npz"
            recorder.save(path)
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(data["kind"].tolist(), ["spatial"])
                self.assertEqual(data["spike_offsets"].tolist(), [0, 2])
                self.assertEqual(data["channel_offsets"].tolist(), [0, 1])
                self.assertEqual(data["spike_gids"].tolist(), [4, 5])
                self.assertEqual(data["spike_channels"].tolist(), [10])

    def test_holdout_accuracy_returns_rowwise_correctness(self):
        features = np.asarray([[0.0], [0.1], [0.2], [0.3],
                               [9.7], [9.8], [9.9], [10.0]])
        labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
        accuracy, correct = holdout_accuracy(features, labels, seed=21)
        self.assertEqual(accuracy, 1.0)
        self.assertEqual(correct.shape, (2,))
        self.assertTrue(np.all(correct))


if __name__ == "__main__":
    unittest.main()
