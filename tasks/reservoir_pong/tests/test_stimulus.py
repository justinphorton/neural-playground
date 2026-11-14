import unittest

import numpy as np

from tasks.reservoir_pong.physics import CellEntry
import tasks.reservoir_pong.stimulus as stimulus
from tasks.reservoir_pong.stimulus import (
    PulseEvent,
    extract_post_pulse_features,
    make_waveform,
    map_entries_to_electrodes,
)


class StimulusTests(unittest.TestCase):
    def test_recording_guard_removes_synthetic_spikes_during_each_blind_interval(self):
        channels = np.asarray([10, 10, 11, 10], dtype=np.int32)
        times = np.asarray([5.05, 5.30, 5.45, 6.00])
        self.assertTrue(hasattr(stimulus, "apply_recording_guard"),
                        "the recording guard has not been implemented")
        kept_channels, kept_times = stimulus.apply_recording_guard(
            channels, times, [PulseEvent(5.0, 10)], blank_ms=0.3,
        )
        self.assertTrue(np.array_equal(kept_channels, [10, 11, 10]))
        self.assertTrue(np.array_equal(kept_times, [5.30, 5.45, 6.00]))

    def test_biphasic_stimulation_is_charge_balanced(self):
        event = PulseEvent(1.0, 4, 40.0, 0.1, 1, 1)
        waveform = make_waveform([event], 3.0, 8, dt_ms=0.1)
        self.assertEqual(waveform.shape, (30, 8))
        self.assertAlmostEqual(float(waveform[:, 4].sum()), 0.0)
        self.assertEqual(float(waveform[10, 4]), 40.0)
        self.assertEqual(float(waveform[11, 4]), -40.0)

    def test_cell_mapping_uses_physical_position(self):
        entries = [CellEntry(0.0, 0, 0), CellEntry(1.0, 1, 0)]
        ids = np.arange(16, dtype=np.int32)
        events = map_entries_to_electrodes(entries, ids, cols=4, rows=4)
        self.assertEqual([e.electrode_id for e in events], [0, 1])

    def test_post_pulse_window_is_half_open_and_skips_blind_phase(self):
        events = [PulseEvent(10.0, 0, 40.0, 0.1, 0, 0)]
        channels = np.array([0, 0, 0, 1, 1], dtype=np.int32)
        times = np.array([10.2, 10.3, 10.8, 10.4, 10.9])
        features = extract_post_pulse_features(
            channels, times, events, recording_channels=(0, 1), blank_ms=0.3, window_ms=0.5
        )
        self.assertEqual(features.tolist(), [[1.0, 1.0]])

    def test_grid_selection_returns_a_contiguous_physical_block(self):
        coords = np.array([[ch, x, y, 0] for y in range(8) for x in range(8)
                           for ch in [y * 8 + x]], dtype=float)
        grid = __import__("tasks.reservoir_pong.stimulus", fromlist=["select_grid"]).select_grid(
            coords, (3.5, 3.5), side=4
        )
        self.assertEqual(len(grid), 16)
        self.assertEqual(len({round(float(x), 2) for _, x, _ in grid}), 4)
        self.assertEqual(len({round(float(y), 2) for _, _, y in grid}), 4)


if __name__ == "__main__":
    unittest.main()
