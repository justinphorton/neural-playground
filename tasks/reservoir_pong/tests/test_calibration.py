import unittest

from tasks.reservoir_pong.calibration import (
    CalibrationCandidate,
    MemoryGapResult,
    choose_calibration,
    contiguous_memory_limit,
)


class CalibrationTests(unittest.TestCase):
    def test_calibration_chooses_shortest_phase_then_lowest_current_and_window(self):
        candidates = [
            CalibrationCandidate(0.2, 40, 0.5, True, 0.8, 0.7),
            CalibrationCandidate(0.1, 80, 1.0, True, 0.9, 0.8),
            CalibrationCandidate(0.1, 40, 1.0, True, 0.9, 0.8),
            CalibrationCandidate(0.1, 40, 0.5, True, 0.9, 0.8),
        ]
        chosen = choose_calibration(candidates)
        self.assertEqual((chosen.phase_ms, chosen.amplitude_ua, chosen.window_ms), (0.1, 40, 0.5))

    def test_memory_limit_stops_at_first_failed_gap_after_minimum(self):
        measured = [
            MemoryGapResult(1.0, 0.30, 0.1, False),
            MemoryGapResult(2.0, 0.70, 0.4, True),
            MemoryGapResult(3.0, 0.65, 0.3, True),
            MemoryGapResult(6.0, 0.60, 0.3, True),
            MemoryGapResult(10.0, 0.45, 0.2, False),
        ]
        self.assertEqual(contiguous_memory_limit(measured, minimum_gap_ms=1.5), 6.0)

    def test_memory_limit_fails_when_minimum_gap_does_not_pass(self):
        measured = [MemoryGapResult(2.0, 0.4, 0.2, False)]
        self.assertIsNone(contiguous_memory_limit(measured, minimum_gap_ms=1.0))


if __name__ == "__main__":
    unittest.main()
