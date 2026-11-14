import tempfile
import unittest
from pathlib import Path

from tasks.reservoir_pong.calibration import (
    CalibrationCandidate,
    MemoryGapResult,
    ReservoirCalibration,
)
from tasks.reservoir_pong.calibration_report import write_calibration_report
from tasks.reservoir_pong.records import ExperimentConfig


class CalibrationReportTests(unittest.TestCase):
    def test_report_keeps_calibration_evidence_and_limits_together(self):
        candidate = CalibrationCandidate(0.1, 80.0, 0.5, True, 0.9, 0.7)
        memory = MemoryGapResult(3.0, 0.6, 0.3, True, 0.1, 80.0, 0.5)
        report = ReservoirCalibration(
            0, "reservoir-hash", 0.0, 0.0, (candidate,), (memory,), 3.0,
            "passed", controls={"no_stimulation": {"mean": 0.0}},
            electrode_grid=((10, 400.0, 1200.0),),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "calibration").mkdir()
            write_calibration_report(
                root, {"status": "passed", "minimum_gap_ms": 0.8,
                       "maximum_gap_ms": 3.0, "fall_ms": 21.6},
                [report], candidate, ExperimentConfig(),
            )
            text = (root / "calibration" / "report.md").read_text()

        self.assertIn("Selected waveform", text)
        self.assertIn("Paired-pulse memory", text)
        self.assertIn("No-input control", text)
        self.assertIn("does not establish biological amplifier recovery", text)


if __name__ == "__main__":
    unittest.main()
