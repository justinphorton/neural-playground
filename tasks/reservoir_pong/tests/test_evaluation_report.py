import json
import tempfile
import unittest
from pathlib import Path

from tasks.reservoir_pong.evaluation_report import write_evaluation_artifacts


class EvaluationReportTests(unittest.TestCase):
    def test_report_writer_saves_metrics_and_human_readable_summary(self):
        metrics = {
            "scientific_success": False,
            "mean_catch_gain": 0.0,
            "gain_ci_lower": -0.1,
            "gain_ci_upper": 0.1,
            "opposite_trajectory_count": 0,
            "reservoirs": {},
            "success_reasons": ["test gate not met"],
        }
        with tempfile.TemporaryDirectory() as directory:
            stage_dir = Path(directory) / "final"
            write_evaluation_artifacts(stage_dir, "final", metrics)

            saved = json.loads((stage_dir / "metrics" / "evaluation.json").read_text())
            report = (stage_dir / "metrics" / "evaluation.md").read_text()

        self.assertEqual(saved, metrics)
        self.assertIn("# Final Pong evaluation", report)
        self.assertIn("test gate not met", report)
        self.assertIn("does not model biological drift", report)


if __name__ == "__main__":
    unittest.main()
