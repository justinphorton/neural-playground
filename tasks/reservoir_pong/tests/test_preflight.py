import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tasks.reservoir_pong.preflight import run_preflight
from tasks.reservoir_pong.records import ExperimentConfig


class PreflightTests(unittest.TestCase):
    def test_fixed_assignment_preflight_evaluates_only_the_declared_medium_band(self):
        self.assertIn("candidate_ranges", inspect.signature(run_preflight).parameters)
        config = ExperimentConfig()
        medium = tuple(config.selected_residual_range)
        report = {
            "residual_range": list(medium),
            "passes_task_design_gate": True,
            "oracle_catch_rate": 1.0,
            "follow_ball_catch_rate": 0.2,
            "raw_pulse_ridge_catch_rate": 0.3,
            "raw_pulse_landing_mae": 0.2,
            "geometry_pulse_prefix_catch_rate": 0.7,
            "geometry_pulse_prefix_landing_mae": 0.1,
        }
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "metrics").mkdir()
            (run_dir / "logs").mkdir()
            with patch("tasks.reservoir_pong.preflight._evaluate_candidate",
                       return_value=report) as evaluate:
                updated, result = run_preflight(
                    run_dir, config, counts=(2, 2, 2), seed=19,
                    candidate_ranges=(medium,),
                )

        self.assertEqual(evaluate.call_count, 1)
        self.assertEqual(evaluate.call_args.args[1], medium)
        self.assertEqual(tuple(updated.selected_residual_range), medium)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(len(result["candidate_results"]), 1)


if __name__ == "__main__":
    unittest.main()
