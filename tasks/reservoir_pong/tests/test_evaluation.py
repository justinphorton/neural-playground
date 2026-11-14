import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from tasks.reservoir_pong.evaluation import (
    count_opposite_trajectory_ids,
    opposite_side_mask,
    paired_trajectory_bootstrap,
    success_decision,
)
import tasks.reservoir_pong.evaluation as evaluation


class EvaluationTests(unittest.TestCase):
    def test_opposite_side_uses_commit_position_and_reflected_landing(self):
        x0 = np.array([0.2, 0.2, 0.8])
        vx = np.array([0.2, 1.6, -0.2])
        fall = np.ones(3)
        mask = opposite_side_mask(x0, vx, fall, 0.5)
        self.assertTrue(np.array_equal(mask, [False, True, False]))

    def test_bootstrap_is_paired_by_whole_trajectory_and_repeat_averaged(self):
        groups = np.repeat(np.arange(100), 2)
        neural = np.ones(200, dtype=bool)
        baseline = np.zeros(200, dtype=bool)
        estimate, lower, upper = paired_trajectory_bootstrap(neural, baseline, groups, seed=2)
        self.assertEqual(estimate, 1.0)
        self.assertAlmostEqual(lower, 1.0)
        self.assertAlmostEqual(upper, 1.0)

    def test_remaining_distance_error_slices_are_precommit_unclipped_and_grouped(self):
        self.assertTrue(hasattr(evaluation, "normalized_remaining_error_slices"))
        rows = []
        for slot in range(3):
            for repeat in range(2):
                rows.append(SimpleNamespace(
                    trajectory_id="ball-a", landing_x=0.6,
                    metadata={"fall_ms": 10.0, "reservoir_slot": slot},
                    availability_ms=np.asarray([0.0, 2.5, 5.0, 7.5]),
                    neural_predictions=np.asarray([0.7, 0.8, 0.5, 1.2]),
                    raw_predictions=np.asarray([0.6, 0.7, 0.7, 0.9]),
                ))
        result = evaluation.normalized_remaining_error_slices(rows, bootstrap_repeats=100)
        far = result["0.875_1"]
        self.assertEqual(far["trajectories"], 1)
        self.assertEqual(far["decisions"], 6)
        self.assertAlmostEqual(far["neural_mae"], 0.1)
        self.assertAlmostEqual(far["raw_mae"], 0.0)
        near = result["0.5_0.625"]
        self.assertEqual(near["decisions"], 6)

    def test_scientific_success_requires_ci_gain_replication_and_mae(self):
        metrics = {
            "opposite_count": 360,
            "opposite_trajectory_count": 120,
            "opposite_trajectories_by_reservoir": {"0": 120, "1": 120, "2": 120},
            "mean_catch_gain": 0.08,
            "gain_ci_lower": 0.02,
            "reservoir_gains": [0.01, 0.07, 0.12],
            "neural_landing_mae": 0.2,
            "raw_landing_mae": 0.2,
        }
        self.assertTrue(success_decision(metrics)[0])
        metrics["opposite_trajectory_count"] = 99
        self.assertFalse(success_decision(metrics)[0])
        metrics["opposite_trajectory_count"] = 120
        metrics["gain_ci_lower"] = -0.01
        self.assertFalse(success_decision(metrics)[0])

    def test_opposite_side_supplement_does_not_change_overall_catch_rate(self):
        base = {"neural_catch_rate": 0.42, "raw_catch_rate": 0.3,
                "reservoirs": {str(i): {"n": 1} for i in range(3)}}
        supplement = [
            SimpleNamespace(
                metadata={"reservoir_slot": slot, "x0": 0.2, "vx": 1.6,
                          "fall_ms": 1.0},
                trajectory_id=f"supp-{slot}", caught=True, raw_caught=False,
            ) for slot in range(3)
        ]
        from tasks.reservoir_pong.evaluation import merge_opposite_analysis
        result = merge_opposite_analysis(base, supplement, bootstrap_repeats=100)
        self.assertEqual(result["neural_catch_rate"], 0.42)
        self.assertEqual(result["opposite_trajectory_count"], 3)
        self.assertEqual(result["reservoir_gains"], [1.0, 1.0, 1.0])

    def test_supplement_count_uses_opposite_trajectories_and_collapses_repeats(self):
        records = [
            SimpleNamespace(trajectory_id="opp", metadata={
                "reservoir_slot": 0, "x0": 0.2, "vx": 1.6, "fall_ms": 1.0,
            }),
            SimpleNamespace(trajectory_id="opp", metadata={
                "reservoir_slot": 0, "x0": 0.2, "vx": 1.6, "fall_ms": 1.0,
            }),
            SimpleNamespace(trajectory_id="same-side", metadata={
                "reservoir_slot": 0, "x0": 0.2, "vx": 0.2, "fall_ms": 1.0,
            }),
            SimpleNamespace(trajectory_id="opp-slot1", metadata={
                "reservoir_slot": 1, "x0": 0.2, "vx": 1.6, "fall_ms": 1.0,
            }),
        ]
        self.assertEqual(count_opposite_trajectory_ids(records, 0.5), 2)
        self.assertEqual(count_opposite_trajectory_ids(records[:3], 0.5), 1)

if __name__ == "__main__":
    unittest.main()
