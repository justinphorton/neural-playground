from __future__ import annotations

import unittest
from types import SimpleNamespace

from tasks.reservoir_pong.evaluation_run import (
    _plan_shared_opposite_supplement,
    _validate_shared_opposite_supplement,
)


def record(trajectory_id: str, opposite: bool):
    return SimpleNamespace(
        trajectory_id=trajectory_id,
        metadata={
            "x0": 0.2,
            "vx": 1.6 if opposite else 0.2,
            "fall_ms": 1.0,
        },
    )


class EvaluationRunTests(unittest.TestCase):
    def test_supplement_is_shared_opposite_only_and_excluded_from_primary(self):
        base = {
            slot: [record("base-opp", True), record("base-same", False)]
            for slot in range(3)
        }
        base_opposite, missing = _plan_shared_opposite_supplement(
            base, commit_fraction=0.5, minimum_opposite=3,
            expected_trajectory_count=2,
        )
        self.assertEqual(base_opposite, {"base-opp"})
        self.assertEqual(missing, 2)
        supplements = {
            slot: [record("supp-1", True), record("supp-2", True)]
            for slot in range(3)
        }
        ids = _validate_shared_opposite_supplement(
            supplements, base_opposite, 0.5, 3,
        )
        self.assertEqual(ids[0], {"supp-1", "supp-2"})

    def test_supplement_rejects_mismatched_subreservoir_trajectories(self):
        base = {slot: [record("base", True)] for slot in range(3)}
        supplements = {
            slot: [record(f"supp-{slot}", True)]
            for slot in range(3)
        }
        with self.assertRaisesRegex(ValueError, "differ across subreservoirs"):
            _validate_shared_opposite_supplement(supplements, {"base"}, 0.5, 2)


if __name__ == "__main__":
    unittest.main()
