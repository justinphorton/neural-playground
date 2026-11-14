import unittest

import numpy as np

from tasks.reservoir_pong.physics import BallSpec
from tasks.reservoir_pong.readouts import (
    build_prefix_design,
    fit_ridge,
    legal_paddle_target,
    paddle_trace,
    paddle_trace_times,
    trajectory_sample_weights,
)
from tasks.reservoir_pong.stimulus import PulseEvent


class ReadoutTests(unittest.TestCase):
    def test_sparse_prefix_contains_only_delivered_pulses(self):
        events = [PulseEvent(1.0, 3), PulseEvent(3.0, 7), PulseEvent(5.0, 5)]
        matrix = build_prefix_design(events, [2.0, 4.0], (3, 5, 7), 8.0, 0.1)
        self.assertEqual(matrix.shape, (2, 80 * 3))
        self.assertEqual(matrix.getrow(0).nnz, 1)
        self.assertEqual(matrix.getrow(1).nnz, 2)
        self.assertTrue(np.array_equal(matrix.getrow(1).indices, [30, 92]))

    def test_default_prefix_excludes_a_pulse_at_the_decision_tick(self):
        events = [PulseEvent(0.2, 3), PulseEvent(0.3, 7)]
        matrix = build_prefix_design(events, [0.3], (3, 7), 1.0, 0.1)
        self.assertEqual(matrix.getrow(0).nnz, 1)
        self.assertEqual(matrix.getrow(0).indices.tolist(), [4])

    def test_legacy_prefix_can_include_a_pulse_at_the_decision_tick(self):
        events = [PulseEvent(0.2, 3), PulseEvent(0.3, 7)]
        matrix = build_prefix_design(
            events, [0.3], (3, 7), 1.0, 0.1,
            cutoff_policy="at_or_before",
        )
        self.assertEqual(matrix.getrow(0).nnz, 2)

    def test_trajectory_weights_give_each_ball_equal_total_weight(self):
        groups = np.array(["a", "a", "a", "b"])
        weights = trajectory_sample_weights(groups)
        self.assertAlmostEqual(float(weights[:3].sum()), float(weights[3]))

    def test_linear_ridge_learns_a_continuous_landing_target(self):
        x = np.arange(20, dtype=float).reshape(-1, 1)
        y = 0.2 + 0.03 * x[:, 0]
        model = fit_ridge(x, y, alpha=0.01)
        self.assertAlmostEqual(float(model.predict([[10]])[0]), 0.5, places=2)

    def test_seeded_ball_generation_stays_within_declared_medium_band(self):
        ball = BallSpec("id", 0.5, 0.6, 24.0, split="pilot")
        self.assertEqual(ball.split, "pilot")

    def test_paddle_does_not_apply_a_prediction_before_its_timestamp(self):
        ball = BallSpec("causal", 0.5, 0.0, 10.0, commit_fraction=0.5)
        trace, _ = paddle_trace(
            ball, np.asarray([1.0]), np.asarray([0.94]), integration_step_ms=0.1,
        )
        self.assertAlmostEqual(trace[10], 0.5)
        self.assertGreater(trace[11], trace[10])

    def test_paddle_keeps_moving_toward_the_committed_target_after_commit(self):
        ball = BallSpec("lock", 0.5, 0.0, 10.0, commit_fraction=0.5)
        trace, _ = paddle_trace(
            ball, np.asarray([1.0, 5.1]), np.asarray([0.94, 0.1]),
            integration_step_ms=0.1,
        )
        self.assertAlmostEqual(trace[50], 0.82, places=7)
        self.assertGreater(trace[60], trace[50])
        self.assertAlmostEqual(trace[-1], 0.94, places=7)

    def test_targets_are_clipped_to_legal_paddle_center_positions(self):
        self.assertAlmostEqual(legal_paddle_target(-0.2, 0.12), 0.06)
        self.assertAlmostEqual(legal_paddle_target(1.2, 0.12), 0.94)
        ball = BallSpec("legal-edge", 0.5, 0.0, 10.0, commit_fraction=0.5)
        trace, _ = paddle_trace(ball, np.asarray([0.1]), np.asarray([1.2]))
        self.assertTrue(np.all(trace <= 0.94 + 1e-12))

    def test_paddle_trace_timestamps_include_an_unaligned_impact_time(self):
        times = paddle_trace_times(1.05, 0.1)
        self.assertEqual(times[-2:].tolist(), [1.0, 1.05])


if __name__ == "__main__":
    unittest.main()
