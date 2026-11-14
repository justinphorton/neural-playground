import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tasks.reservoir_pong.physics import BallSpec, landing_position
from tasks.reservoir_pong.readouts import paddle_trace, paddle_trace_times
from tasks.reservoir_pong.records import ExperimentConfig
from tasks.reservoir_pong.experiment import (
    TrialTrace,
    _has_wall_bounce_after_commit,
    generate_accepted_balls,
    trace_to_record,
)
from tasks.reservoir_pong.physics import reflected_position
from tasks.reservoir_pong.stimulus import PulseEvent


class ExperimentTests(unittest.TestCase):
    def test_commitment_ignores_predictions_that_arrive_after_commit(self):
        ball = BallSpec("b", 0.5, 0.8, 24, commit_fraction=0.5)
        times = np.array([5.0, 11.0, 13.0, 18.0])
        predictions = np.array([0.7, 0.7, 0.1, 0.1])
        trace, _ = paddle_trace(ball, times, predictions, integration_step_ms=0.1)
        self.assertEqual(trace.shape[0], 241)
        self.assertTrue(np.allclose(trace[121:], trace[120:121]))

    def test_paddle_catches_a_stationary_center_ball_when_aimed_correctly(self):
        ball = BallSpec("b", 0.5, 0.0, 24)
        _, caught = paddle_trace(ball, np.array([4.0]), np.array([0.5]))
        self.assertTrue(caught)

    def test_trial_records_share_the_configured_paddle_clock_across_controllers(self):
        config = ExperimentConfig(fall_ms=21.65, stimulus_dt_ms=0.2)
        ball = BallSpec("fractional-impact", 0.3, 0.6, config.fall_ms)
        simulation = SimpleNamespace(
            spike_times_ms=np.zeros(0), spike_gids=np.zeros(0, dtype=int),
            spike_channels=np.zeros(0, dtype=int),
            spike_channel_times_ms=np.zeros(0), noise_seed=3,
        )
        trace = TrialTrace(
            ball,
            tuple(PulseEvent(time, 1, 80, 0.1, 0, 0) for time in (0.0, 2.0, 4.0)),
            simulation, np.zeros((3, 1)), np.array([0.8, 2.8, 4.8]),
            ((1, 0.0, 0.0),), (1,), config.config_hash, "reservoir",
        )
        with patch("tasks.reservoir_pong.experiment.locked_prediction",
                   return_value=np.full(3, 0.5)):
            record = trace_to_record(
                trace, config, 0, readouts={"neural": object(), "raw": object(), "readout_id": "ridge"},
            )
        expected_times = paddle_trace_times(ball.fall_ms, config.stimulus_dt_ms)
        self.assertTrue(np.array_equal(record.paddle_times_ms, expected_times))
        for positions in (
            record.paddle_positions, record.raw_paddle_positions,
            record.follow_paddle_positions,
        ):
            self.assertEqual(len(positions), len(expected_times))

    def test_wall_touch_at_commit_is_not_a_post_commit_bounce(self):
        ball = BallSpec("boundary", 0.5, 1.0, 24.0, commit_fraction=0.5)
        self.assertAlmostEqual(reflected_position(ball.x0 + ball.vx * ball.commit_fraction), 1.0)
        self.assertFalse(_has_wall_bounce_after_commit(ball))

    def test_wall_touch_at_commit_is_not_a_bounce_when_moving_left(self):
        ball = BallSpec("left-boundary", 0.5, -1.0, 24.0, commit_fraction=0.5)
        self.assertAlmostEqual(reflected_position(ball.x0 + ball.vx * ball.commit_fraction), 0.0)
        self.assertFalse(_has_wall_bounce_after_commit(ball))

    def test_generation_enforces_spacing_memory_and_post_commit_wall_bounce(self):
        config = ExperimentConfig(minimum_gap_ms=0.8, maximum_gap_ms=9.0, fall_ms=24.0)
        balls = generate_accepted_balls(config, (0.40, 0.80), 20, "pilot", seed=9)
        self.assertEqual(len(balls), 20)
        for ball, events in balls:
            self.assertGreaterEqual(sum(e.onset_ms + 0.8 <= ball.commit_ms for e in events), 3)
            lo, hi = sorted((ball.x0 + ball.vx * ball.commit_fraction, ball.x0 + ball.vx))
            crossings = [(k - ball.x0) / ball.vx
                         for k in range(int(np.floor(lo)) - 1, int(np.ceil(hi)) + 2)
                         if ball.vx and ball.commit_fraction < (k - ball.x0) / ball.vx < 1.0]
            self.assertTrue(crossings)
            self.assertEqual(landing_position(ball) >= 0, True)

    def test_live_trajectory_namespace_is_not_a_training_split(self):
        config = ExperimentConfig(minimum_gap_ms=0.8, maximum_gap_ms=9.0, fall_ms=24.0)
        (ball, _), = generate_accepted_balls(
            config, (0.40, 0.80), 1, "live", seed=11,
        )
        self.assertEqual(ball.split, "live")
        self.assertTrue(ball.trajectory_id.startswith("live-"))

    def test_opposite_only_supplement_contains_no_same_side_cases(self):
        config = ExperimentConfig(minimum_gap_ms=0.8, maximum_gap_ms=9.0, fall_ms=24.0)
        balls = generate_accepted_balls(
            config, (0.40, 0.80), 12, "test", seed=31,
            opposite_only=True,
        )
        self.assertEqual(len(balls), 12)
        for ball, _ in balls:
            commit = reflected_position(ball.x0 + ball.vx * ball.commit_fraction)
            self.assertNotEqual(commit < 0.5, landing_position(ball) < 0.5)


if __name__ == "__main__":
    unittest.main()
