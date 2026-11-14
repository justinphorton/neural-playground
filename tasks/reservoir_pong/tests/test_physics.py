import unittest

from tasks.reservoir_pong.physics import (
    BallSpec,
    landing_position,
    pulse_events,
    reflected_position,
    validate_pulse_spacing,
)


class PhysicsTests(unittest.TestCase):
    def test_reflection_handles_multiple_wall_bounces(self):
        self.assertAlmostEqual(reflected_position(3.25), 0.75)
        self.assertAlmostEqual(reflected_position(-2.25), 0.25)

    def test_landing_position_uses_signed_horizontal_travel(self):
        ball = BallSpec("b1", x0=0.9, vx=0.8, fall_ms=20.0)
        self.assertAlmostEqual(landing_position(ball), 0.3)

    def test_corner_crossing_emits_one_diagonal_entry(self):
        ball = BallSpec("b2", x0=0.125, vx=0.5, fall_ms=20.0)
        events = pulse_events(ball, cols=4, rows=4)
        corner = [e for e in events if abs(e.onset_ms - 5.0) < 1e-9]
        self.assertEqual(len(corner), 1)
        self.assertEqual((corner[0].col, corner[0].row), (1, 1))

    def test_wall_reflection_does_not_emit_an_extra_entry(self):
        ball = BallSpec("b3", x0=0.875, vx=1.0, fall_ms=20.0)
        events = pulse_events(ball, cols=4, rows=4)
        self.assertFalse(any(abs(e.onset_ms - 2.5) < 1e-9 for e in events))

    def test_quantized_pulse_collisions_are_rejected(self):
        ball = BallSpec("b4", x0=0.1, vx=0.3, fall_ms=1.0)
        events = pulse_events(ball, cols=4, rows=4, quantize_ms=0.1)
        with self.assertRaises(ValueError):
            validate_pulse_spacing(events, minimum_gap_ms=0.3, maximum_gap_ms=10.0)

    def test_only_sequential_events_have_a_gap_constraint(self):
        ball = BallSpec("b5", x0=0.2, vx=0.4, fall_ms=20.0)
        events = pulse_events(ball, cols=4, rows=4)
        self.assertGreater(len(events), 2)
        validate_pulse_spacing(events, minimum_gap_ms=0.1, maximum_gap_ms=20.0)


if __name__ == "__main__":
    unittest.main()
