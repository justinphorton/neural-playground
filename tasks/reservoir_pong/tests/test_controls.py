import unittest


class ControlTests(unittest.TestCase):
    def test_synthetic_artifacts_are_blanked_without_erasing_valid_spikes(self):
        from tasks.reservoir_pong.controls import synthetic_artifact_blanking_control

        result = synthetic_artifact_blanking_control(
            phase_ms=0.1, guard_ms=0.1, window_ms=0.5,
            recording_channels=(0, 1, 2, 3),
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["artifact_events_injected"], 3)
        self.assertEqual(result["artifact_feature_sum"], 0)
        self.assertEqual(result["post_blank_spikes_counted"], 2)


if __name__ == "__main__":
    unittest.main()
