import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tasks.piano_notes_task.analysis import holm_bonferroni, paired_label_permutation
from tasks.piano_notes_task.dataset_manifest import build_training_manifest
from tasks.piano_notes_task.experiment_config import ExperimentConfig
from tasks.piano_notes_task.fixed_reservoir import SimulationStage, build_trial_specs, trial_checkpoint_matches
from tasks.piano_notes_task.raw_trials import RawTrial, save_raw_trial
from tasks.piano_notes_task.screening import AmplitudeScreenSummary, ScreeningGateFailed, choose_amplitude
from tasks.piano_notes_task.screening import build_screen_schedule


class FixedReservoirContractTests(unittest.TestCase):
    def test_simulation_stage_reuses_loaded_reservoir_until_closed(self):
        class FakeEnv:
            def close(self):
                pass

        fake_env = FakeEnv()
        fake_manifest = object()
        fake_site = object()
        with patch("tasks.piano_notes_task.fixed_reservoir.load_canonical_reservoir", return_value=(fake_env, fake_manifest, fake_site)) as loader:
            stage = SimulationStage(ExperimentConfig(), Path("experiment"), Path("reservoir"))
            self.assertEqual(stage._reservoir(), (fake_env, fake_manifest, fake_site))
            self.assertEqual(stage._reservoir(), (fake_env, fake_manifest, fake_site))
            loader.assert_called_once_with(Path("reservoir"))
            stage.close()

    def test_current_training_manifest_is_exactly_256_c4_e4_and_128_pairs(self):
        rows = build_training_manifest(Path(__file__).parents[1] / "training_data")
        self.assertEqual(len(rows), 256)
        self.assertEqual(sum(row.note == "C4" for row in rows), 128)
        self.assertEqual(sum(row.note == "E4" for row in rows), 128)
        self.assertEqual(len({row.condition_group_id for row in rows}), 128)
        self.assertTrue(all(row.dataset_split == "training" for row in rows))

    def test_trial_specs_have_unique_seeds_and_expected_identity(self):
        rows = build_training_manifest(Path(__file__).parents[1] / "training_data")[:2]
        specs = build_trial_specs(rows, "reservoir", 6.25, 3)
        self.assertEqual(len(specs), 6)
        self.assertEqual(len({spec.noise_seed for spec in specs}), 6)
        self.assertTrue(all(spec.expected_reservoir_hash == "reservoir" for spec in specs))

    def test_checkpoint_rejects_config_or_seed_changes(self):
        rows = build_training_manifest(Path(__file__).parents[1] / "training_data")[:1]
        spec = build_trial_specs(rows, "reservoir", 6.25, 1)[0]
        trial = RawTrial(spec.trial_id, spec.recording_id, spec.label, spec.repeat, spec.expected_reservoir_hash, spec.noise_seed, np.array([]), np.array([], dtype=np.int64), np.array([]), np.array([], dtype=np.int32), metadata={"config_hash": "config", "amplitude_ua": 6.25})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trial.npz"
            save_raw_trial(trial, path)
            self.assertTrue(trial_checkpoint_matches(path, spec, "config"))
            self.assertFalse(trial_checkpoint_matches(path, spec, "changed"))

    def test_paired_permutation_only_swaps_within_condition_group(self):
        labels = np.array(["C4", "E4", "C4", "E4"])
        groups = np.array(["g1", "g1", "g2", "g2"])
        shuffled = paired_label_permutation(labels, groups, np.random.default_rng(4))
        for group in ("g1", "g2"):
            self.assertEqual(sorted(shuffled[groups == group]), ["C4", "E4"])

    def test_holm_correction_is_monotonic_and_screen_gate_stops_without_candidate(self):
        adjusted = holm_bonferroni([0.001, 0.02, 0.5])
        self.assertTrue(np.all(adjusted >= np.array([0.001, 0.02, 0.5])))
        with self.assertRaises(ScreeningGateFailed):
            choose_amplitude([AmplitudeScreenSummary(1.5, 1, 1, 1, 1, 0.5, 0.5, 0.6, (0.4, 0.4), 0.1)])

    def test_screen_schedule_is_240_plus_96_plus_48(self):
        rows = build_training_manifest(Path(__file__).parents[1] / "training_data")
        first, selection = build_screen_schedule(rows, "reservoir", second_pass_amplitudes=(1.5, 3.125))
        self.assertEqual(len(selection.condition_groups), 24)
        self.assertEqual(len(first), 384)
        self.assertEqual(sum(spec.is_control for spec in first), 48)
        self.assertEqual(len({spec.noise_seed for spec in first}), 384)


if __name__ == "__main__":
    unittest.main()
