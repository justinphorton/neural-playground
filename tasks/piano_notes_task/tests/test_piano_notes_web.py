import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from tasks.piano_notes_task.experiment_config import ExperimentConfig
from tasks.piano_notes_task.reservoir_artifact import ReservoirManifest
from tasks.piano_notes_task.piano_notes_web import (
    ModelBundle,
    build_upload_page,
    score_prediction,
    summarize_probabilities,
)


class PianoNotesWebTests(unittest.TestCase):
    def test_probability_summary_averages_repeats_and_preserves_class_order(self):
        result = summarize_probabilities(
            ("C4", "E4"),
            np.asarray([[0.8, 0.2], [0.6, 0.4], [0.7, 0.3]], dtype=float),
        )
        self.assertEqual(result["predicted_label"], "C4")
        self.assertEqual(result["class_order"], ["C4", "E4"])
        self.assertAlmostEqual(result["probabilities"]["C4"], 0.7)
        self.assertEqual(result["repeat_count"], 3)

    def test_model_bundle_rejects_provenance_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = Path(directory)
            (experiment / "models").mkdir()
            (experiment / "reservoir").mkdir()
            config = ExperimentConfig(recording_electrodes=tuple(range(8)))
            config.write(experiment / "config.json")
            ReservoirManifest(
                selected_gids=(1,), selected_gids_by_population={"EXC": (1,)},
                source_file_hashes={}, livn_version="test", backend="test",
                fixed_simulation_seed=123, model_parameters={}, electrode_geometry=[],
                stimulation_electrode=0, recording_electrodes=tuple(range(8)),
                reservoir_hash="reservoir", n_cells=1,
            ).write(experiment / "reservoir" / "reservoir_manifest.json")
            joblib.dump(object(), experiment / "models" / "fixed_readout.joblib")
            (experiment / "models" / "fixed_readout.joblib.json").write_text(json.dumps({
                "status": "development", "config_hash": "wrong",
                "reservoir_hash": "reservoir", "feature_mode": "fine_temporal",
                "window_ms": [205, 255], "recording_electrodes": list(range(8)),
                "selected_amplitude_ua": 3.125, "class_order": ["C4", "E4"],
            }))
            with self.assertRaises(ValueError):
                ModelBundle.load(experiment)

    def test_upload_page_explains_development_model_and_uses_wav_input(self):
        page = build_upload_page()
        self.assertIn("development model", page.lower())
        self.assertIn('accept=".wav,audio/wav"', page)
        self.assertIn('name="correct_note"', page)
        self.assertIn("C4", page)
        self.assertIn("E4", page)
        self.assertIn("/predict", page)

    def test_score_prediction_compares_supplied_note_without_changing_features(self):
        self.assertEqual(score_prediction("C4", "C4"), {"correct_note": "C4", "is_correct": True})
        self.assertEqual(score_prediction("E4", "C4"), {"correct_note": "C4", "is_correct": False})
        with self.assertRaises(ValueError):
            score_prediction("D4", "C4")


if __name__ == "__main__":
    unittest.main()
