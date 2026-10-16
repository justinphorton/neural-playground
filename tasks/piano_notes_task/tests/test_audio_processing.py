"""Behavior checks for automatic piano-note input preparation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from tasks.piano_notes_task.audio_processing import discover_samples, prepare_audio


class PianoAudioTests(unittest.TestCase):
    def test_discovery_uses_note_folders_and_keeps_splits_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for split, note, name in (
                ("training_data", "C4", "one.wav"),
                ("training_data", "D4", "two.wav"),
                ("test_data", "C4", "three.wav"),
            ):
                destination = root / split / note / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.touch()

            training = discover_samples(root, "training_data")
            testing = discover_samples(root, "test_data")
            self.assertEqual([(s.note, s.path.name) for s in training],
                             [("C4", "one.wav"), ("D4", "two.wav")])
            self.assertEqual([(s.note, s.path.name) for s in testing],
                             [("C4", "three.wav")])

    def test_preparation_crops_silence_downmixes_and_peak_normalizes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "C4.wav"
            sample_rate = 8000
            t = np.arange(sample_rate, dtype=float) / sample_rate
            tone = 0.4 * np.sin(2 * np.pi * 261.63 * t)
            source = np.concatenate((np.zeros(sample_rate), tone, np.zeros(sample_rate)))
            stereo = np.column_stack((source, source))
            wavfile.write(path, sample_rate, (stereo * 32767).astype(np.int16))

            prepared = prepare_audio(path, stimulus_rate=10_000)

            self.assertEqual(prepared.sample_rate, 10_000)
            self.assertAlmostEqual(prepared.original_duration_s, 3.0, places=2)
            self.assertGreater(prepared.start_s, 0.8)
            self.assertLess(prepared.start_s, 1.1)
            self.assertEqual(len(prepared.samples), 10_000)
            self.assertEqual(prepared.duration_s, 1.0)
            self.assertAlmostEqual(float(np.max(np.abs(prepared.samples))), 1.0, places=5)
            self.assertTrue(np.any(prepared.samples < 0))
            self.assertTrue(np.any(prepared.samples > 0))

    def test_short_recording_is_padded_after_peak_normalization(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "short.wav"
            sample_rate = 8000
            t = np.arange(sample_rate // 4, dtype=float) / sample_rate
            tone = 0.4 * np.sin(2 * np.pi * 261.63 * t)
            wavfile.write(path, sample_rate, (tone * 32767).astype(np.int16))

            prepared = prepare_audio(path, stimulus_rate=10_000)

            self.assertEqual(len(prepared.samples), 10_000)
            self.assertTrue(np.all(prepared.samples[3000:] == 0))
            self.assertAlmostEqual(float(np.max(np.abs(prepared.samples))), 1.0, places=5)

    def test_silence_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "silent.wav"
            wavfile.write(path, 8000, np.zeros(8000, dtype=np.int16))
            with self.assertRaisesRegex(ValueError, "silent"):
                prepare_audio(path)


if __name__ == "__main__":
    unittest.main()
