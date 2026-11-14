import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tasks.reservoir_pong.records import ExperimentConfig
from tasks.reservoir_pong.physics import BallSpec, CellEntry
from tasks.reservoir_pong.training import run_training_stage
from tasks.reservoir_pong.tests.test_pipeline import sample_record


class TrainingTests(unittest.TestCase):
    def test_collect_slot_writes_then_resumes_compatible_trial_checkpoints(self):
        from tasks.reservoir_pong import training

        dependencies = (
            "generate_accepted_balls", "make_trial_trace", "trial_record_path",
            "trial_checkpoint_matches",
        )
        self.assertTrue(all(hasattr(training, name) for name in dependencies),
                        "collect_slot must import its trial-generation and checkpoint helpers")

        config = ExperimentConfig(reservoir_count=1, noise_repeats=1, fall_ms=10.0)
        ball = BallSpec("collected-ball", 0.4, 0.2, 10.0, split="train")
        entries = [CellEntry(0.0, 0, 0)]
        trace = SimpleNamespace(simulation=SimpleNamespace(spike_times_ms=[]))
        reservoir = Mock()
        reservoir.reservoir_hash = "reservoir-a"
        record = sample_record(0)
        record.trajectory_id = ball.trajectory_id
        record.config_hash = config.config_hash
        record.reservoir_hash = reservoir.reservoir_hash
        record.metadata.update(split="train", reservoir_slot=0, repeat=0)

        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp)
            for name in ("metrics", "raw_trials", "logs"):
                (stage / name).mkdir()
            common_patches = (
                patch.object(training, "build_reservoir", return_value=reservoir),
                patch.object(training, "_write_manifest"),
                patch.object(training, "ReservoirRunner", return_value=object()),
            )
            with (patch.object(training, "generate_accepted_balls",
                                return_value=[(ball, entries)]) as generate,
                  patch.object(training, "make_trial_trace", return_value=trace) as simulate,
                  patch.object(training, "trace_to_record", return_value=record),
                  common_patches[0], common_patches[1], common_patches[2]):
                saved = training.collect_slot(
                    stage, config, 0, "train", 1,
                    tuple(config.selected_residual_range), repeats=1,
                )
            self.assertEqual(len(saved), 1)
            self.assertTrue(saved[0].is_file())
            self.assertEqual(generate.call_count, 1)
            self.assertEqual(simulate.call_count, 1)

            with (patch.object(training, "generate_accepted_balls",
                                return_value=[(ball, entries)]),
                  patch.object(training, "make_trial_trace",
                                side_effect=AssertionError("compatible checkpoint was rerun")) as rerun,
                  patch.object(training, "trace_to_record",
                                side_effect=AssertionError("compatible checkpoint was rewritten")) as rewrite,
                  common_patches[0], common_patches[1], common_patches[2]):
                resumed = training.collect_slot(
                    stage, config, 0, "train", 1,
                    tuple(config.selected_residual_range), repeats=1,
                )
            self.assertEqual(resumed, saved)
            rerun.assert_not_called()
            rewrite.assert_not_called()

    def test_assignment_ridge_refit_freezes_the_causal_cutoff_metadata(self):
        from tasks.reservoir_pong.pipeline import load_readouts
        from tasks.reservoir_pong import training

        self.assertTrue(hasattr(training, "fit_slot_readouts"))

        config = ExperimentConfig(fall_ms=10.0, reservoir_count=1)
        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp) / "final"
            (stage / "raw_trials").mkdir(parents=True)
            (stage / "reservoirs").mkdir()
            (stage / "models").mkdir()
            (stage / "reservoirs" / "slot0.json").write_text(json.dumps({
                "reservoir_hash": "reservoir-a",
                "grid": [[10, 0.0, 0.0], [11, 1.0, 0.0]],
            }))
            for split, indices in (("train", range(4)), ("validation", range(4, 7))):
                for index in indices:
                    record = sample_record(index)
                    record.config_hash = config.config_hash
                    record.reservoir_hash = "reservoir-a"
                    record.metadata["split"] = split
                    record.save(stage / "raw_trials" / f"{split}-{index}_slot0_r00.npz")

            info = training.fit_slot_readouts(
                stage, config, 0, tuple(config.selected_residual_range),
            )
            saved = load_readouts(
                Path(info["model_path"]), config.config_hash, "reservoir-a",
            )

        self.assertEqual(info["pulse_cutoff_policy"], "strictly_before")
        self.assertEqual(saved["metadata"]["decision_policy_version"],
                         "strict-pulse-prefix-v2")

    def test_training_stage_persists_disjoint_whole_trajectory_splits(self):
        config = ExperimentConfig(
            reservoir_count=1, noise_repeats=2, pilot_counts=(4, 3, 0),
        )

        def collect_records(stage_dir, _config, slot, split, count, _residual,
                            repeats=None, **_kwargs):
            repeats = config.noise_repeats if repeats is None else repeats
            raw_trials = stage_dir / "raw_trials"
            paths = []
            for trajectory_index in range(count):
                trajectory_id = f"{split}-{trajectory_index}"
                for repeat in range(repeats):
                    record = sample_record(trajectory_index)
                    record.trajectory_id = trajectory_id
                    record.metadata.update(split=split, repeat=repeat, reservoir_slot=slot)
                    path = raw_trials / f"{trajectory_id}_slot{slot}_r{repeat:02d}.npz"
                    record.save(path)
                    paths.append(path)
            return paths

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "logs").mkdir()
            with patch("tasks.reservoir_pong.training.collect_slot", side_effect=collect_records), \
                    patch("tasks.reservoir_pong.training.fit_slot_readouts", return_value={"slot": 0}):
                run_training_stage(root, config, stage="pilot")

            assignments = json.loads(
                (root / "pilot" / "metrics" / "split_assignments.json").read_text(),
            )
            train_ids = set(assignments["train"])
            validation_ids = set(assignments["validation"])
            self.assertEqual(len(train_ids), 4)
            self.assertEqual(len(validation_ids), 3)
            self.assertTrue(train_ids.isdisjoint(validation_ids))


if __name__ == "__main__":
    unittest.main()
