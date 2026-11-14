import json
import os
import re
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import numpy as np

from tasks.reservoir_pong.records import ExperimentConfig, TrialRecord, create_run_directory
from tasks.reservoir_pong.web import SimulationWorker, make_handler, setup_payload


def record_fixture():
    return TrialRecord(
        trajectory_id="live-123", reservoir_hash="reservoir", config_hash="config",
        noise_seed=9, pulse_times_ms=np.array([0.0]), pulse_electrode_ids=np.array([30]),
        pulse_cells=np.array([[0, 0]]), spike_times_ms=np.array([20.5]),
        spike_gids=np.array([4]), spike_channels=np.array([30]),
        spike_channel_times_ms=np.array([20.5]), feature_counts=np.array([[1.0, 0.0]]),
        availability_ms=np.array([0.8]), neural_predictions=np.array([0.6]),
        raw_predictions=np.array([0.4]), paddle_positions=np.array([0.55]),
        paddle_times_ms=np.array([0.0]), raw_paddle_positions=np.array([0.5]),
        follow_paddle_positions=np.array([0.5]), landing_x=0.7, caught=True,
        raw_caught=False, follow_caught=False,
        metadata={"split": "test", "mode": "live simulation", "x0": 0.5,
                  "vx": 0.4, "fall_ms": 20.0, "commit_ms": 10.0,
                  "warmup_ms": 20.0, "feature_blank_ms": 0.3,
                  "feature_window_ms": 0.5, "reservoir_slot": 0,
                  "electrode_grid": [[30, 400.0, 1200.0]],
                  "readouts_available": True},
    )


class WebTests(unittest.TestCase):
    def test_dashboard_has_single_definition_of_interactive_renderers(self):
        source = Path(__file__).parents[1].joinpath("dashboard.js").read_text()
        for name in ("renderMetrics", "drawCourt", "renderControllerResults", "renderTrial", "init"):
            declarations = re.findall(
                rf"(?m)^(?:async\s+)?function\s+{name}\s*\(", source,
            )
            self.assertEqual(len(declarations), 1, f"{name} has {len(declarations)} definitions")

    def test_dashboard_serves_external_css_and_javascript_assets(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = create_run_directory(Path(temp), "web", ExperimentConfig(), "fixed")
            worker = SimulationWorker(run_dir)
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(run_dir, worker))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                expected = {"/dashboard.css": "text/css", "/dashboard.js": "text/javascript"}
                for route, content_type in expected.items():
                    with urlopen(base + route) as response:
                        self.assertEqual(response.status, 200)
                        self.assertTrue(response.headers["Content-Type"].startswith(content_type))
                        self.assertGreater(len(response.read()), 100)
            finally:
                server.shutdown()
                thread.join(timeout=2)
                server.server_close()
                worker.close()

    def test_setup_discovers_full_trial_records_for_reload_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = create_run_directory(Path(temp), "web", ExperimentConfig(), "fixed")
            stage = run_dir / "live" / "raw_trials"
            stage.mkdir(parents=True)
            record_fixture().save(stage / "live-123_slot0_r00.npz")
            payload = setup_payload(run_dir)
            self.assertEqual(payload["trials"], ["live-123_slot0_r00"])

    def test_setup_orders_saved_trials_by_record_time(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = create_run_directory(Path(temp), "web", ExperimentConfig(), "fixed")
            stage = run_dir / "live" / "raw_trials"
            stage.mkdir(parents=True)
            first = record_fixture()
            first_path = stage / "live-123_slot0_r00.npz"
            first.save(first_path)
            second = record_fixture()
            second.trajectory_id = "live-456"
            second_path = stage / "live-456_slot0_r00.npz"
            second.save(second_path)
            os.utime(first_path, ns=(100, 100))
            os.utime(second_path, ns=(200, 200))
            payload = setup_payload(run_dir)
            self.assertEqual(payload["trials"], [
                "live-123_slot0_r00", "live-456_slot0_r00",
            ])

    def test_setup_keeps_recent_live_trials_after_many_later_test_records(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = create_run_directory(Path(temp), "web", ExperimentConfig(), "fixed")
            live_dir = run_dir / "live" / "raw_trials"
            final_dir = run_dir / "final" / "raw_trials"
            live_dir.mkdir(parents=True)
            final_dir.mkdir(parents=True)

            live_path = live_dir / "live-123_slot0_r00.npz"
            record_fixture().save(live_path)
            os.utime(live_path, ns=(100, 100))

            for index in range(105):
                record = record_fixture()
                record.trajectory_id = f"test-{index:03d}"
                path = final_dir / f"test-{index:03d}_slot0_r00.npz"
                record.save(path)
                os.utime(path, ns=(200 + index, 200 + index))

            payload = setup_payload(run_dir)

            self.assertIn("live-123_slot0_r00", payload["trials"])
            self.assertEqual(len(payload["trials"]), 101)
            self.assertEqual(payload["trials"][-1], "test-104_slot0_r00")

    def test_recorded_replay_api_returns_saved_record_and_job_status(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = create_run_directory(Path(temp), "web", ExperimentConfig(), "fixed")
            stage = run_dir / "pilot" / "raw_trials"
            stage.mkdir(parents=True)
            record_fixture().save(stage / "live-123_slot0_r00.npz")
            worker = SimulationWorker(run_dir)
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(run_dir, worker))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(base + "/api/trials/live-123_slot0_r00") as response:
                    record = json.load(response)
                self.assertEqual(record["trajectory_id"], "live-123")
                self.assertEqual(record["pulses"][0]["electrode_id"], 30)
                self.assertEqual(record["grid"], [[30, 400.0, 1200.0]])
                self.assertEqual(record["neural_predictions"], [0.6])
                self.assertEqual(record["paddle"]["neural"], [0.55])
                self.assertEqual(record["outcomes"], {"neural": True, "raw": False, "follow": False})
                request = Request(base + "/api/trials", data=json.dumps({
                    "mode": "replay", "trial_id": "live-123_slot0_r00",
                }).encode(), headers={"Content-Type": "application/json"}, method="POST")
                with urlopen(request) as response:
                    job = json.load(response)
                self.assertEqual(job["status"], "complete")
                with urlopen(base + f"/api/jobs/{job['id']}") as response:
                    status = json.load(response)
                self.assertEqual(status["trial_id"], "live-123_slot0_r00")
            finally:
                server.shutdown()
                thread.join(timeout=2)
                server.server_close()
                worker.close()

    def test_missing_replay_trial_returns_json_404(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = create_run_directory(Path(temp), "web", ExperimentConfig(), "fixed")
            worker = SimulationWorker(run_dir)
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(run_dir, worker))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                request = Request(
                    f"http://127.0.0.1:{server.server_port}/api/trials",
                    data=json.dumps({"mode": "replay", "trial_id": "missing"}).encode(),
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with self.assertRaises(HTTPError) as caught:
                    urlopen(request)
                self.assertEqual(caught.exception.code, 404)
                body = json.load(caught.exception)
                self.assertIn("was not found", body["error"])
            finally:
                server.shutdown()
                thread.join(timeout=2)
                server.server_close()
                worker.close()


if __name__ == "__main__":
    unittest.main()
