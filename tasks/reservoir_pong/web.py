"""Local dashboard API and serialized simulation worker."""

from __future__ import annotations

import json
import queue
import re
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .experiment import generate_accepted_balls, make_trial_trace, trace_to_record
from .pipeline import load_readouts
from .records import ExperimentConfig, load_trial, stable_seed
from .reservoir import ReservoirRunner, build_reservoir
from .storage import trial_record_path


def _record_payload(record) -> dict:
    meta = record.metadata
    scored = meta.get("readouts_available", True)
    return {
        "trajectory_id": record.trajectory_id,
        "config_hash": record.config_hash,
        "reservoir_hash": record.reservoir_hash,
        "reservoir_slot": meta.get("reservoir_slot"),
        "noise_seed": record.noise_seed,
        "grid": meta.get("electrode_grid", []),
        "ball": {
            "x0": float(meta["x0"]), "vx": float(meta["vx"]),
            "fall_ms": float(meta["fall_ms"]), "commit_ms": float(meta["commit_ms"]),
            "landing_x": float(record.landing_x),
        },
        "pulses": [{
            "time_ms": float(t), "electrode_id": int(electrode),
            "cell": [int(cell[0]), int(cell[1])],
            "phase_amplitudes_ua": list(meta.get("pulse_phase_amplitudes_ua", [])),
            "phase_durations_ms": list(meta.get("pulse_phase_durations_ms", [])),
        } for t, electrode, cell in zip(
            record.pulse_times_ms, record.pulse_electrode_ids, record.pulse_cells
        )],
        "spikes": [{"channel": int(ch), "time_ms": float(t)} for ch, t in zip(
            record.spike_channels, record.spike_channel_times_ms
        )],
        "features": record.feature_counts.astype(float).tolist(),
        "availability_ms": record.availability_ms.astype(float).tolist(),
        "neural_predictions": (record.neural_predictions.astype(float).tolist()
                               if scored else []),
        "raw_predictions": (record.raw_predictions.astype(float).tolist()
                            if scored else []),
        "paddle": {
            "times_ms": record.paddle_times_ms.tolist(),
            "neural": record.paddle_positions.tolist(),
            "raw": record.raw_paddle_positions.tolist(),
            "follow": record.follow_paddle_positions.tolist(),
        },
        "outcomes": {
            "neural": record.caught if scored else None,
            "raw": record.raw_caught if scored else None,
            "follow": record.follow_caught if scored else None,
        },
        "metadata": meta,
    }


def _find_record(run_dir: Path, trial_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", trial_id):
        raise ValueError("trial ID contains unsupported characters")
    matches = []
    for stage in ("final", "pilot", "live"):
        directory = Path(run_dir) / stage / "raw_trials"
        exact = directory / f"{trial_id}.npz"
        if exact.is_file():
            matches.append(exact)
        else:
            matches.extend(directory.glob(f"{trial_id}_slot*_r*.npz"))
    directory = Path(run_dir) / "raw_trials"
    exact = directory / f"{trial_id}.npz"
    if exact.is_file():
        matches.append(exact)
    else:
        matches.extend(directory.glob(f"{trial_id}_slot*_r*.npz"))
    if not matches:
        raise FileNotFoundError(f"trial {trial_id!r} was not found in this run")
    return sorted(matches)[0]


def _read_json(path: Path):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else None


def _trial_ids(run_dir: Path) -> list[str]:
    records = {}
    live_records = {}
    for stage in ("final", "pilot", "live"):
        for path in (Path(run_dir) / stage / "raw_trials").glob("*_slot*_r*.npz"):
            records[path.stem] = path.stat().st_mtime_ns
            if stage == "live":
                live_records[path.stem] = path.stat().st_mtime_ns
    ordered = sorted(records.items(), key=lambda item: (item[1], item[0]))
    recent_ids = {trial_id for trial_id, _ in ordered[-100:]}
    recent_live_ids = {
        trial_id for trial_id, _ in sorted(live_records.items(), key=lambda item: (item[1], item[0]))[-100:]
    }
    retained = recent_ids | recent_live_ids
    return [trial_id for trial_id, _ in ordered if trial_id in retained]


def setup_payload(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    config = ExperimentConfig.from_dict(json.loads((run_dir / "config.json").read_text()))
    calibration = _read_json(run_dir / "calibration" / "summary.json")
    preflight = _read_json(run_dir / "metrics" / "preflight.json")
    final_metrics = _read_json(run_dir / "final" / "metrics" / "evaluation.json")
    pilot_metrics = _read_json(run_dir / "pilot" / "metrics" / "evaluation.json")
    selected_stage = "final" if final_metrics else "pilot" if pilot_metrics else None
    grid = None
    if selected_stage:
        manifest = _read_json(run_dir / selected_stage / "reservoirs" / "slot0.json")
        if manifest:
            grid = manifest.get("grid")
    if grid is None and calibration and calibration.get("reservoirs"):
        grid = calibration["reservoirs"][0].get("electrode_grid")
    if grid is None:
        calibration_manifest = _read_json(run_dir / "calibration" / "reservoir_slot0.json")
        if calibration_manifest:
            grid = calibration_manifest.get("electrode_grid")
    return {
        "run_id": run_dir.name,
        "config": config.to_dict(),
        "calibration": calibration,
        "preflight": preflight,
        "metrics": final_metrics or pilot_metrics,
        "metrics_stage": selected_stage,
        "grid": grid,
        "trials": _trial_ids(run_dir),
    }


def metrics_payload(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    return {
        "calibration": _read_json(run_dir / "calibration" / "summary.json"),
        "preflight": _read_json(run_dir / "metrics" / "preflight.json"),
        "pilot": _read_json(run_dir / "pilot" / "metrics" / "evaluation.json"),
        "final": _read_json(run_dir / "final" / "metrics" / "evaluation.json"),
    }


class SimulationWorker:
    """One serialized native simulator loop shared by all live jobs."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.config = ExperimentConfig.from_dict(json.loads((self.run_dir / "config.json").read_text()))
        self.jobs: dict[str, dict] = {}
        self.lock = threading.RLock()
        self.pending: queue.Queue[str | None] = queue.Queue()
        self.reservoirs = {}
        self.thread = threading.Thread(target=self._work, name="pong-simulation-worker", daemon=True)
        self.thread.start()

    def submit(self, body: dict) -> dict:
        mode = body.get("mode", "live")
        job_id = uuid.uuid4().hex
        if mode == "replay":
            trial_id = str(body.get("trial_id", ""))
            _find_record(self.run_dir, trial_id)
            job = {"id": job_id, "mode": "replay", "status": "complete",
                   "trial_id": trial_id, "progress": "Recorded replay ready"}
            with self.lock:
                self.jobs[job_id] = job
            return job
        if mode != "live":
            raise ValueError("mode must be 'live' or 'replay'")
        slot = int(body.get("slot", 0))
        if not 0 <= slot < self.config.reservoir_count:
            raise ValueError("reservoir slot is outside the configured range")
        job = {"id": job_id, "mode": "live", "status": "queued",
               "progress": "Waiting for serialized simulation worker", "slot": slot}
        with self.lock:
            self.jobs[job_id] = job
        self.pending.put(job_id)
        return job

    def _update(self, job_id: str, **fields) -> None:
        with self.lock:
            self.jobs[job_id].update(fields)

    def _ensure_reservoir(self, slot: int):
        if slot not in self.reservoirs:
            reservoir = build_reservoir(self.config, slot)
            self.reservoirs[slot] = (reservoir, ReservoirRunner(reservoir, self.config))
        return self.reservoirs[slot]

    def _models(self, slot: int, reservoir_hash: str):
        for stage in ("final", "pilot"):
            path = self.run_dir / stage / "models" / f"readouts_slot{slot}.joblib"
            if path.is_file():
                return load_readouts(path, self.config.config_hash, reservoir_hash), stage
        return None, None

    def _run_live(self, job_id: str, slot: int) -> None:
        self._update(job_id, status="running", progress="Loading fixed reservoir")
        reservoir, runner = self._ensure_reservoir(slot)
        seed = stable_seed("live", job_id, base_seed=self.config.seed)
        stats = {}
        self._update(job_id, progress="Generating a timing-valid ball")
        balls = generate_accepted_balls(
            self.config, tuple(self.config.selected_residual_range), 1, "live", seed,
            require_post_commit_bounce=True, stats=stats,
        )
        ball, entries = balls[0]
        self._update(job_id, progress="Running the native neural simulation",
                     trajectory_id=ball.trajectory_id, generation=stats)
        trace = make_trial_trace(ball, entries, reservoir, runner, self.config, seed)
        readouts, model_stage = self._models(slot, reservoir.reservoir_hash)
        record = trace_to_record(trace, self.config, slot, readouts=readouts)
        record.metadata.update({"mode": "live simulation", "model_stage": model_stage})
        path = trial_record_path(self.run_dir / "live" / "raw_trials", ball.trajectory_id, slot, 0)
        record.save(path)
        self._update(job_id, status="complete", progress="Live trial recorded",
                     trial_id=path.stem, trajectory_id=ball.trajectory_id,
                     file=str(path.relative_to(self.run_dir)),
                     readouts_available=bool(readouts))

    def _work(self) -> None:
        while True:
            job_id = self.pending.get()
            if job_id is None:
                self.pending.task_done()
                break
            with self.lock:
                job = dict(self.jobs[job_id])
            try:
                self._run_live(job_id, int(job["slot"]))
            except Exception as error:
                self._update(job_id, status="error", progress="Simulation failed",
                             error=f"{type(error).__name__}: {error}")
            finally:
                self.pending.task_done()

    def get_job(self, job_id: str) -> dict | None:
        with self.lock:
            job = self.jobs.get(job_id)
            return None if job is None else dict(job)

    def close(self) -> None:
        self.pending.put(None)
        self.thread.join(timeout=5)
        for reservoir, _ in self.reservoirs.values():
            reservoir.close()


def make_handler(run_dir: Path, worker: SimulationWorker):
    dashboard = Path(__file__).with_name("dashboard.html")
    assets = {
        "/dashboard.css": (dashboard.with_name("dashboard.css"), "text/css; charset=utf-8"),
        "/dashboard.js": (dashboard.with_name("dashboard.js"), "text/javascript; charset=utf-8"),
    }

    class Handler(BaseHTTPRequestHandler):
        server_version = "ReservoirPong/1.0"

        def _send(self, status: int, content: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def _send_json(self, status: int, value) -> None:
            self._send(status, json.dumps(value, allow_nan=False).encode(), "application/json; charset=utf-8")

        def do_GET(self):
            path = urlparse(self.path).path
            try:
                if path == "/":
                    self._send(200, dashboard.read_bytes(), "text/html; charset=utf-8")
                elif path in assets:
                    asset, content_type = assets[path]
                    self._send(200, asset.read_bytes(), content_type)
                elif path == "/api/setup":
                    self._send_json(200, setup_payload(run_dir))
                elif path == "/api/metrics":
                    self._send_json(200, metrics_payload(run_dir))
                elif path.startswith("/api/jobs/"):
                    job = worker.get_job(unquote(path.rsplit("/", 1)[-1]))
                    self._send_json(200 if job else 404, job or {"error": "job not found"})
                elif path.startswith("/api/trials/"):
                    trial_id = unquote(path.rsplit("/", 1)[-1])
                    record = load_trial(_find_record(run_dir, trial_id))
                    self._send_json(200, _record_payload(record))
                else:
                    self._send_json(404, {"error": "route not found"})
            except FileNotFoundError as error:
                self._send_json(404, {"error": str(error)})
            except ValueError as error:
                self._send_json(400, {"error": str(error)})

        def do_POST(self):
            if urlparse(self.path).path != "/api/trials":
                self._send_json(404, {"error": "route not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("request body must be between 1 and 4096 bytes")
                body = json.loads(self.rfile.read(length))
                job = worker.submit(body)
                self._send_json(202, job)
            except FileNotFoundError as error:
                self._send_json(404, {"error": str(error)})
            except ValueError as error:
                self._send_json(400, {"error": str(error)})

        def log_message(self, format, *args):
            print("[pong-http] " + format % args, flush=True)

    return Handler


def serve(run_dir: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    run_dir = Path(run_dir).resolve()
    worker = SimulationWorker(run_dir)
    server = ThreadingHTTPServer((host, port), make_handler(run_dir, worker))
    print(f"Reservoir Pong dashboard: http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        worker.close()
