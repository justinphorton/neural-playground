"""Small local web checker for the frozen C4-vs-E4 development readout.

The page is intentionally local-only and dependency-free.  An uploaded WAV is
run through the same fixed-reservoir simulation used for the experiment, then
the saved neural-only readout receives the 205--255 ms post-stimulus features.
This is a convenience checker, not the external-piano validation study.
"""

from __future__ import annotations

import argparse
import cgi
import hashlib
import json
import tempfile
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np

from .experiment_config import ExperimentConfig
from .fixed_reservoir import FeatureStage, SimulationStage, TrialSpec, deterministic_seed
from .reservoir_artifact import ReservoirManifest


MAX_UPLOAD_BYTES = 50 * 1024 * 1024
DEFAULT_REPEATS = 3
SUPPORTED_NOTES = ("C4", "E4")


def summarize_probabilities(class_order: Sequence[str], probabilities: np.ndarray) -> dict[str, Any]:
    """Average repeat probabilities and return a JSON-friendly prediction."""
    classes = [str(value) for value in class_order]
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(classes) or values.shape[0] < 1:
        raise ValueError("probabilities must be a non-empty (repeats, classes) matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("probabilities contain non-finite values")
    means = values.mean(axis=0)
    return {
        "predicted_label": classes[int(np.argmax(means))],
        "class_order": classes,
        "probabilities": {label: float(means[index]) for index, label in enumerate(classes)},
        "repeat_count": int(values.shape[0]),
        "repeat_predictions": [classes[int(np.argmax(row))] for row in values],
    }


def score_prediction(predicted_label: str, correct_note: str) -> dict[str, Any]:
    """Score the displayed prediction against the user's known note label."""
    expected = str(correct_note).strip().upper()
    predicted = str(predicted_label).strip().upper()
    if expected not in SUPPORTED_NOTES:
        raise ValueError("correct_note must be C4 or E4")
    if predicted not in SUPPORTED_NOTES:
        raise ValueError("predicted_label must be C4 or E4")
    return {"correct_note": expected, "is_correct": predicted == expected}


@dataclass(frozen=True)
class ModelBundle:
    experiment_dir: Path
    model: Any
    config: ExperimentConfig
    reservoir: ReservoirManifest
    metadata: dict[str, Any]

    @property
    def class_order(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.metadata["class_order"])

    @property
    def feature_mode(self) -> str:
        return str(self.metadata["feature_mode"])

    @property
    def window_ms(self) -> tuple[float, float]:
        return tuple(float(value) for value in self.metadata["window_ms"])

    @property
    def recording_electrodes(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.metadata["recording_electrodes"])

    @property
    def amplitude_ua(self) -> float:
        return float(self.metadata["selected_amplitude_ua"])

    @classmethod
    def load(cls, experiment_dir: Path) -> "ModelBundle":
        experiment_dir = Path(experiment_dir).resolve()
        config_path = experiment_dir / "config.json"
        reservoir_path = experiment_dir / "reservoir" / "reservoir_manifest.json"
        model_path = experiment_dir / "models" / "fixed_readout.joblib"
        metadata_path = model_path.with_suffix(model_path.suffix + ".json")
        required = (config_path, reservoir_path, model_path, metadata_path)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("model bundle is incomplete: " + ", ".join(missing))

        config = ExperimentConfig.read(config_path)
        reservoir = ReservoirManifest.read(reservoir_path)
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("config_hash") != config.config_hash:
            raise ValueError("saved model config hash does not match experiment config")
        if metadata.get("reservoir_hash") != reservoir.reservoir_hash:
            raise ValueError("saved model reservoir hash does not match reservoir manifest")
        if tuple(metadata.get("class_order", ())) != ("C4", "E4"):
            raise ValueError("website checker requires a C4/E4 model")
        if not (0 < float(metadata.get("selected_amplitude_ua", 0)) <= 25):
            raise ValueError("saved model amplitude must be in (0, 25] uA")
        electrodes = tuple(int(value) for value in metadata.get("recording_electrodes", ()))
        if electrodes != tuple(reservoir.recording_electrodes) or len(electrodes) != 8:
            raise ValueError("saved model electrodes do not match the canonical eight-electrode layout")
        window = tuple(float(value) for value in metadata.get("window_ms", ()))
        if len(window) != 2 or not (5 <= window[0] < window[1] <= 1000):
            raise ValueError("saved model window is not a post-stimulus memory window")

        model = joblib.load(model_path)
        if not hasattr(model, "predict_proba"):
            raise ValueError("saved model does not expose predict_proba")
        return cls(experiment_dir, model, config, reservoir, metadata)


class InferenceService:
    """Serialize simulation access while reusing one canonical reservoir."""

    def __init__(self, bundle: ModelBundle, repeats: int = DEFAULT_REPEATS):
        if repeats < 1:
            raise ValueError("repeats must be positive")
        self.bundle = bundle
        self.repeats = int(repeats)
        self.stage = SimulationStage(bundle.config, bundle.experiment_dir, bundle.experiment_dir / "reservoir")
        self._lock = threading.Lock()

    def close(self) -> None:
        self.stage.close()

    def predict_bytes(self, data: bytes, filename: str = "upload.wav", correct_note: str | None = None) -> dict[str, Any]:
        if not data:
            raise ValueError("the uploaded WAV is empty")
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError(f"the uploaded WAV exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit")
        safe_name = Path(filename).name or "upload.wav"
        with tempfile.NamedTemporaryFile(prefix="piano-upload-", suffix=".wav") as handle:
            handle.write(data)
            handle.flush()
            return self.predict_path(Path(handle.name), safe_name, hashlib.sha256(data).hexdigest(), correct_note)

    def predict_path(self, wav_path: Path, filename: str, file_hash: str | None = None, correct_note: str | None = None) -> dict[str, Any]:
        file_hash = file_hash or hashlib.sha256(Path(wav_path).read_bytes()).hexdigest()
        recording_id = f"upload_{file_hash[:16]}"
        specs = [TrialSpec(
            trial_id=f"{recording_id}_r{repeat:02d}_{self.bundle.amplitude_ua:g}uA",
            recording_id=recording_id,
            wav_path=str(wav_path),
            label="UNKNOWN",
            condition_group_id=recording_id,
            amplitude_ua=self.bundle.amplitude_ua,
            repeat=repeat,
            noise_seed=deterministic_seed(recording_id, repeat, base_seed=20_000, amplitude_ua=self.bundle.amplitude_ua),
            expected_reservoir_hash=self.bundle.reservoir.reservoir_hash,
        ) for repeat in range(self.repeats)]

        with self._lock:
            trials = [self.stage.run_trial(spec) for spec in specs]
        features = FeatureStage(self.bundle.recording_electrodes).from_trials(
            trials, self.bundle.window_ms, self.bundle.feature_mode,
        )
        probabilities = np.asarray(self.bundle.model.predict_proba(features.X), dtype=float)
        result = summarize_probabilities(self.bundle.class_order, probabilities)
        result.update({
            "filename": filename,
            "file_sha256": file_hash,
            "model_status": self.bundle.metadata.get("status", "unknown"),
            "experiment": self.bundle.experiment_dir.name,
            "selected_amplitude_ua": self.bundle.amplitude_ua,
            "feature_mode": self.bundle.feature_mode,
            "window_ms": list(self.bundle.window_ms),
            "reservoir_hash": self.bundle.reservoir.reservoir_hash,
            "config_hash": self.bundle.config.config_hash,
            "note": "Development-model check; this is not external-piano validation.",
        })
        if correct_note is not None:
            result.update(score_prediction(result["predicted_label"], correct_note))
        return result


def build_upload_page() -> str:
    """Return the standalone HTML UI served by the local checker."""
    return r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fixed Reservoir Piano Check</title>
  <style>
    :root { color-scheme: dark; font-family: ui-sans-serif, system-ui, sans-serif; }
    body { margin: 0; min-height: 100vh; background: #10131a; color: #edf0f7; display: grid; place-items: center; }
    main { width: min(720px, calc(100% - 36px)); padding: 34px; background: #191e29; border: 1px solid #2e3748; border-radius: 20px; box-shadow: 0 24px 80px #0006; }
    h1 { margin: 0 0 10px; font-size: clamp(1.7rem, 4vw, 2.4rem); }
    p { color: #aeb8c9; line-height: 1.55; }
    .notice { padding: 13px 15px; border-radius: 12px; background: #342d18; color: #f6d992; border: 1px solid #6a592d; font-size: .92rem; }
    form { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; margin: 26px 0 10px; }
    input[type=file] { flex: 1 1 320px; padding: 13px; background: #11151e; border: 1px solid #3a465b; border-radius: 10px; color: #edf0f7; }
    button { border: 0; border-radius: 10px; padding: 14px 20px; color: #0c1118; background: #8ce0bf; font-weight: 750; cursor: pointer; }
    button:disabled { opacity: .55; cursor: wait; }
    #status { min-height: 25px; color: #aeb8c9; }
    .result { margin-top: 22px; padding: 20px; border-radius: 14px; background: #11151e; border: 1px solid #2e3748; }
    .label { font-size: 2.2rem; font-weight: 800; color: #8ce0bf; }
    .prob { display: flex; justify-content: space-between; margin-top: 8px; color: #d5dcea; }
    code { color: #c9d4e8; word-break: break-all; }
    .small { font-size: .84rem; }
  </style>
</head>
<body><main>
  <h1>Fixed-reservoir piano check</h1>
  <p>Upload a one-note WAV. The server runs three stochastic trials through the frozen reservoir, reads only the post-stimulus neural spikes, and averages the saved C4/E4 probabilities.</p>
  <div class="notice">This is a <strong>development model</strong> trained on the two current piano families. It is not evidence of generalization to a new piano yet.</div>
  <form id="form"><input id="audio" name="audio" type="file" accept=".wav,audio/wav" required><label>Known note <select id="correct_note" name="correct_note"><option value="C4">C4</option><option value="E4">E4</option></select></label><button id="submit" type="submit">Run reservoir check</button></form>
  <div id="status" role="status"></div><section id="result" class="result" hidden></section>
  <p class="small">Only WAV input is accepted. The deployed model is fixed at 3.125 µA and the 205–255 ms fine-temporal window.</p>
  <script>
    const form = document.getElementById('form'), input = document.getElementById('audio'), button = document.getElementById('submit'), status = document.getElementById('status'), result = document.getElementById('result');
    form.addEventListener('submit', async (event) => {
      event.preventDefault(); if (!input.files.length) return;
      button.disabled = true; result.hidden = true; status.textContent = 'Running three reservoir trials…';
      try {
        const body = new FormData(); body.append('audio', input.files[0]); body.append('correct_note', document.getElementById('correct_note').value);
        const response = await fetch('/predict', { method: 'POST', body });
        const payload = await response.json(); if (!response.ok) throw new Error(payload.error || 'prediction failed');
        const probs = Object.entries(payload.probabilities).map(([label, value]) => `<div class="prob"><span>${label}</span><strong>${(value * 100).toFixed(1)}%</strong></div>`).join('');
        const verdict = payload.is_correct ? 'MATCH — the model guessed the supplied note.' : 'MISMATCH — the model did not guess the supplied note.';
        result.innerHTML = `<div class="small">Prediction for ${payload.filename}</div><div class="label">${payload.predicted_label}</div><p><strong>${verdict}</strong><br>Known note: ${payload.correct_note}</p>${probs}<p class="small">${payload.repeat_count} repeats · ${payload.window_ms[0]}–${payload.window_ms[1]} ms · reservoir <code>${payload.reservoir_hash.slice(0, 12)}…</code></p>`;
        result.hidden = false; status.textContent = 'Complete.';
      } catch (error) { status.textContent = error.message; }
      finally { button.disabled = false; }
    });
  </script>
</main></body></html>'''


class PianoNotesRequestHandler(BaseHTTPRequestHandler):
    server_version = "FixedReservoirPiano/1.0"

    @property
    def service(self) -> InferenceService:
        return self.server.inference_service  # type: ignore[attr-defined]

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler interface
        if self.path in {"/", "/index.html"}:
            self._send(200, build_upload_page().encode(), "text/html; charset=utf-8")
        else:
            self._send(404, b"Not found\n", "text/plain; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler interface
        if self.path != "/predict":
            self._send(404, b'{"error":"Not found"}', "application/json")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_UPLOAD_BYTES + 2 * 1024 * 1024:
                raise ValueError("invalid or oversized upload")
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={
                "REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                "CONTENT_LENGTH": str(length),
            })
            field = form["audio"] if "audio" in form else None
            if field is None or not getattr(field, "file", None):
                raise ValueError("choose a WAV file")
            data = field.file.read(MAX_UPLOAD_BYTES + 1)
            correct_note = form.getfirst("correct_note", "")
            if str(correct_note).strip().upper() not in SUPPORTED_NOTES:
                raise ValueError("choose C4 or E4 as the known note")
            result = self.service.predict_bytes(data, getattr(field, "filename", None) or "upload.wav", correct_note)
            self._send(200, json.dumps(result).encode(), "application/json; charset=utf-8")
        except Exception as exc:  # return a useful browser error without a traceback page
            self._send(400, json.dumps({"error": str(exc)}).encode(), "application/json; charset=utf-8")

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[piano-web] {format % args}")


def find_latest_experiment(root: Path = Path("experiments")) -> Path:
    candidates = [path.parent.parent for path in Path(root).glob("*/models/fixed_readout.joblib")]
    if not candidates:
        raise FileNotFoundError(f"no trained fixed_readout.joblib found below {Path(root).resolve()}")
    return max(candidates, key=lambda path: path.name)


def run_server(experiment: Path, host: str = "127.0.0.1", port: int = 8765, repeats: int = DEFAULT_REPEATS) -> None:
    bundle = ModelBundle.load(experiment)
    service = InferenceService(bundle, repeats=repeats)
    server = ThreadingHTTPServer((host, port), PianoNotesRequestHandler)
    server.inference_service = service  # type: ignore[attr-defined]
    print(f"Fixed-reservoir piano checker: http://{host}:{port}/")
    print(f"Model: {bundle.experiment_dir / 'models' / 'fixed_readout.joblib'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping piano checker.")
    finally:
        server.server_close()
        service.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=None, help="completed experiment directory")
    parser.add_argument("--root", type=Path, default=Path("experiments"), help="search root when --experiment is omitted")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    args = parser.parse_args(argv)
    experiment = args.experiment or find_latest_experiment(args.root)
    run_server(experiment, args.host, args.port, args.repeats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
