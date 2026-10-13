"""Web playground: draw a stimulation pattern, watch the neurons and readout work.

Run from the repo folder:
    python playground/web.py demo_xor              # any task file in tasks/ written like demo_xor.py
    python playground/web.py demo_xor --retrain    # throw away the saved readout, train again

Opens a page in your browser (http://localhost:8765). The first time, it trains
the linear regression readout (~3 min) and saves it to playground/saved_<task>.pkl;
after that it starts instantly. Ctrl+C in the terminal stops it.

Hosting (e.g. Render): if the PORT environment variable is set, it listens on
0.0.0.0:$PORT instead and doesn't try to open a browser.

How it's split:
    web.py    (this file) runs the real livn simulation + readout
    web.html  draws everything; talks to this file over two URLs:
                       GET  /api/setup  -> the task, the culture layout, preset patterns
                       POST /api/run    -> run one trial with YOUR pulses, return everything

A task file in tasks/ plugs in if it has (demo_xor.py and template_task.py both do):
    INPUTS, target(), TRIAL_MS, make_dataset(), build_reservoir(),
    strongest_electrodes(), make_pipeline(), collect_trials(),
    and an Encoding with a pulses(input) -> [(time_ms, electrode), ...] method.
"""

import importlib
import json
import os
import pickle
import random
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from livn.types import Encoding
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict

HERE = os.path.dirname(os.path.abspath(__file__))  # playground/
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tasks"))  # so task files in tasks/ can be imported
# Hosting services (e.g. Render) tell the app which port to use through $PORT.
# Locally that variable isn't set, so we use localhost:8765 and open your browser.
HOSTED = "PORT" in os.environ
PORT = int(os.environ.get("PORT", 8765))


MAX_MS = 1000  # longest trial the page can ask for


class ManualPulses(Encoding):
    """The simplest possible encoding: you hand it the pulses directly.

    inputs = [(time_ms, electrode, strength_uA), ...]. Every task encoding boils
    down to this; it just decides the pulses from the task input for you.
    """

    dt: float = 0.1

    def __call__(self, env, t_end, inputs):
        electrodes = np.zeros((int(t_end / self.dt), env.io.num_channels), dtype=np.float32)
        for t, ch, amp in inputs:
            start = int(t / self.dt)
            electrodes[start:start + int(1.0 / self.dt), int(ch)] = amp
        return env.cell_stimulus(electrodes, dt=self.dt)


def train_or_load(task, env, encoding, decoding, retrain=False):
    """Train the readout once, then reuse it from disk.

    Saves {"readout", "X", "y", "inputs"}: the trained linear regression plus
    the training trials, so the playground can plot them.
    """
    path = os.path.join(HERE, f"saved_{task.__name__}.pkl")
    if os.path.exists(path) and not retrain:
        with open(path, "rb") as f:
            saved = pickle.load(f)
        if isinstance(saved, dict):  # older saves were just the model: retrain those
            return saved

    dataset = task.make_dataset()
    print(f"Training the readout on {len(dataset)} trials (one time only, ~2 s each)...")
    X, y = task.collect_trials(env, encoding, decoding, dataset)[:2]
    readout = LinearRegression().fit(X, y)  # the ONLY training in the whole pipeline
    saved = {"readout": readout, "X": X, "y": y, "inputs": np.array(dataset, dtype=float)}
    with open(path, "wb") as f:
        pickle.dump(saved, f)
    print(f"Saved trained readout to {path}")
    return saved


def line_pictures(saved):
    """Data for the "can a straight line split it?" plots.

    raw:   linear regression fit on the raw inputs: its score for each task input
    cloud: every training trial as (readout score, total spikes, answer). Scores
           are cross-validated: each trial is scored by a readout trained WITHOUT
           it, so this shows how new trials land, not memorized ones.
    """
    X, y, inputs = saved["X"], saved["y"], saved["inputs"]
    raw_fit = LinearRegression().fit(inputs, y)
    unique = np.unique(inputs, axis=0)
    raw = [{"input": u.tolist(), "score": float(raw_fit.predict([u])[0]),
            "answer": int(y[(inputs == u).all(axis=1)][0])} for u in unique]
    scores = cross_val_predict(LinearRegression(), X, y, cv=5)
    cloud = [[round(float(s), 3), int(t), int(a)] for s, t, a in zip(scores, X.sum(axis=1), y)]
    return {"raw": raw, "raw_coef": raw_fit.coef_.tolist(), "raw_start": float(raw_fit.intercept_),
            "cloud": cloud, "best_line": best_line_accuracy(inputs, y)}


def best_line_accuracy(inputs, y, n_directions=4000):
    """The best accuracy ANY single straight cut through the raw inputs can get on this task.

    This is the bar the neurons have to beat: 0.75 for XOR, 1.0 for a task like AND.
    Tries many cut directions; for each, every possible threshold, labelling either side "1".
    """
    unique, counts = np.unique(inputs, axis=0, return_counts=True)
    labels = np.array([y[(inputs == u).all(axis=1)][0] for u in unique])
    rng = np.random.default_rng(0)
    directions = np.vstack([np.eye(unique.shape[1]), rng.normal(size=(n_directions, unique.shape[1]))])
    best = 0.0
    for w in directions:
        order = np.argsort(unique @ w)
        lab, cnt = labels[order], counts[order]
        for k in range(len(lab) + 1):   # cut between point k-1 and k
            right = cnt[:k][lab[:k] == 0].sum() + cnt[k:][lab[k:] == 1].sum()
            best = max(best, right, cnt.sum() - right)
    return float(best / counts.sum())


def fresh_noise(env):
    """Jump to background noise the readout has never seen.

    Every fresh start numbers its noise draws 1, 2, 3, ... and env.clear()
    moves to the next one, so without this, the first 100 trials here would
    be exact replays of the 100 training trials (and look suspiciously perfect).
    """
    env.reseed_noise(stream=random.randint(10**6, 10**9))


class Lab:
    """Holds the network, the task's encoding/decoding, and the trained readout."""

    def __init__(self, task, retrain=False):
        self.task = task
        print("Building the reservoir...")
        self.env = task.build_reservoir()
        self.encoding, self.decoding = task.make_pipeline(self.env)
        saved = train_or_load(task, self.env, self.encoding, self.decoding, retrain)
        self.readout = saved["readout"]
        self.pictures = line_pictures(saved)
        fresh_noise(self.env)

        # The task's inputs, written as the pulses its encoding would deliver
        self.presets = [
            {"label": str(x), "input": list(x), "answer": int(task.target(x)),
             "pulses": [[float(t), int(ch), float(self.encoding.amplitude_ua)]
                        for t, ch in self.encoding.pulses(x)]}
            for x in task.INPUTS
        ]

        # Timeline rows: the electrodes the task uses, then a few strong neighbours
        used = sorted({pulse[1] for p in self.presets for pulse in p["pulses"]})
        extra = [ch for ch in task.strongest_electrodes(self.env, n=6) if ch not in used]
        self.rows = (used + extra)[:4]

        # Where our neurons and electrodes sit in the dish (micrometers)
        gids = set(int(g) for g in self.env.simulated_gids())
        coords = np.asarray(self.env.active_neuron_coordinates())
        coords = coords[[int(g) in gids for g in coords[:, 0]]]
        self.gid_index = {int(g): i for i, g in enumerate(coords[:, 0])}
        self.xy = coords[:, 1:3]
        elec = np.asarray(self.env.io.electrode_coordinates)
        lo, hi = self.xy.min(axis=0) - 100, self.xy.max(axis=0) + 100
        self.in_view = [int(e[0]) for e in elec if (lo <= e[1:3]).all() and (e[1:3] <= hi).all()]
        self.elec_xy = {int(e[0]): e[1:3] for e in elec}

        # Raster order: nearest neuron to the main electrode at the bottom
        distance = np.linalg.norm(self.xy - self.elec_xy[self.rows[0]], axis=1)
        self.raster_row = np.argsort(np.argsort(distance))

    def weight_grid(self):
        """The readout's learned weights laid out like the fingerprint: [electrode in view][time window].

        None if this task's features aren't electrode x time-window counts.
        """
        coef, n_ch = np.asarray(self.readout.coef_), self.env.io.num_channels
        if coef.size % n_ch:
            return None
        return coef.reshape(n_ch, -1)[self.in_view].round(4).tolist()

    def setup(self):
        return {
            "task": self.task.__name__,
            "about": (self.task.__doc__ or "").strip().splitlines()[0],
            "trial_ms": self.task.TRIAL_MS,
            "max_ms": MAX_MS,
            "amplitude": float(getattr(self.encoding, "amplitude_ua", 40.0)),
            "presets": self.presets,
            "rows": self.rows,
            "neurons": [[round(float(x), 1), round(float(y), 1)] for x, y in self.xy],
            "electrodes": [[ch, *map(float, self.elec_xy[ch])] for ch in self.in_view],
            "n_features": int(self.readout.n_features_in_),
            "n_channels": int(self.env.io.num_channels),
            "weights": self.weight_grid(),
            "bin_ms": self.task.TRIAL_MS * self.env.io.num_channels / self.readout.n_features_in_,
            "page": getattr(self.task, "PLAYGROUND", {}),   # the task's own page text (optional)
            "n_train": int(len(self.pictures["cloud"])),
            **self.pictures,
        }

    def run(self, pulses, duration):
        # Same steps as one trial in demo_xor.py (reset, encode -> simulate -> decode), except we
        # may simulate LONGER than the task's trial, to watch the network settle. The readout
        # still only sees the first TRIAL_MS (the "readout window"), exactly like in training.
        duration = float(np.clip(duration, self.task.TRIAL_MS, MAX_MS))
        self.env.clear()
        self.decoding.setup(self.env)
        stimulus = ManualPulses()(self.env, duration, pulses)
        recording = self.env.run(duration, stimulus)
        out = self.decoding(recording.slice(0, self.task.TRIAL_MS), self.env)
        features = np.asarray(out["features"], dtype=float)

        # The readout is just: output = start value + sum(weight * feature)
        votes = self.readout.coef_ * features
        output = float(self.readout.intercept_ + votes.sum())

        # Group features/votes by time window (works for electrode x time-bin features)
        n_ch = self.env.io.num_channels
        if features.size % n_ch == 0:
            n_bins = features.size // n_ch
            grid = features.reshape(n_ch, n_bins)[self.in_view].tolist()
            window_votes = votes.reshape(n_ch, n_bins).sum(axis=0)
        else:  # some other kind of features: show them in 12 groups
            n_bins, grid = 12, None
            window_votes = np.array([g.sum() for g in np.array_split(votes, n_bins)])

        # Does this pattern match one of the task's inputs exactly?
        def key(ps):
            return sorted((round(p[0], 1), int(p[1]), round(p[2])) for p in ps)

        match = next((p for p in self.presets if key(p["pulses"]) == key(pulses)), None)

        spikes = [[int(self.raster_row[self.gid_index[int(g)]]), self.gid_index[int(g)], round(float(t), 2)]
                  for g, t in zip(recording.spike_ids, recording.spike_times) if int(g) in self.gid_index]
        return {
            "spikes": spikes,  # [raster row, neuron index, time ms]
            "grid": grid,  # spike counts: [electrode in view][time window]
            "bin_ms": self.task.TRIAL_MS / n_bins,
            "duration": duration,
            "start": float(self.readout.intercept_),
            "votes": window_votes.round(4).tolist(),
            "output": output,
            "total": int(features.sum()),  # total spike count, for the "you are here" dot
            "prediction": int(output > 0.5),
            "match": match["label"] if match else None,
            "answer": match["answer"] if match else None,
        }

    def repeat(self, pulses, n=10):
        """Run the same pulses n times (fresh background noise each time) and collect the readout's scores.

        One trial is one noisy sample; the spread over many trials shows whether an answer is reliable,
        and whether a change to the input matters more than the noise does.
        """
        trials = [self.run(pulses, self.task.TRIAL_MS) for _ in range(min(max(n, 1), 20))]
        return {
            "scores": [round(t["output"], 3) for t in trials],
            "predictions": [t["prediction"] for t in trials],
            "match": trials[0]["match"],
            "answer": trials[0]["answer"],
        }


def serve(lab):
    class Handler(BaseHTTPRequestHandler):
        def send(self, body, kind="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.startswith("/api/setup"):
                self.send(lab.setup())
            else:
                with open(os.path.join(HERE, "web.html"), "rb") as f:
                    self.send(f.read(), "text/html; charset=utf-8")

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            pulses = [(float(t), int(ch), float(amp)) for t, ch, amp in body["pulses"]]
            if self.path.startswith("/api/repeat"):
                self.send(lab.repeat(pulses, int(body.get("n", 10))))
            else:
                self.send(lab.run(pulses, float(body.get("duration", 0))))

        def log_message(self, *args):  # keep the terminal quiet
            pass

    if HOSTED:  # on a host like Render: listen on its port, accept outside visitors, no browser to open
        print(f"Playground serving on port {PORT}")
        HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
    url = f"http://localhost:{PORT}"
    print(f"Playground running at {url}  (Ctrl+C to stop)")
    webbrowser.open(url)
    HTTPServer(("localhost", PORT), Handler).serve_forever()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    task = importlib.import_module(os.path.basename(sys.argv[1]).removesuffix(".py"))
    lab = Lab(task, retrain="--retrain" in sys.argv)
    try:
        serve(lab)
    except KeyboardInterrupt:
        lab.env.close()


if __name__ == "__main__":
    main()
