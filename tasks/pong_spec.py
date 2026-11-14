"""Reservoir Pong, built from the team's task spec: does the network's response to a falling ball predict where it lands?

    python tasks/pong_spec.py --width 2     # width = court widths the ball can still travel after commit

=============================================================== PLAN
GAME:        a ball falls through a box at a fixed speed, sideways speed varying
             per ball, bouncing off the side walls. The paddle commits at frame
             COMMIT, so chasing the ball is off the table.
INPUT:       absolute ball position only. One pulse each time the ball enters a
             new grid cell; neighbouring cells use neighbouring electrodes. The
             paddle is never encoded. Nothing is delivered after commit.
SYSTEM:      fixed network, no plasticity, no feedback. One trial per ball.
OUTPUT:      spike counts per electrode, skipping BLIND_MS after every pulse, go
             through a straight line that predicts the landing position.
BASELINE:    the same line on the raw pulse history (perfect memory, no noise).
CONTROLS:    follow the ball, shuffled labels, decoding the previous ball.
METRIC:      held-out balls: landing side, opposite-side balls, catch rate.

Differences from the spec, forced by the simulation: the grid is 4 x 4 because the
300-cell patch has 16 strong electrodes (the spec's figure shows about 8 x 4), and
the blind phase is imposed by discarding spikes, since the simulator has no artifact.
===========================================================================
"""

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from livn.decoding import Decoding
from livn.env import Env
from livn.system import predefined
from livn.types import Encoding
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV

# ===========================================================================
# 1. SETTINGS
# ===========================================================================
N_CELLS = 300
AMPLITUDE_UA = 40.0
FALL_FRAMES = 40
COMMIT = 24  # paddle commits here, 60% of the way down
FRAME_MS = 2.0  # neural time per game frame
START_MS = 10.0
TRIAL_MS = int(START_MS + COMMIT * FRAME_MS + 7)  # ends just after the last possible pulse's response
BIN_MS = 5
BLIND_MS = 2.0  # recording is blind this long after each pulse
CATCH_HALF = 0.15  # the paddle catches within this distance of the landing spot
N_BALLS = 2000
TRAIN_FRACTION = 0.7  # earliest balls train; the later block tests
WORKERS = 4

# The 4 x 4 block of strongest electrodes, neighbours next to each other. Row 0 is the top.
SCREEN = [[72, 73, 55, 54],
          [76, 77, 51, 50],
          [124, 125, 3, 2],
          [120, 121, 7, 6]]
COLS, ROWS = 4, 4

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(DATA_DIR, "results", "pong_spec")


# ===========================================================================
# 2. THE GAME
# ===========================================================================
def fold(u):
    u = np.mod(u, 2.0)
    return np.where(u > 1, 2 - u, u)


def make_balls(width, n, seed=0):
    """Positions at every frame. width = sideways distance still possible after commit."""
    rng = np.random.default_rng(seed)
    vmax = width / (FALL_FRAMES - COMMIT)
    x0, v = rng.uniform(0, 1, n), rng.uniform(-vmax, vmax, n)
    return fold(x0[:, None] + v[:, None] * np.arange(FALL_FRAMES + 1))


def cell(x, frame):
    col = min(int(x * COLS), COLS - 1)
    row = min(int(frame / FALL_FRAMES * ROWS), ROWS - 1)
    return row, col


def pulse_train(xs):
    """[(frame, row, col), ...]: one pulse each time the ball enters a new cell, up to commit."""
    pulses, last = [], None
    for f in range(COMMIT):
        c = cell(xs[f], f)
        if c != last:
            pulses.append((f, *c))
            last = c
    return pulses


def pulse_ms(frame):
    return START_MS + frame * FRAME_MS


def history_features(trains):
    """The baseline's input: one on/off flag per (frame, cell). Perfect memory, no noise."""
    F = np.zeros((len(trains), COMMIT * ROWS * COLS), dtype=np.float32)
    for i, train in enumerate(trains):
        for f, r, c in train:
            F[i, f * ROWS * COLS + r * COLS + c] = 1
    return F


# ===========================================================================
# 3. ENCODING + DECODING
# ===========================================================================
class BallEncoding(Encoding):
    amplitude_ua: float = AMPLITUDE_UA
    dt: float = 0.1

    def __call__(self, env, t_end, inputs):
        electrodes = np.zeros((int(t_end / self.dt), env.io.num_channels), dtype=np.float32)
        half = int(0.5 / self.dt)
        for f, r, c in inputs:  # biphasic: negative half, then positive half
            start = int(pulse_ms(f) / self.dt)
            electrodes[start:start + half, SCREEN[r][c]] = -self.amplitude_ua
            electrodes[start + half:start + 2 * half, SCREEN[r][c]] = self.amplitude_ua
        return env.cell_stimulus(electrodes, dt=self.dt)


class MillisecondCounts(Decoding):
    """Keeps spikes per electrode per ms, so bins and the blind phase can change without re-simulating."""
    duration: int = TRIAL_MS

    def setup(self, env):
        env.record_spikes()

    def __call__(self, signal, env=None):
        _, times = env.channel_recording(signal.spike_ids, signal.spike_times)
        edges = np.arange(self.duration + 1)
        return {"counts": np.stack([np.histogram(times.get(ch, []), bins=edges)[0]
                                    for ch in env.io.channel_ids]).astype(np.uint8)}


def _simulate(job):
    stream, trains, weights, cells = job
    env = Env(predefined("EI", download_directory=DATA_DIR))
    env.selection(cells, method="patch")
    env.init().apply_model_defaults()
    if weights:  # optional connection strengths (see --exc-scale)
        env.set_params(weights)
    env.reseed_noise(stream=stream)
    encoding, decoding = BallEncoding(), MillisecondCounts()
    out = []
    for train in trains:
        env.clear()
        out.append(env(decoding, inputs=train, encoding=encoding)["counts"])
    env.close()
    return out


def simulate(trains, weights=None, cells=N_CELLS):
    chunks = [trains[k::WORKERS] for k in range(WORKERS)]
    base = random.randint(10**6, 10**8)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        parts = list(pool.map(_simulate, [(base + k * 100_000, c, weights, cells) for k, c in enumerate(chunks)]))
    counts = np.zeros((len(trains), 128, TRIAL_MS), dtype=np.uint8)
    for k, part in enumerate(parts):
        counts[k::WORKERS] = np.array(part)
    return counts


# ===========================================================================
# 4. DECODING THE RESPONSE
# ===========================================================================
def response_features(counts, trains, blind_ms=BLIND_MS, last_ms=None):
    """Spike counts per electrode per BIN_MS, with spikes in the blind phase thrown away.

    last_ms keeps only the final stretch before commit, to test how much recent memory matters.
    """
    kept = counts.astype(np.float32).copy()
    for i, train in enumerate(trains):
        for f, _, _ in train:
            t = int(pulse_ms(f))
            kept[i, :, t:t + int(np.ceil(blind_ms))] = 0
    if last_ms is not None:
        kept[:, :, :max(0, TRIAL_MS - last_ms)] = 0
    n_bins = TRIAL_MS // BIN_MS
    binned = kept[:, :, :n_bins * BIN_MS].reshape(len(kept), 128, n_bins, BIN_MS).sum(3)
    active = binned.reshape(len(kept), -1).any(0)
    return binned.reshape(len(kept), -1)[:, active]


def score(pred, xs, test):
    """Held-out scores for a landing-position prediction."""
    land, now = xs[test, -1], xs[test, COMMIT]
    side_ok = (pred > 0.5) == (land > 0.5)
    opposite = (land > 0.5) != (now > 0.5)
    return {"side": float(side_ok.mean()), "opposite_side": float(side_ok[opposite].mean()),
            "catch": float((np.abs(pred - land) <= CATCH_HALF).mean()),
            "error": float(np.abs(pred - land).mean())}


def line(X, y, train, test):
    return RidgeCV(alphas=[1, 10, 100, 1000, 10000]).fit(X[train], y[train]).predict(X[test])


# ===========================================================================
# 5. MAIN
# ===========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=float, default=2.0)
    parser.add_argument("--balls", type=int, default=N_BALLS)
    parser.add_argument("--exc-scale", type=float, default=None,
                        help="set excitatory connections to this multiple of livn's tuned EI values")
    parser.add_argument("--inh-scale", type=float, default=1.0)
    parser.add_argument("--silence", action="store_true", help="set every connection to zero (control)")
    parser.add_argument("--cells", type=int, default=N_CELLS, help="neurons in the simulated patch")
    args = parser.parse_args()
    weights, tag = None, ""
    if args.exc_scale is not None or args.silence:
        tuned = json.load(open(os.path.join(DATA_DIR, "systems", "graphs", "EI", "params", "default.json")))
        tuned = tuned["ReducedCalciumSomaDendrite"]["default"]["params"]
        exc = ["EXC_EXC-dend-AMPA-weight", "EXC_EXC-dend-NMDA-weight"]
        inh = ["EXC_INH-soma-GABA_A-weight", "INH_EXC-soma-AMPA-weight"]
        es = 0.0 if args.silence else args.exc_scale
        is_ = 0.0 if args.silence else args.inh_scale
        weights = {k: tuned[k] * es for k in exc} | {k: tuned[k] * is_ for k in inh}
        tag = "_silenced" if args.silence else f"_exc{args.exc_scale:g}_inh{args.inh_scale:g}"
    if args.cells != N_CELLS:
        tag += f"_cells{args.cells}"
    os.makedirs(OUT_DIR, exist_ok=True)

    xs = make_balls(args.width, args.balls)
    trains = [pulse_train(x) for x in xs]
    y = xs[:, -1]
    cut = int(args.balls * TRAIN_FRACTION)
    train, test = np.arange(args.balls) < cut, np.arange(args.balls) >= cut
    print(f"width {args.width}: {args.balls} balls, {np.mean([len(t) for t in trains]):.1f} pulses per ball, "
          f"{TRIAL_MS} ms per trial", flush=True)

    started = time.time()
    counts = simulate(trains, weights, args.cells)
    print(f"simulated in {(time.time() - started) / 60:.1f} min, "
          f"{counts.sum() / args.balls:.0f} spikes per ball", flush=True)
    np.savez_compressed(os.path.join(OUT_DIR, f"width{args.width:g}{tag}.npz"), counts=counts, xs=xs)

    X = response_features(counts, trains)
    H = history_features(trains)
    ceiling = HistGradientBoostingRegressor(max_iter=200).fit(H[train], y[train]).predict(H[test])
    rng = np.random.default_rng(1)
    shuffled = xs.copy()
    shuffled[train, -1] = rng.permutation(y[train])
    results = {
        "follow the ball": score(xs[test, COMMIT], xs, test),
        "line on pulse history (baseline)": score(line(H, y, train, test), xs, test),
        "best decoder on pulse history": score(ceiling, xs, test),
        "neurons + line": score(line(X, y, train, test), xs, test),
        "neurons + line, last 20 ms only": score(line(response_features(counts, trains, last_ms=20),
                                                     y, train, test), xs, test),
        "CONTROL shuffled labels": score(line(X, shuffled[:, -1], train, test), xs, test),
    }
    prev = np.roll(y, 1)
    print(f"\n{'':36s}{'side':>7s}{'opposite':>10s}{'catch':>7s}{'error':>7s}")
    for name, r in results.items():
        print(f"{name:36s}{r['side']:>7.0%}{r['opposite_side']:>10.0%}{r['catch']:>7.0%}{r['error']:>7.2f}")
    r_prev = float(np.corrcoef(line(X, prev, train, test), prev[test])[0, 1])
    print(f"\nCONTROL decoding the previous ball, correlation: {r_prev:.2f} (expected about 0)")
    with open(os.path.join(OUT_DIR, f"width{args.width:g}{tag}.json"), "w") as f:
        json.dump({"results": results, "previous_ball_r": r_prev, "pulses_per_ball": float(np.mean([len(t) for t in trains]))}, f, indent=2)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
