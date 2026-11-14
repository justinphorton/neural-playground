"""Team-spec Pong with a probe pulse that reads the culture's hidden memory.

    python tasks/pong_probe.py --width 2

The culture forgets a pulse within ~2 ms in its spikes, but neurons hit recently are
still recovering for ~3-12 ms and respond less to a new pulse (what XOR and Lily's
memory_task rely on). So at the commit point we fire a PROBE on all 16 grid electrodes
at once. Its response is weaker wherever the ball passed recently: a snapshot of the
recent trajectory, turned back into spikes. The readout reads that snapshot.

Same game, balls, grid, pulse rule, baseline and test split as tasks/pong_spec.py.
"""

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pong_spec as P  # noqa: E402

PROBE_GAP_FRAMES = 1  # probe fires one frame (2 ms) after the last possible ball pulse
PROBE_MS = P.pulse_ms(P.COMMIT + PROBE_GAP_FRAMES)
TRIAL_MS = int(PROBE_MS + 10)
VARIANT = {"raw": "apply_model_defaults (what every task uses)", "silenced": "raw, all connections cut"}


def _simulate(job):
    stream, trains, silence = job
    from livn.env import Env
    from livn.system import predefined
    env = Env(predefined("EI", download_directory=P.DATA_DIR))
    env.selection(P.N_CELLS, method="patch")
    env.init().apply_model_defaults()
    if silence:
        env.set_params({w: 0.0 for w in env.weight_names})
    env.reseed_noise(stream=stream)
    encoding, decoding = P.BallEncoding(), P.MillisecondCounts(duration=TRIAL_MS)
    probe = [(P.COMMIT + PROBE_GAP_FRAMES, r, c) for r in range(P.ROWS) for c in range(P.COLS)]
    out = [None] * len(trains)
    for i, train in enumerate(trains):
        env.clear()
        out[i] = env(decoding, inputs=train + probe, encoding=encoding)["counts"]
    env.close()
    return out


def simulate(trains, silence):
    chunks = [trains[k::P.WORKERS] for k in range(P.WORKERS)]
    base = random.randint(10**6, 10**8)
    with ProcessPoolExecutor(max_workers=P.WORKERS) as pool:
        parts = list(pool.map(_simulate, [(base + k * 100_000, c, silence) for k, c in enumerate(chunks)]))
    counts = np.zeros((len(trains), 128, TRIAL_MS), dtype=np.uint8)
    for k, part in enumerate(parts):
        counts[k::P.WORKERS] = np.array(part)
    return counts


def window(counts, a, b):
    X = counts[:, :, a:b].sum(2).astype(np.float32)
    return X[:, X.any(0)]


def before_probe(counts):
    """Spike counts per electrode per 5 ms bin, up to the probe (what pong_spec reads)."""
    end = int(PROBE_MS) // P.BIN_MS * P.BIN_MS
    X = counts[:, :, :end].reshape(len(counts), 128, -1, P.BIN_MS).sum(3).reshape(len(counts), -1).astype(np.float32)
    return X[:, X.any(0)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=float, default=2.0)
    parser.add_argument("--balls", type=int, default=P.N_BALLS)
    parser.add_argument("--silence", action="store_true")
    args = parser.parse_args()
    xs = P.make_balls(args.width, args.balls)  # same seed as pong_spec: same balls
    trains = [P.pulse_train(x) for x in xs]
    y = xs[:, -1]
    cut = int(args.balls * P.TRAIN_FRACTION)
    train, test = np.arange(args.balls) < cut, np.arange(args.balls) >= cut
    name = f"probe_width{args.width:g}{'_silenced' if args.silence else ''}"
    print(f"{name}: {args.balls} balls, probe at {PROBE_MS:.0f} ms, {TRIAL_MS} ms trials", flush=True)

    started = time.time()
    counts = simulate(trains, args.silence)
    print(f"simulated in {(time.time() - started) / 60:.1f} min", flush=True)
    os.makedirs(P.OUT_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(P.OUT_DIR, f"{name}.npz"), counts=counts, xs=xs)

    t = int(PROBE_MS)
    H = P.history_features(trains)
    probe_only = window(counts, t, t + 5)
    trajectory = before_probe(counts)
    feats = {
        "line on pulse history (baseline)": H,
        "neurons + line, before the probe only": trajectory,
        "neurons + line, probe response only": probe_only,
        "neurons + line, both": np.hstack([trajectory, probe_only]),
    }
    results = {k: P.score(P.line(X, y, train, test), xs, test) for k, X in feats.items()}
    shuffled = y.copy()
    shuffled[train] = np.random.default_rng(1).permutation(y[train])
    results["CONTROL probe response, shuffled labels"] = P.score(P.line(probe_only, shuffled, train, test), xs, test)
    print(f"\n{'':42s}{'side':>7s}{'opposite':>10s}{'catch':>7s}")
    for k, r in results.items():
        print(f"{k:42s}{r['side']:>7.0%}{r['opposite_side']:>10.0%}{r['catch']:>7.0%}")
    print(f"probe response: {probe_only.sum(1).mean():.0f} spikes per ball in its 5 ms window")
    with open(os.path.join(P.OUT_DIR, f"{name}.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
