"""Does the boosted culture's network compute? XOR of two pulses on far-apart electrodes.

    python tasks/xor_far.py --cells 1000 --exc-scale 30
    python tasks/xor_far.py --cells 1000 --silence

Pong on a culture needs two things from the network: memory of earlier frames, and
mixing of frames together (so a straight line can read a non-straight answer). The
boosted culture (docs/pong_iterations.md, sections 12-13) has some memory. This checks
mixing, with no game involved.

Electrode A (top-left of the Pong grid) may be pulsed at 10 ms; electrode B (bottom-
right, far away) may be pulsed `delay` ms later. The readout sees spikes in the 10 ms
after B's slot and must answer A XOR B.

  - A straight line on the inputs themselves (A, B) can get at most 75% on XOR.
  - Single neurons can't help here: A and B reach different neurons (section 8 found
    single-cell XOR only on the same or neighbouring electrodes).
  - So a straight line on the culture's spikes beating 75%, and beating the same culture
    with its connections cut, means the network mixed A and B.

Also reported: "remembers A", decoding whether A was pulsed from the same window, which
separates "forgot A" from "remembered A but didn't mix it with B".
"""

import argparse
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import network_bench as N  # noqa: E402

TUNED = json.load(open(N.TUNED))["ReducedCalciumSomaDendrite"]["default"]["params"]
EXC = ["EXC_EXC-dend-AMPA-weight", "EXC_EXC-dend-NMDA-weight"]
INH = ["EXC_INH-soma-GABA_A-weight", "INH_EXC-soma-AMPA-weight"]
A_EL, B_EL = 72, 6  # opposite corners of the Pong grid
A_MS = 10
DELAYS = [5, 10, 20, 40]
WINDOW = 10
TRIAL_MS = A_MS + max(DELAYS) + WINDOW + 5
DT = 0.1


def _job(args):
    stream, trials, weights, cells = args
    from livn.env import Env
    from livn.system import predefined
    env = Env(predefined("EI", download_directory=N.DATA_DIR))
    env.selection(cells, method="patch")
    env.init()
    env.apply_model_defaults()
    if weights is not None:
        env.set_params(weights)
    env.record_spikes()
    env.reseed_noise(stream=stream)
    out = []
    for a, b, delay in trials:
        current = np.zeros((int(TRIAL_MS / DT), env.io.num_channels), dtype=np.float32)
        for on, el, t in [(a, A_EL, A_MS), (b, B_EL, A_MS + delay)]:
            if on:
                s = int(t / DT)
                current[s:s + 5, el] = -40.0
                current[s + 5:s + 10, el] = 40.0
        env.clear()
        rec = env.run(TRIAL_MS, env.cell_stimulus(current, dt=DT))
        _, times = env.channel_recording(rec.spike_ids, rec.spike_times)
        edges = np.arange(TRIAL_MS + 1)
        out.append(np.stack([np.histogram(times.get(ch, []), bins=edges)[0] for ch in env.io.channel_ids]).astype(np.uint16))
    env.close()
    return out


def run(trials, weights, cells):
    chunks = [trials[k::N.WORKERS] for k in range(N.WORKERS)]
    base = random.randint(10**6, 10**8)
    with ProcessPoolExecutor(N.WORKERS) as pool:
        parts = list(pool.map(_job, [(base + k * 100_000, c, weights, cells) for k, c in enumerate(chunks)]))
    C = np.zeros((len(trials), 128, TRIAL_MS))
    for k, part in enumerate(parts):
        C[k::N.WORKERS] = np.array(part)
    return C


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=1000)
    parser.add_argument("--exc-scale", type=float, default=30)
    parser.add_argument("--silence", action="store_true")
    parser.add_argument("--reps", type=int, default=40)
    args = parser.parse_args()
    if args.silence:
        weights, tag = {k: 0.0 for k in EXC + INH}, f"silenced_cells{args.cells}"
    else:
        weights = {k: TUNED[k] * args.exc_scale for k in EXC} | {k: TUNED[k] for k in INH}
        tag = f"exc{args.exc_scale:g}_cells{args.cells}"

    trials = [(a, b, d) for d in DELAYS for a in (0, 1) for b in (0, 1) for _ in range(args.reps)]
    random.Random(0).shuffle(trials)
    print(f"xor_far {tag}: {len(trials)} trials of {TRIAL_MS} ms", flush=True)
    C = run(trials, weights, args.cells)
    out_dir = os.path.join(N.DATA_DIR, "results", "xor_far")
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{tag}.npz"), counts=C, trials=np.array(trials))

    T = np.array(trials)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    clf = LogisticRegression(max_iter=3000, C=0.5)
    rows = {}
    print(f"\n{'delay':>6}{'XOR':>7}{'remembers A':>13}{'spikes in window':>18}")
    for d in DELAYS:
        m = T[:, 2] == d
        t0 = A_MS + d
        X = np.hstack([C[m][:, :, t0:t0 + 5].sum(2), C[m][:, :, t0 + 5:t0 + WINDOW].sum(2)])
        X = X[:, X.any(0)]
        a, b = T[m, 0], T[m, 1]
        xor = float(np.mean(cross_val_score(clf, X, a ^ b, cv=cv)))
        mem = float(np.mean(cross_val_score(clf, X, a, cv=cv)))
        rows[d] = {"xor": xor, "remembers_a": mem, "spikes": float(X.sum(1).mean())}
        print(f"{d:>4}ms{xor:>7.0%}{mem:>13.0%}{X.sum(1).mean():>18.0f}", flush=True)
    print("XOR from the inputs with a straight line: at most 75%. Chance: 50%.")
    with open(os.path.join(out_dir, f"{tag}.json"), "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
