"""Turn up the culture's connections and measure memory: can livn's EI culture become a recurrent reservoir?

    python tasks/recurrence_sweep.py

The software reservoir sweep (tasks/pong_reservoir_sweep.py) showed the team-spec Pong
needs memory carried by recurrent connections. In the culture every task uses, cutting
the connections changes nothing, so here we scale the connection strengths up and
measure, with no game involved:

  spontaneous   firing with no stimulus (a runaway network fires everywhere)
  late spikes   evoked spikes 5-40 ms after a pulse, above background
  memory        which of 16 electrodes was pulsed, decoded 5, 10, 20 and 40 ms later
  latched       with --trial-ms 300 or more: fraction of pulsed trials still firing far
                above background in the last 50 ms. A boosted network can lock "on" after
                one pulse and never stop, which looks like memory in short trials but is
                a 1-bit switch (docs/pong_iterations.md, section 14)

A setting is only useful if late activity still says where it started. Pong is run
only on a setting that passes, so the network is chosen by its properties, not by a
task score.
"""

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
REPS = 15
TRIAL_MS = 80
DELAYS = [5, 10, 20, 40]


def settings(excs=(1, 3, 10, 30), inhs=(1, 0.3), include_raw=True, nmda=None):
    """(name, base loader, weight overrides). Weights are set as multiples of livn's tuned values.

    nmda: if given, the slow NMDA synapses get this multiple and `excs` scales only the fast
    AMPA synapses (NMDA is the classic driver of self-sustaining activity)."""
    out = [("raw (what every task uses)", "model", {})] if include_raw else []
    for exc in excs:
        for inh in inhs:
            w = {k: TUNED[k] * exc for k in EXC}
            label = f"excitation x{exc}"
            if nmda is not None:
                w["EXC_EXC-dend-NMDA-weight"] = TUNED["EXC_EXC-dend-NMDA-weight"] * nmda
                label = f"AMPA x{exc}, NMDA x{nmda}"
            w.update({k: TUNED[k] * inh for k in INH})
            out.append((f"raw + tuned weights, {label}, inhibition x{inh}", "model", w))
    return out


def _job(args):
    loader, weights, stream, trials, cells, trial_ms = args
    from livn.env import Env
    from livn.system import predefined
    env = Env(predefined("EI", download_directory=N.DATA_DIR))
    env.selection(cells, method="patch")
    env.init()
    env.apply_model_defaults() if loader == "model" else env.apply_default_params()
    if weights:
        env.set_params(weights)
    env.record_spikes()
    env.reseed_noise(stream=stream)
    out = []
    for electrode in trials:
        current = np.zeros((int(trial_ms / 0.1), env.io.num_channels), dtype=np.float32)
        if electrode is not None:
            s = int(N.PULSE_MS / 0.1)
            current[s:s + 5, electrode] = -40.0
            current[s + 5:s + 10, electrode] = 40.0
        env.clear()
        rec = env.run(trial_ms, env.cell_stimulus(current, dt=0.1))
        _, times = env.channel_recording(rec.spike_ids, rec.spike_times)
        edges = np.arange(trial_ms + 1)
        out.append(np.stack([np.histogram(times.get(ch, []), bins=edges)[0] for ch in env.io.channel_ids]).astype(np.uint16))
    env.close()
    return out


def run(loader, weights, cells=300, trial_ms=TRIAL_MS):
    trials = [None] * 20 + [e for e in N.SCREEN for _ in range(REPS)]
    chunks = [trials[k::N.WORKERS] for k in range(N.WORKERS)]
    base = random.randint(10**6, 10**8)
    with ProcessPoolExecutor(N.WORKERS) as pool:
        parts = list(pool.map(_job, [(loader, weights, base + k * 100_000, c, cells, trial_ms) for k, c in enumerate(chunks)]))
    C = np.zeros((len(trials), 128, trial_ms))
    for k, part in enumerate(parts):
        C[k::N.WORKERS] = np.array(part)
    return C


def main():
    global REPS
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exc", type=float, nargs="+", default=[1, 3, 10, 30])
    parser.add_argument("--inh", type=float, nargs="+", default=[1, 0.3])
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--out", default="recurrence_sweep.json")
    parser.add_argument("--cells", type=int, default=300)
    parser.add_argument("--trial-ms", type=int, default=TRIAL_MS)
    parser.add_argument("--nmda", type=float, default=None, help="scale NMDA separately; --exc then scales AMPA only")
    args = parser.parse_args()
    REPS = args.reps
    grid = settings(args.exc, args.inh, include_raw=args.exc == [1, 3, 10, 30], nmda=args.nmda)
    t0 = int(N.PULSE_MS)
    labels = np.repeat(np.arange(16), REPS)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    rows = []
    print(f"{'setting':50s}{'spont Hz':>9s}{'late':>7s}{'latched':>9s}  memory: " + "  ".join(f"{d}ms" for d in DELAYS), flush=True)
    for name, loader, weights in grid:
        C = run(loader, weights, args.cells, args.trial_ms)
        spont, mem = C[:20], C[20:]
        spont_hz = spont.sum() / 20 / args.cells / (args.trial_ms / 1000)
        tail = slice(args.trial_ms - 50, args.trial_ms)
        rest = spont[:, :, tail].sum((1, 2))
        latched = float(np.mean(mem[:, :, tail].sum((1, 2)) > 5 * np.median(rest) + 20)) if args.trial_ms >= 300 else float("nan")
        late = mem[:, :, t0 + 5:t0 + 40].sum((1, 2)).mean() - spont[:, :, t0 + 5:t0 + 40].sum((1, 2)).mean()
        accs = []
        for d in DELAYS:
            X = mem[:, :, t0 + d:t0 + d + 5].sum(2)
            accs.append(float(np.mean(cross_val_score(LogisticRegression(max_iter=3000, C=0.5), X, labels, cv=cv))))
        rows.append({"setting": name, "spont_hz": spont_hz, "late_evoked": late, "latched": latched,
                     "spont_ignited": float(np.mean(rest > 5 * np.median(rest) + 20)), "memory": dict(zip(DELAYS, accs))})
        print(f"{name:50s}{spont_hz:>9.2f}{late:>7.0f}{latched:>9.0%}  " + "  ".join(f"{a:4.0%}" for a in accs), flush=True)
    out = os.path.join(N.DATA_DIR, "results", args.out)
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print("chance for memory: 6%")


if __name__ == "__main__":
    main()
