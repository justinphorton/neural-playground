"""Screen one Pong reservoir configuration: record each ball once, score it with and without a 2 ms blind window.

usage:
    python tasks/pong_screen.py NAME N_CELLS 'WEIGHTS_JSON' ELECTRODES_PER_COLUMN TRIALS_PER_INPUT [N_ROWS ROW_GAP_MS DIFFICULTY]

examples:
    python tasks/pong_screen.py base 300 '{}' 2 15
    python tasks/pong_screen.py ei 300 '{"EXC_EXC-dend-AMPA-weight": 50, "EXC_EXC-dend-NMDA-weight": 20, "INH_EXC-soma-AMPA-weight": 28}' 2 15
    python tasks/pong_screen.py rows4 300 '{...}' 2 15 4 4 4

Weight names (livn model defaults): EXC_EXC-dend-AMPA-weight 7, EXC_EXC-dend-NMDA-weight 0,
EXC_INH-soma-GABA_A-weight 5 (inhibition onto excitatory cells), INH_EXC-soma-AMPA-weight 7
(excitation onto inhibitory cells). Names read TARGET_SOURCE.

Prints one JSON line with test accuracy, in-game accuracy and spikes per ball for the
reservoir (no blind window and 2 ms blind window), the one-hot baseline and a shuffled-label
control; saves the recorded features to pong_screen_NAME.npz in the repo folder. Uses 2 processes.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ARGS = sys.argv[1:]
NAME, N_CELLS, WEIGHTS, EPC, TPI = ARGS[0], int(ARGS[1]), json.loads(ARGS[2]), int(ARGS[3]), int(ARGS[4])
N_ROWS, ROW_GAP, DIFFICULTY = (int(ARGS[5]), float(ARGS[6]), int(ARGS[7])) if len(ARGS) > 5 else (3, 5.0, 4)
WORKERS = 2


def configure(T):
    """Point pong_task at this configuration (the game patterns depend on rows and difficulty)."""
    T.N_CELLS, T.WEIGHTS, T.ELECTRODES_PER_COLUMN, T.TRIALS_PER_INPUT = N_CELLS, WEIGHTS, EPC, TPI
    T.N_ROWS, T.ROW_GAP_MS, T.DIFFICULTY = N_ROWS, ROW_GAP, DIFFICULTY
    T.ROW_TIMES_MS = [T.FIRST_PULSE_MS + r * ROW_GAP for r in range(N_ROWS)]
    T.TRIAL_MS = int(T.FIRST_PULSE_MS + ROW_GAP * (N_ROWS - 1) + T.WINDOW_MS)
    T.INPUTS, T.LABELS, T.PURITY, T.FREQUENCY = T._game_patterns()


def work(args):
    k, chunk = args
    import pong_task as T
    configure(T)
    np.random.seed(2000 + k)
    env = T.build_reservoir()
    enc, dec = T.make_pipeline(env)
    dec.duration = T.TRIAL_MS
    X0, X2 = [], []
    for x in chunk:
        env.clear()
        dec.blank_ms = 0.0
        r = env(dec, inputs=x, encoding=enc)
        X0.append(r["features"])
        sig = type("S", (), {"spike_ids": r["spike_ids"], "spike_times": r["spike_times"]})
        dec.blank_ms = 2.0  # same recording, decoded again with the blind window
        X2.append(dec(sig, env)["features"])
    return np.array(X0), np.array(X2), enc.channels


def main():
    import pong_task as T
    configure(T)
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import train_test_split

    ds = T.make_dataset()
    t0 = time.time()
    print(f"{NAME}: {len(ds)} balls, {len(T.INPUTS)} patterns", flush=True)
    with Pool(WORKERS) as p:
        res = p.map(work, [(k, ds[k::WORKERS]) for k in range(WORKERS)])
    X = {}
    for j, b in enumerate((0.0, 2.0)):
        M = np.zeros((len(ds), res[0][j].shape[1]))
        for k in range(WORKERS):
            M[k::WORKERS] = res[k][j]
        X[b] = M

    y = np.array([T.target(x) for x in ds])
    pat = np.array([T.INPUTS.index(x) for x in ds])
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=pat, random_state=0)
    ds_te = [ds[i] for i in te]
    ridge = lambda: RidgeCV(alphas=np.logspace(-1, 5, 7))

    def score(F, lab=y):
        pred = (ridge().fit(F[tr], lab[tr]).predict(F[te]) > 0.5).astype(int)
        return float(np.mean(pred == lab[te])), float(T.game_accuracy(ds_te, pred == y[te]))

    rng = np.random.default_rng(0)
    out = {"name": NAME, "cells": N_CELLS, "weights": WEIGHTS, "electrodes_per_col": EPC, "rows": N_ROWS,
           "row_gap_ms": ROW_GAP, "difficulty": DIFFICULTY, "balls": len(ds), "patterns": len(T.INPUTS),
           "minutes": round((time.time() - t0) / 60, 1), "channels": res[0][2],
           "game_ceiling": round(sum(T.FREQUENCY[x] * T.PURITY[x] for x in T.INPUTS), 3)}
    acc, game = score(T.onehot_features(ds))
    out["one_hot"] = {"test": round(acc, 3), "game": round(game, 3)}
    for b in (0.0, 2.0):
        F = X[b][:, X[b].std(0) > 0]
        acc, game = score(F)
        shuf = []
        for _ in range(10):
            ys = y.copy()
            ys[tr] = rng.permutation(y[tr])
            shuf.append(np.mean((ridge().fit(F[tr], ys[tr]).predict(F[te]) > 0.5) == y[te]))
        out[f"blank{b:g}"] = {"test": round(acc, 3), "game": round(game, 3),
                              "spikes_per_ball": round(float(X[b].sum(1).mean()), 1),
                              "shuffled_mean": round(float(np.mean(shuf)), 3),
                              "shuffled_95": round(float(np.percentile(shuf, 95)), 3)}
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    np.savez_compressed(os.path.join(repo, f"pong_screen_{NAME}.npz"), X0=X[0.0], X2=X[2.0], y=y,
                        inputs=np.asarray(ds), channels=np.asarray(res[0][2]))
    print(json.dumps(out, default=float), flush=True)


if __name__ == "__main__":
    main()
