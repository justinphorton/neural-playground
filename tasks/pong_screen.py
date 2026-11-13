"""Screen one Pong reservoir configuration: record each ball once, score it with and without a 2 ms blind window.
usage: python tasks/pong_screen.py NAME N_CELLS AMPA NMDA ELECTRODES_PER_COLUMN TRIALS_PER_INPUT"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
import numpy as np

name, n_cells, ampa, nmda, epc, tpi = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])

def configure(T):
    T.N_CELLS = n_cells; T.ELECTRODES_PER_COLUMN = epc; T.TRIALS_PER_INPUT = tpi
    T.WEIGHTS = {} if (ampa == 7 and nmda == 0) else {"EXC_EXC-dend-AMPA-weight": ampa, "EXC_EXC-dend-NMDA-weight": nmda}

def work(args):
    k, chunk = args
    import pong_task as T; configure(T)
    np.random.seed(2000 + k)
    env = T.build_reservoir(); enc, dec = T.make_pipeline(env)
    out = {0.0: [], 2.0: []}
    for x in chunk:
        env.clear()
        dec.blank_ms = 0.0
        r = env(dec, inputs=x, encoding=enc)
        out[0.0].append(r["features"])
        sig = type("S", (), {"spike_ids": r["spike_ids"], "spike_times": r["spike_times"]})
        dec.blank_ms = 2.0
        out[2.0].append(dec(sig, env)["features"])
    return np.array(out[0.0]), np.array(out[2.0]), enc.channels

if __name__ == "__main__":
    import pong_task as T; configure(T)
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import train_test_split
    ds = T.make_dataset(); t0 = time.time(); print("start", len(ds), flush=True)
    with Pool(2) as p:
        res = p.map(work, [(k, ds[k::2]) for k in range(2)])
    X = {}
    for j, b in enumerate((0.0, 2.0)):
        M = np.zeros((len(ds), res[0][j].shape[1]))
        for k in range(2): M[k::2] = res[k][j]
        X[b] = M
    y = np.array([T.target(x) for x in ds]); pat = np.array([T.INPUTS.index(x) for x in ds])
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=pat, random_state=0)
    fit = lambda F, lab: np.mean((RidgeCV(alphas=np.logspace(-1, 5, 7)).fit(F[tr], lab[tr]).predict(F[te]) > .5) == lab[te])
    rng = np.random.default_rng(0); summary = {"name": name, "cells": n_cells, "ampa": ampa, "nmda": nmda, "electrodes_per_col": epc,
                                              "balls": len(ds), "minutes": round((time.time() - t0) / 60, 1), "channels": res[0][2]}
    summary["one_hot"] = fit(T.onehot_features(ds), y)
    for b in (0.0, 2.0):
        F = X[b][:, X[b].std(0) > 0]
        shuf = []
        for _ in range(15):
            ys = y.copy(); ys[tr] = rng.permutation(y[tr])
            shuf.append(np.mean((RidgeCV(alphas=np.logspace(-1, 5, 7)).fit(F[tr], ys[tr]).predict(F[te]) > .5) == y[te]))
        summary[f"blank{b:g}"] = {"reservoir": round(fit(F, y), 3), "spikes_per_ball": round(X[b].sum(1).mean(), 1),
                                  "shuffled_mean": round(np.mean(shuf), 3), "shuffled_95": round(np.percentile(shuf, 95), 3)}
    np.savez_compressed(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f"pong_screen_{name}.npz"), X0=X[0.0], X2=X[2.0], y=y, inputs=np.asarray(ds), channels=np.asarray(res[0][2]))
    print(json.dumps(summary, default=float), flush=True)
