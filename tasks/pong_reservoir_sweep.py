"""How much memory and mixing does the team-spec Pong need? A software reservoir answers it.

    python tasks/pong_reservoir_sweep.py

Same falling-ball game, same 4 x 4 grid and pulse rule as tasks/pong_spec.py, but the
pulses drive a software echo state network (ESN) instead of the livn culture. The ESN's
memory and mixing are knobs, so we can find what a network must have for a straight-line
readout to beat the straight line on the raw pulse history.

Knobs:
  leak      how much of the old state each frame keeps (memory length: ~1/leak frames)
  radius    spectral radius of the recurrent weights (0 = no recurrence, no mixing
            between units; ~1 = rich mixing and long memory)
  frame_ms  converts frames to milliseconds for comparison with the culture (2 ms/frame)
"""

import itertools
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pong_spec import COLS, COMMIT, FRAME_MS, ROWS, history_features, make_balls, pulse_train  # noqa: E402

N_UNITS = 300  # same size as the culture patch
N_BALLS = 2000
TRAIN_FRACTION = 0.7
NOISE = 0.02  # small state noise, so the ESN isn't unrealistically clean


def esn_states(trains, leak, radius, seed=0):
    """Run each ball's pulses through the ESN; return the state at the commit point."""
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1, (N_UNITS, N_UNITS)) * (rng.random((N_UNITS, N_UNITS)) < 0.1)
    eig = np.max(np.abs(np.linalg.eigvals(W)))
    W = W * (radius / eig) if radius > 0 else W * 0
    W_in = rng.normal(0, 1, (N_UNITS, ROWS * COLS))
    bias = rng.normal(0, 0.2, N_UNITS)
    noise = np.random.default_rng(seed + 1)
    U = np.zeros((COMMIT, len(trains), ROWS * COLS))  # every ball's input, frame by frame
    for i, train in enumerate(trains):
        for f, r, c in train:
            U[f, i, r * COLS + c] = 1.0
    x = np.zeros((len(trains), N_UNITS))  # all balls step together
    for f in range(COMMIT):
        x = (1 - leak) * x + leak * np.tanh(x @ W.T + U[f] @ W_in.T + bias)
        x += noise.normal(0, NOISE, x.shape)
    return x


def side_and_catch(pred, xs, test):
    land = xs[test, -1]
    return float(np.mean((pred > 0.5) == (land > 0.5))), float(np.mean(np.abs(pred - land) <= 0.15))


def line(X, y, train, test):
    return RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000]).fit(X[train], y[train]).predict(X[test])


def memory_frames(leak):
    """Frames until a pulse's trace in the state decays to ~1/e (leak-only estimate)."""
    return 1.0 / leak


def main():
    print(f"{'width':>5} {'leak':>5} {'radius':>6} {'memory':>9} {'ESN + line':>11} {'line on history':>16} {'ceiling':>8}")
    rows = []
    for width in [1, 2, 4]:
        xs = make_balls(width, N_BALLS)
        trains = [pulse_train(x) for x in xs]
        y = xs[:, -1]
        cut = int(N_BALLS * TRAIN_FRACTION)
        train, test = np.arange(N_BALLS) < cut, np.arange(N_BALLS) >= cut
        H = history_features(trains)
        base, _ = side_and_catch(line(H, y, train, test), xs, test)
        ceiling, _ = side_and_catch(HistGradientBoostingRegressor(max_iter=200).fit(H[train], y[train]).predict(H[test]),
                                    xs, test)
        for leak, radius in itertools.product([1.0, 0.5, 0.25, 0.1, 0.05], [0.0, 0.5, 0.9, 1.1]):
            X = esn_states(trains, leak, radius)
            acc, catch = side_and_catch(line(X, y, train, test), xs, test)
            mem_ms = memory_frames(leak) * FRAME_MS
            rows.append((width, leak, radius, mem_ms, acc, catch, base, ceiling))
            print(f"{width:>5} {leak:>5} {radius:>6} {mem_ms:>7.0f}ms {acc:>11.0%} {base:>16.0%} {ceiling:>8.0%}", flush=True)
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "pong_reservoir_sweep.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("width,leak,radius,memory_ms,esn_side,esn_catch,line_on_history_side,ceiling_side\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
