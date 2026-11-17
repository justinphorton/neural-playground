"""Reservoir Pong, continuous: where exactly does the ball land, and does the paddle get there?

Run pong_task.py first. It saves its recorded trials to pong_trials.npz, and this
script reuses them, so it needs NO new simulation and runs in seconds.

How the trials are reused:
  The network only ever sees a ball's column pattern, so its response depends on
  the pattern (plus noise), not on the exact ball. So we:
    1. split each pattern's recorded trials 70/30 into training and test pools,
    2. train every readout on the training trials, with target = the average landing
       spot of balls with that pattern (weighted by how common the pattern is),
    3. play thousands of real balls from the game, each paired with a test-pool
       trial of its own pattern, and score them.

Every readout is linear (ridge). After each pulse k, each readout makes a fresh
guess from what it has seen so far. A slow paddle starts in the middle and moves
toward the latest guess, PADDLE_SPEED cells per row. After the last observed row
the guess is frozen (the commit point) and the paddle keeps moving toward it
until the ball lands DIFFICULTY rows later.

Readouts compared:
  follow the ball     paddle aims at the column it last saw (no prediction at all)
  linear, one-hot     linear readout on which (row, column) slots got a pulse
  RESERVOIR           linear readout on the network's spike counts
  random features     linear readout on a noise-free random nonlinear expansion (reference)
  best possible       average landing spot for each pattern seen so far (lookup table)

The key column is "bounce after commit": balls that hit a wall after the paddle
stopped seeing them. Those are the ones that need the fold.

    python tasks/pong_landing.py              # prints the scores, writes and opens the replay page
    python tasks/pong_landing.py --no-open    # same, without opening a browser
"""

import json
import os
import sys
import webbrowser

import numpy as np
from sklearn.linear_model import RidgeCV

import pong_game as game
import pong_task as task

# ======================================================== SETTINGS
N_TEST_BALLS = 4000  # real balls played against the test-pool trials
PADDLE_HALF = task.N_COLS / 8  # paddle half-width in cells: the paddle is 1/4 of the court wide
PADDLE_SPEED = task.N_COLS / 8  # cells per row (1/8 of the court): after the commit it covers DIFFICULTY * this
N_REPLAY = 60  # balls exported to the replay page (half of them bounce after the commit)
TABLE_BALLS = 400_000  # game balls used to build the best-possible lookup table
SEED = 1

W = task.N_COLS
N_ROWS = task.N_ROWS
LANDING_ROW = N_ROWS - 1 + task.DIFFICULTY
HERE = os.path.dirname(os.path.abspath(__file__))
REPLAY_TEMPLATE = os.path.join(os.path.dirname(HERE), "playground", "pong_replay.html")
REPLAY_OUT = os.path.join(os.path.dirname(HERE), "playground", "pong_replay_saved.html")


# ======================================================== DATA
def load_trials(path=task.SAVE_PATH):
    if not os.path.exists(path):
        sys.exit(f"No saved trials at {path}. Run `python tasks/pong_task.py` first.")
    d = np.load(path)
    meta = {k: d[k].tolist() for k in ("channels", "blank_ms", "row_gap_ms") if k in d}
    return d["X"], [tuple(int(c) for c in r) for r in d["inputs"]], meta


def split_pools(inputs, rng, test_frac=0.3):
    """Per pattern: which recorded trials are for training and which for testing."""
    pools = {}
    for x in set(inputs):
        idx = rng.permutation([i for i, t in enumerate(inputs) if t == x])
        n_test = max(1, int(round(test_frac * len(idx))))
        pools[x] = (idx[n_test:], idx[:n_test])
    return pools


def play_balls(n, rng):
    """Real balls from the game: start, speed, what the network sees, where they land."""
    x0, vx = game.sample_balls(n, rng)
    cols = game.observe(x0, vx)
    land = game.landing(x0, vx, task.DIFFICULTY)
    seen_end = x0 + vx * (N_ROWS - 1)  # unfolded position at the last observed row
    unfolded_land = x0 + vx * LANDING_ROW
    bounce = np.floor(seen_end / W) != np.floor(unfolded_land / W)  # hits a wall after the commit
    return x0, vx, cols, land, bounce


# ======================================================== FEATURES AFTER k PULSES
def prefix(inputs, k):
    """Column patterns with only the first k rows known (unknown rows set to 0, a constant)."""
    return [tuple(x[:k]) + (0,) * (N_ROWS - k) for x in inputs]


def features(name, k, X_trials, trial_idx, inputs):
    if name == "RESERVOIR":
        return task.row_prefix(X_trials[trial_idx], k)
    if name == "linear, one-hot":
        return task.onehot_features(prefix(inputs, k))
    if name == "random features":
        return task.random_features(prefix(inputs, k))
    raise ValueError(name)


# ======================================================== THE READOUTS
def best_tables(rng):
    """Average landing spot for each pattern prefix (k = 1..N_ROWS), from many game balls."""
    x0, vx, cols, land, _ = play_balls(TABLE_BALLS, rng)
    tables = {}
    for k in range(1, N_ROWS + 1):
        c = np.array(prefix([tuple(r) for r in cols], k))
        tables[k], _ = game.lookup_table(c, land)
    return tables


def predict_all(X_trials, inputs, pools, balls, tables, rng):
    """Predicted landing spot after each pulse, for every readout: {name: (n_balls, N_ROWS)}."""
    x0, vx, cols, land, bounce = balls
    ball_inputs = [tuple(int(c) for c in r) for r in cols]
    test_trial = np.array([rng.choice(pools[x][1]) for x in ball_inputs])

    train_idx = np.concatenate([pools[x][0] for x in pools])
    train_inputs = [inputs[i] for i in train_idx]
    # Train target: the average landing of each training trial's pattern; weight each
    # trial so every pattern counts as often as it occurs in the game.
    target = tables[N_ROWS][game.pattern_ids(np.array(train_inputs))]
    freq = task.FREQUENCY
    n_per = {x: len(pools[x][0]) for x in pools}
    weight = np.array([freq.get(x, 0.0) / n_per[x] for x in train_inputs])
    weight *= len(weight) / weight.sum()

    preds = {}
    seen_col_centre = cols + 0.5
    preds["follow the ball"] = seen_col_centre.astype(float)
    for name in ("linear, one-hot", "RESERVOIR", "random features"):
        P = np.zeros((len(land), N_ROWS))
        for k in range(1, N_ROWS + 1):
            Xtr = features(name, k, X_trials, train_idx, train_inputs)
            readout = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Xtr, target, sample_weight=weight)
            P[:, k - 1] = readout.predict(features(name, k, X_trials, test_trial, ball_inputs))
        preds[name] = np.clip(P, 0, W)
    preds["best possible"] = np.column_stack(
        [tables[k][game.pattern_ids(np.array(prefix(ball_inputs, k)))] for k in range(1, N_ROWS + 1)])

    # Control: reservoir readout trained on shuffled targets (should do no better than guessing the middle).
    Xtr = features("RESERVOIR", N_ROWS, X_trials, train_idx, train_inputs)
    shuffled = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Xtr, rng.permutation(target), sample_weight=weight)
    control = shuffled.predict(features("RESERVOIR", N_ROWS, X_trials, test_trial, ball_inputs))
    return preds, control, test_trial


# ======================================================== THE PADDLE
def paddle_paths(P):
    """Paddle position at every row, 0..LANDING_ROW. The guess made after pulse k is
    available from row k-1 on; after the last observed row it stays frozen."""
    pos = np.full(len(P), W / 2)
    path = [pos.copy()]
    for row in range(LANDING_ROW):
        aim = P[:, min(row, N_ROWS - 1)]
        pos = pos + np.clip(aim - pos, -PADDLE_SPEED, PADDLE_SPEED)
        path.append(pos.copy())
    return np.stack(path, axis=1)


# ======================================================== REPORT
def report(preds, control, balls):
    land, bounce = balls[3], balls[4]
    print(f"\n{len(land)} test balls, {bounce.mean():.0%} bounce after the commit point. "
          f"Paddle {2 * PADDLE_HALF:g} of {W} cells wide, {PADDLE_SPEED:g} cells/row.\n")
    head = (f"{'readout':<18}" + "".join(f"{'err k=' + str(k):>9}" for k in range(1, N_ROWS + 1))
            + f"{'catch':>8}{'catch, bounce':>15}{'catch, no bounce':>18}")
    print(head)
    print("-" * len(head))
    paths, catches = {}, {}
    for name, P in preds.items():
        paths[name] = paddle_paths(P)
        catches[name] = np.abs(paths[name][:, -1] - land) <= PADDLE_HALF
        errs = "".join(f"{np.abs(P[:, k] - land).mean():9.2f}" for k in range(N_ROWS))
        c = catches[name]
        print(f"{name:<18}{errs}{c.mean():8.0%}{c[bounce].mean():15.0%}{c[~bounce].mean():18.0%}")
    print(f"\ncontrol: reservoir trained on shuffled targets, error after all pulses "
          f"{np.abs(control - land).mean():.2f} (guessing the middle every time: {np.abs(W / 2 - land).mean():.2f})")
    print("errors are mean |predicted - actual landing| in cells; k = pulses seen")
    return paths, catches


def export_replay(preds, paths, catches, balls, test_trial, meta, rng, open_browser=True):
    x0, vx, cols, land, bounce = balls
    b_idx, nb_idx = np.flatnonzero(bounce), np.flatnonzero(~bounce)
    half = N_REPLAY // 2
    pick = np.concatenate([rng.choice(b_idx, min(half, len(b_idx)), replace=False),
                           rng.choice(nb_idx, min(N_REPLAY - half, len(nb_idx)), replace=False)])
    rng.shuffle(pick)
    r = lambda a: np.round(np.asarray(a, float), 3).tolist()
    data = {
        "meta": {"cols": W, "rows": N_ROWS, "landing_row": LANDING_ROW, "difficulty": task.DIFFICULTY,
                 "paddle_half": PADDLE_HALF, "paddle_speed": PADDLE_SPEED,
                 "row_times_ms": task.ROW_TIMES_MS, "toy": bool(meta.get("toy", False)),
                 "channels": meta.get("channels"), "blank_ms": meta.get("blank_ms")},
        "summary": {name: {"catch": float(c.mean()), "catch_bounce": float(c[bounce].mean()),
                           "catch_no_bounce": float(c[~bounce].mean())} for name, c in catches.items()},
        "balls": [{"x0": float(x0[i]), "vx": float(vx[i]), "cols": cols[i].tolist(), "land": float(land[i]),
                   "bounce": bool(bounce[i]), "trial": int(test_trial[i]),
                   "pred": {n: r(P[i]) for n, P in preds.items()},
                   "path": {n: r(p[i]) for n, p in paths.items()},
                   "caught": {n: bool(c[i]) for n, c in catches.items()}} for i in pick],
    }
    with open(REPLAY_TEMPLATE, encoding="utf-8") as f:
        page = f.read().replace("__PONG_DATA__", json.dumps(data))
    with open(REPLAY_OUT, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"\nReplay page: {REPLAY_OUT}")
    if open_browser:
        webbrowser.open("file://" + REPLAY_OUT)


def main(X_trials=None, inputs=None, meta=None, open_browser=True):
    if X_trials is None:
        X_trials, inputs, meta = load_trials()
    rng = np.random.default_rng(SEED)
    pools = split_pools(inputs, rng)
    tables = best_tables(rng)
    balls = play_balls(N_TEST_BALLS, rng)
    keep = np.array([tuple(int(c) for c in r) in pools for r in balls[2]])  # drop patterns never recorded (rare)
    balls = tuple(a[keep] for a in balls)
    preds, control, test_trial = predict_all(X_trials, inputs, pools, balls, tables, rng)
    paths, catches = report(preds, control, balls)
    export_replay(preds, paths, catches, balls, test_trial, meta or {}, rng, open_browser)


if __name__ == "__main__":
    main(open_browser="--no-open" not in sys.argv)
