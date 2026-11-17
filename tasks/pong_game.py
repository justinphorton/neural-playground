"""Pong, game only (no neurons): how hard is the landing task, before any reservoir?

Run this first. It answers: for a given court and difficulty, how well can
  - the follow-the-ball rule,
  - a linear readout on the raw input (the baseline the reservoir must beat), and
  - the best possible readout (a lookup table on what the network can see)
predict where the ball lands? The interesting difficulty is where the linear
baseline is well below the best possible: that gap is the room a reservoir can win.

Geometry (all in "cell" units, so the court is N_COLS wide):
  - The ball starts at a random x, with a random sideways speed vx (cells per row).
  - It falls one row per step and bounces off the side walls.
  - The network only sees which COLUMN the ball is in at each of the first N_ROWS rows
    (one pulse per row, on that column's electrode).
  - After the last observed row the paddle has to commit. The ball then falls
    DIFFICULTY more rows before landing. More rows = more sideways travel the paddle
    can't see = more bounces = harder.

Runs in seconds with numpy only:  python tasks/pong_game.py
"""

import numpy as np

# ======================================================== SETTINGS
N_COLS = 4  # electrodes across the court (sideways resolution); set by how many electrodes reach the patch
N_ROWS = 3  # observed rows = pulses per ball (must fit in the reservoir's ~15-20 ms memory)
VMAX = 1.0  # max sideways speed, cells per row
DIFFICULTIES = [0, 1, 2, 3, 4, 6, 8]  # rows the ball falls after the commit point
N_BALLS = 200_000  # simulated balls per difficulty (cheap: no neurons)
CLEAR_PATTERN = 0.8  # a column pattern is "clear" if >= this fraction of its balls land on one side
SEED = 0


# ======================================================== GAME
def fold(x, width=None):
    """Position after bouncing off walls at 0 and `width` (a triangle wave). Width defaults to
    N_COLS at call time, so changing N_COLS (as pong_task.py does) widens the court."""
    width = N_COLS if width is None else width
    m = np.mod(x, 2 * width)
    return np.where(m <= width, m, 2 * width - m)


def column(x):
    """Which column (electrode) a position falls in."""
    return np.minimum(np.floor(x).astype(int), N_COLS - 1)


def sample_balls(n, rng):
    """Random start positions and sideways speeds."""
    x0 = rng.uniform(0, N_COLS, n)
    vx = rng.uniform(-VMAX, VMAX, n)
    return x0, vx


def observe(x0, vx):
    """What the network sees: the column at each observed row, shape (n, N_ROWS)."""
    rows = np.arange(N_ROWS)
    return column(fold(x0[:, None] + vx[:, None] * rows[None, :]))


def landing(x0, vx, difficulty):
    """Where the ball lands: DIFFICULTY rows after the last observed row."""
    return fold(x0 + vx * (N_ROWS - 1 + difficulty))


def lands_right(x_land):
    return (x_land > N_COLS / 2).astype(int)


# ======================================================== FEATURES (what a linear readout gets)
def raw_features(cols):
    """The column index at each row: N_ROWS numbers. A line on these = straight-line extrapolation."""
    return cols.astype(float)


def onehot_features(cols):
    """One input per (row, column) slot, 1 if the ball was there. A stronger linear baseline:
    it can give each slot its own weight, but still can't combine two rows."""
    out = np.zeros((len(cols), N_ROWS * N_COLS))
    for r in range(N_ROWS):
        out[np.arange(len(cols)), r * N_COLS + cols[:, r]] = 1.0
    return out


def fit_linear(X, y):
    """Least squares with an intercept (same as the repo's LinearRegression)."""
    A = np.column_stack([X, np.ones(len(X))])
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    return lambda Z: np.column_stack([Z, np.ones(len(Z))]) @ w


# ======================================================== THE BEST ANY READOUT CAN DO
def pattern_ids(cols):
    return np.ravel_multi_index(cols.T, (N_COLS,) * N_ROWS)


def lookup_table(cols, y):
    """Mean target per column pattern: the best possible prediction from what the network sees."""
    ids = pattern_ids(cols)
    n = np.bincount(ids, minlength=N_COLS ** N_ROWS)
    s = np.bincount(ids, weights=y, minlength=N_COLS ** N_ROWS)
    table = np.divide(s, n, out=np.full(len(n), np.nan), where=n > 0)
    return table, n


# ======================================================== ONE DIFFICULTY
def evaluate(difficulty, rng):
    x0, vx = sample_balls(N_BALLS, rng)
    cols = observe(x0, vx)
    x_land = landing(x0, vx, difficulty)
    side = lands_right(x_land)
    last_side = (cols[:, -1] >= N_COLS / 2).astype(int)
    opposite = side != last_side  # lands on the other half from where it was last seen

    half = N_BALLS // 2  # fit on the first half, score on the second
    tr, te = slice(0, half), slice(half, None)

    def score(pred_side):
        ok = pred_side[te] == side[te]
        return ok.mean(), ok[opposite[te]].mean() if opposite[te].any() else np.nan

    res = {"difficulty": difficulty, "opposite_frac": opposite.mean()}
    res["follow"] = score(last_side)
    for name, feats in (("linear_raw", raw_features), ("linear_onehot", onehot_features)):
        X = feats(cols)
        res[name] = score((fit_linear(X[tr], side[tr])(X) > 0.5).astype(int))
        res[name + "_mae"] = np.abs(fit_linear(X[tr], x_land[tr])(X)[te] - x_land[te]).mean()

    table, counts = lookup_table(cols[tr], side[tr].astype(float))
    p_right = table[pattern_ids(cols)]
    res["best"] = score((p_right > 0.5).astype(int))
    mean_land, _ = lookup_table(cols[tr], x_land[tr])
    res["best_mae"] = np.abs(mean_land[pattern_ids(cols)][te] - x_land[te]).mean()

    seen = counts > 0
    clear = np.maximum(table[seen], 1 - table[seen]) >= CLEAR_PATTERN
    res["patterns"] = int(seen.sum())
    res["clear_patterns"] = int(clear.sum())
    return res


def main():
    rng = np.random.default_rng(SEED)
    print(f"Court {N_COLS} columns, {N_ROWS} observed rows, |vx| <= {VMAX} cells/row\n")
    print("Accuracy predicting the landing SIDE (all balls / balls landing opposite to where last seen)")
    head = f"{'diff':>4} {'opp%':>5} | {'follow ball':>13} | {'linear raw':>13} | {'linear 1-hot':>13} | {'best possible':>13} | {'patterns (clear)':>16}"
    print(head)
    print("-" * len(head))
    results = [evaluate(d, rng) for d in DIFFICULTIES]
    f = lambda t: f"{t[0]:5.0%} / {t[1]:4.0%}" if not np.isnan(t[1]) else f"{t[0]:5.0%} /   - "
    for r in results:
        print(f"{r['difficulty']:>4} {r['opposite_frac']:5.0%} | {f(r['follow']):>13} | {f(r['linear_raw']):>13} | "
              f"{f(r['linear_onehot']):>13} | {f(r['best']):>13} | {r['patterns']:>8} ({r['clear_patterns']})")

    print("\nMean error predicting the landing POSITION, in cells (court is", N_COLS, "wide)")
    print(f"{'diff':>4} | {'linear raw':>10} | {'linear 1-hot':>12} | {'best possible':>13}")
    for r in results:
        print(f"{r['difficulty']:>4} | {r['linear_raw_mae']:10.2f} | {r['linear_onehot_mae']:12.2f} | {r['best_mae']:13.2f}")

    gap = [(r["best"][0] - r["linear_onehot"][0], r["difficulty"]) for r in results]
    g, d = max(gap)
    print(f"\nLargest gap between best possible and the 1-hot linear baseline: {g:.0%} at difficulty {d}.")
    print("Pick a difficulty near there: the reservoir has the most room to beat the baseline.")


if __name__ == "__main__":
    main()
