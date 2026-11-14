"""Pong physics for the reservoir task. No neurons here.

A ball falls through a box at fixed downward speed with a random sideways
speed, bouncing off the side walls, and lands at the bottom. The box is
divided into a grid of cells; each time the ball enters a new cell we record
a pulse event (cell, time). The paddle must commit at a chosen height, so
the pulse sequence the readout sees is cut off there.

    ball = make_ball(rng)
    pulses, landing = run_ball(ball, commit_height=0.5)
"""

import numpy as np

# ---- court -----------------------------------------------------------------
WIDTH = 1.0            # court spans x in [0, 1]
HEIGHT = 1.0           # ball falls from y = 1 (top) to y = 0 (bottom)
GRID_W = 8             # cells across
GRID_H = 6             # cells down
FALL_TIME_MS = 60.0    # every ball takes this long to fall, so trials are the same length
COMMIT_HEIGHT = 0.5    # paddle commits here. picked by pong/baseline.py at vx_max=3: a compact
                       # linear readout catches 40% of non-bouncing balls but 26% of bouncing ones
                       # (follow-the-ball 21%). That gap is where the dish can matter. vx_max=12 was
                       # tried first: every linear baseline was at chance, too hard for anyone.
DT_MS = 0.1            # physics step


def make_ball(rng, vx_max=3.0):
    """Random start x and sideways speed. vx is in court-widths per fall."""
    return {
        "x0": rng.uniform(0.05, 0.95),
        "vx": rng.uniform(-vx_max, vx_max),
    }


def simulate(ball):
    """Step the ball to the bottom. Returns arrays of (t_ms, x, y)."""
    n = int(FALL_TIME_MS / DT_MS)
    ts = np.arange(n) * DT_MS
    ys = 1.0 - ts / FALL_TIME_MS                       # fixed downward speed
    xs = np.empty(n)
    x, vx = ball["x0"], ball["vx"] / FALL_TIME_MS      # convert to per-ms
    for i in range(n):
        xs[i] = x
        x += vx * DT_MS
        if x < 0.0:
            x, vx = -x, -vx                             # bounce off left wall
        elif x > WIDTH:
            x, vx = 2 * WIDTH - x, -vx                  # bounce off right wall
    return ts, xs, ys


def cell_of(x, y):
    cx = min(int(x * GRID_W), GRID_W - 1)
    cy = min(int((1.0 - y) * GRID_H), GRID_H - 1)      # row 0 is the top
    return cy * GRID_W + cx


def run_ball(ball, commit_height=0.5):
    """Pulse events up to the commit height, plus the true landing x.

    commit_height: paddle must decide when the ball is still this high
    (1 = top, 0 = bottom). Pulses below it are not seen by the readout.
    """
    ts, xs, ys = simulate(ball)
    pulses, last = [], None
    for t, x, y in zip(ts, xs, ys):
        if y < commit_height:
            break
        c = cell_of(x, y)
        if c != last:
            pulses.append((float(t), int(c)))
            last = c
    landing = float(xs[-1])
    return pulses, landing


def bounces_after_commit(ball, commit_height):
    """How many wall bounces happen after the commit point. 0 = linear case."""
    ts, xs, ys = simulate(ball)
    vx = ball["vx"] / FALL_TIME_MS
    n = 0
    x = ball["x0"]
    for t, y in zip(ts, ys):
        x_prev = x
        x += vx * DT_MS
        if x < 0.0 or x > WIDTH:
            if y < commit_height:
                n += 1
            x = -x if x < 0 else 2 * WIDTH - x
            vx = -vx
    return n


def make_dataset(n_balls, commit_height, seed=0, vx_max=3.0):
    rng = np.random.default_rng(seed)
    balls = [make_ball(rng, vx_max) for _ in range(n_balls)]
    data = []
    for b in balls:
        pulses, landing = run_ball(b, commit_height)
        data.append({
            "ball": b,
            "pulses": pulses,
            "landing": landing,
            "bounces_after": bounces_after_commit(b, commit_height),
        })
    return data


def pulses_to_matrix(pulses, n_cells=GRID_W * GRID_H):
    """Raw-input representation for the baseline: one-hot cell occupancy
    over the whole fall, in order. Perfect memory of every pulse."""
    v = np.zeros(n_cells)
    for _, c in pulses:
        v[c] = 1.0
    return v


def pulses_to_compact(pulses, k=6):
    """Small baseline representation: x and y of the last k cells entered,
    and the time gaps between them. 3k numbers. Enough to read position,
    direction and speed without a thousand one-hot columns."""
    v = []
    last = pulses[-k:]
    for t, c in last:
        v += [c % GRID_W / GRID_W, (c // GRID_W) / GRID_H]
    gaps = [last[i + 1][0] - last[i][0] for i in range(len(last) - 1)]
    v += gaps
    while len(v) < 3 * k - 1:
        v.insert(0, 0.0)
    return np.array(v)


def pulses_to_sequence(pulses, max_len=24, n_cells=GRID_W * GRID_H):
    """Ordered version: which cell at each pulse index. Keeps order info
    so the baseline can read velocity from the sequence."""
    v = np.zeros(max_len * n_cells)
    for i, (_, c) in enumerate(pulses[:max_len]):
        v[i * n_cells + c] = 1.0
    return v
