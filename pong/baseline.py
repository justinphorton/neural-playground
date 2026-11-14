"""Game-only calculation: can a linear readout predict landing position from
the raw pulse history, with no dish? Run this before touching the simulator.

    python pong/baseline.py

For each commit height (easy / medium / hard), fit ridge regression from the
pulse history seen before the commit to the true landing x. Report error on
all balls and on balls that bounce after the commit (the only ones where the
dish could matter). Also report the follow-the-ball rule.

What we want: a commit height where the linear baseline is clearly worse on
bouncing balls than on non-bouncing ones. That's "medium". If the baseline
is perfect everywhere, the task is too easy. If it's bad everywhere, too hard.
"""

import os
import sys

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import game

COMMIT_HEIGHTS = {"easy": 0.35, "medium": 0.5, "hard": 0.65}
N_BALLS = 400   # match what the dish run can afford
PADDLE_HALF_WIDTH = 1.0 / game.GRID_W   # catch if |pred - landing| < this


def fit_and_score(X, y, bounce_mask, label):
    X_tr, X_te, y_tr, y_te, b_tr, b_te = train_test_split(
        X, y, bounce_mask, test_size=0.3, random_state=0)
    model = Ridge(alpha=1.0).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    err = np.abs(pred - y_te)
    caught = err < PADDLE_HALF_WIDTH
    print(f"  {label:<28} "
          f"err all {err.mean():.3f}  "
          f"err no-bounce {err[~b_te].mean():.3f}  "
          f"err bounce {err[b_te].mean():.3f}  "
          f"catch all {caught.mean():.0%}  "
          f"catch bounce {caught[b_te].mean():.0%}")
    return err, caught


def follow_the_ball(data):
    """Dumbest rule: paddle goes to the ball's x at the commit point."""
    pred = np.array([d["pulses"][-1][1] % game.GRID_W / game.GRID_W + 0.5 / game.GRID_W
                     for d in data])
    y = np.array([d["landing"] for d in data])
    b = np.array([d["bounces_after"] > 0 for d in data])
    err = np.abs(pred - y)
    caught = err < PADDLE_HALF_WIDTH
    print(f"  {'follow-the-ball rule':<28} "
          f"err all {err.mean():.3f}  "
          f"err no-bounce {err[~b].mean():.3f}  "
          f"err bounce {err[b].mean():.3f}  "
          f"catch all {caught.mean():.0%}  "
          f"catch bounce {caught[b].mean():.0%}")


def main():
    for name, h in COMMIT_HEIGHTS.items():
        data = game.make_dataset(N_BALLS, commit_height=h, seed=1)
        y = np.array([d["landing"] for d in data])
        bounce = np.array([d["bounces_after"] > 0 for d in data])
        n_pulses = np.mean([len(d["pulses"]) for d in data])
        print(f"\n{name.upper()}  commit at height {h}   "
              f"{bounce.mean():.0%} of balls bounce after commit   "
              f"avg {n_pulses:.1f} pulses seen")

        follow_the_ball(data)
        X_occ = np.array([game.pulses_to_matrix(d["pulses"]) for d in data])
        fit_and_score(X_occ, y, bounce, "linear on cells visited")
        X_seq = np.array([game.pulses_to_sequence(d["pulses"]) for d in data])
        fit_and_score(X_seq, y, bounce, "linear on ordered sequence")
        X_cmp = np.array([game.pulses_to_compact(d["pulses"]) for d in data])
        fit_and_score(X_cmp, y, bounce, "linear on last-6 compact")

    print("\nchance err (predict court middle):",
          f"{np.abs(y - 0.5).mean():.3f}")


if __name__ == "__main__":
    main()
