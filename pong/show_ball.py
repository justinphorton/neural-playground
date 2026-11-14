"""Draw one ball: its path, the pulses it sent to the court, and what the dish did.

    python pong/show_ball.py            # random ball, saves results/ball_<seed>.png
    python pong/show_ball.py 7          # ball from seed 7
    python pong/show_ball.py 7 --nodish # path and pulses only, no simulation (fast)

Three panels: the court with the ball's path and the commit line, the pulse
table the dish receives (cell vs time), and the spike raster from the dish.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tasks"))
import game  # noqa: E402

RESULTS = os.path.join(os.path.dirname(HERE), "results")


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0
    nodish = "--nodish" in sys.argv
    rng = np.random.default_rng(seed)
    ball = game.make_ball(rng)
    pulses, landing = game.run_ball(ball, game.COMMIT_HEIGHT)
    ts, xs, ys = game.simulate(ball)
    bounces = game.bounces_after_commit(ball, game.COMMIT_HEIGHT)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_panels = 2 if nodish else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5))

    # ---- panel 1: the court
    ax = axes[0]
    for i in range(game.GRID_W + 1):
        ax.axvline(i / game.GRID_W, color="#ddd", lw=0.8)
    for j in range(game.GRID_H + 1):
        ax.axhline(j / game.GRID_H, color="#ddd", lw=0.8)
    seen = ys >= game.COMMIT_HEIGHT
    ax.plot(xs[seen], ys[seen], color="#6250d6", lw=2, label="seen by dish")
    ax.plot(xs[~seen], ys[~seen], color="#bbb", lw=2, ls="--", label="after commit (hidden)")
    ax.axhline(game.COMMIT_HEIGHT, color="#c3651b", lw=1.5, label="commit line")
    ax.plot([landing], [0], "v", color="#159a6a", ms=12, label=f"lands at x={landing:.2f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"ball {seed}: vx={ball['vx']:+.1f}, {bounces} bounce(s) after commit")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_xticks([]); ax.set_yticks([])

    # ---- panel 2: pulses the dish gets
    ax = axes[1]
    for t, c in pulses:
        ax.plot([t], [c], "s", color="#6250d6", ms=6)
    ax.set_xlim(0, game.FALL_TIME_MS / 2 + 2)
    ax.set_ylim(-1, game.GRID_W * game.GRID_H)
    ax.set_xlabel("time (ms)"); ax.set_ylabel("court cell (0 = top left)")
    ax.set_title(f"{len(pulses)} cell entries → pulses")
    ax.grid(alpha=.3)

    # ---- panel 3: dish spikes
    if not nodish:
        import pong_task as T
        env = T.build_reservoir()
        enc, dec = T.make_pipeline(env)
        env.clear()
        out = env(dec, inputs={"pulses": pulses, "landing": landing, "bounces_after": bounces}, encoding=enc)
        ax = axes[2]
        ax.plot(out["spike_times"], out["spike_ids"], "|", color="#159a6a", ms=4, mew=0.8)
        for t, _ in enc.pulses({"pulses": pulses}):
            ax.axvline(t, color="#6250d6", alpha=.25, lw=1)
        last = pulses[-1][0]
        ax.axvspan(last - T.PRE_MS, last + T.POST_MS, color="#c3651b", alpha=.15, label="end-of-fall readout window")
        ax.set_xlim(0, 45); ax.set_xlabel("time (ms)"); ax.set_ylabel("neuron")
        ax.set_title(f"dish response: {len(out['spike_times'])} spikes")
        ax.legend(fontsize=7, loc="upper right")
        env.close()

    fig.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"ball_{seed}.png")
    fig.savefig(path, dpi=130)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
