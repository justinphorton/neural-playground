"""Hidden-memory curve: probe-zap accuracy vs how long after the conditioning zap the probe comes.

    python tasks/sweep_probe.py            # 200 trials per delay, delays in parallel
    python tasks/sweep_probe.py --quick    # 40 trials per delay

Unlike sweep_memory.py, the probe zap moves with the delay, so each delay needs
its own simulations. Delays run in parallel worker processes (one culture each).

Also re-scores memory_curve.py's recordings with the same ridge readout + error
bars, so the two curves are directly comparable. Writes results/memory_both.csv
and results/memory_both.png.
"""

import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RESULTS_DIR = os.path.join(os.path.dirname(HERE), "results")

PROBE_DELAYS_MS = [2, 4, 6, 8, 10, 15, 20, 30]
WINDOW_DELAYS_MS = [0, 5, 10, 15, 20, 25, 30, 40]


def run_probe_delay(args):
    """Worker: simulate one probe delay on its own copy of the culture."""
    import contextlib
    import io

    delay, trials_per_input, seed = args
    import memory_probe as task

    task.TRIALS_PER_INPUT = trials_per_input
    with contextlib.redirect_stdout(io.StringIO()):
        env = task.build_reservoir()
        env.reseed_noise(stream=seed)
        enc, dec = task.make_pipeline(env, probe_delay_ms=float(delay))
        X, y, _ = task.collect_trials(env, enc, dec, task.make_dataset())
        env.close()
    mean, std = task.evaluate(X, y)
    total = X.sum(axis=1)
    return delay, mean, std, float(total[y == 1].mean()), float(total[y == 0].mean())


def run_window_curve(trials_per_input):
    """Re-run memory_curve.py's experiment with the ridge readout + error bars."""
    import contextlib
    import io

    import memory_curve as task
    import memory_probe  # for the shared ridge evaluate()

    task.TRIALS_PER_INPUT = trials_per_input
    with contextlib.redirect_stdout(io.StringIO()):
        env = task.build_reservoir()
        env.reseed_noise(stream=4242)
        enc, dec = task.make_pipeline(env)
        _, y, recordings = task.collect_trials(env, enc, dec, task.make_dataset())
    rows = []
    for d in WINDOW_DELAYS_MS:
        decoder = task.DelayedWindowCounts(delay_ms=float(d))
        X = np.array([decoder.features_from_channels(r, env) for r in recordings])
        mean, std = memory_probe.evaluate(X, y)
        total = X.sum(axis=1)
        rows.append((d, mean, std, float(total[y == 1].mean()), float(total[y == 0].mean())))
    env.close()
    return rows


def main():
    quick = "--quick" in sys.argv
    tpi = 20 if quick else 100
    workers = max(1, min(len(PROBE_DELAYS_MS), (os.cpu_count() or 2)))

    print(f"Probe curve: {len(PROBE_DELAYS_MS)} delays x {2 * tpi} trials, {workers} workers...")
    jobs = [(d, tpi, 1000 + i * 7919) for i, d in enumerate(PROBE_DELAYS_MS)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        probe_rows = sorted(pool.map(run_probe_delay, jobs))
    print(f"\n{'probe at':>9}  {'accuracy':>12}  {'probe spikes, zap before':>24}  {'no zap before':>13}")
    for d, m, s, a, b in probe_rows:
        print(f"{d:>6} ms  {m:>7.0%} ± {s:>3.0%}  {a:>24.1f}  {b:>13.1f}")

    print(f"\nWindow curve (memory_curve.py, ridge readout): {2 * tpi} trials...")
    window_rows = run_window_curve(tpi)
    print(f"{'window at':>9}  {'accuracy':>12}")
    for d, m, s, a, b in window_rows:
        print(f"{d:>6} ms  {m:>7.0%} ± {s:>3.0%}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "memory_both.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "delay_ms", "accuracy_mean", "accuracy_std", "spikes_zap", "spikes_nozap"])
        for r in window_rows:
            w.writerow(["window", *r])
        for r in probe_rows:
            w.writerow(["probe", *r])
    plot(window_rows, probe_rows, 2 * tpi)


def plot(window_rows, probe_rows, n_trials):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.3))
    for rows, color, label, marker in [(window_rows, "#9aa0a6", "look at an empty window (memory_curve)", "o"),
                                        (probe_rows, "#159a6a", "look at a probe zap's response (memory_probe)", "s")]:
        d = [r[0] for r in rows]
        ax1.errorbar(d, [r[1] for r in rows], yerr=[r[2] for r in rows], fmt=marker + "-", color=color, label=label, capsize=3)
    ax1.axhline(0.5, color="#bbb", ls="--", lw=1)
    ax1.set_ylim(0.4, 1.02)
    ax1.set_xlabel("ms after the zap")
    ax1.set_ylabel("accuracy: was there a zap?")
    ax1.set_title(f"Two kinds of memory ({n_trials} trials per point)")
    ax1.legend(fontsize=8, loc="upper right")

    d = [r[0] for r in probe_rows]
    ax2.plot(d, [r[3] for r in probe_rows], "s-", color="#6250d6", label="zap came before")
    ax2.plot(d, [r[4] for r in probe_rows], "s-", color="#9aa0a6", label="no zap before")
    ax2.set_xlabel("probe comes this long after the zap (ms)")
    ax2.set_ylabel("spikes in the probe's response")
    ax2.set_title("The probe burst is smaller after a zap")
    ax2.legend(fontsize=8)
    for ax in (ax1, ax2):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "memory_both.png")
    fig.savefig(path, dpi=150)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
